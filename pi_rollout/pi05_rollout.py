#!/usr/bin/env python3
"""Smoothed Pi0.5 remote rollout runner (chunk blending, RTC blending, temporal aggregation, action interpolation).

Reuses the protected capture, image-contract, transport, and recording modules
from ``src/`` while keeping all new Pi0.5 smoothing logic under ``pi_rollout/``.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time
from types import SimpleNamespace
import uuid

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (str(REPO_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from inference import policy_chunks
from inference.inference_remote import (
    OPENPI_MAX_RTC_PREFIX_STEPS,
    PolicyRuntime,
    discover_episodes,
    open_episode,
    read_robot_array,
    row_metadata,
)
from inference.policy_transport import WebsocketOpenPIClient, timing_now, wait_until_loop_deadline
from pi_rollout.smoothing import (
    ActionGapInterpolator,
    BLENDING_METHODS,
    TEMPORAL_AGG_DECAY_K,
    TemporalAggregator,
    blend_actions,
    blend_rtc_boundary,
)


def _apply_chunk_blending(
    prev_actions: np.ndarray,
    new_actions: np.ndarray,
    *,
    elapsed_in_prev: int,
    prev_chunk_start: int,
    new_chunk_start: int,
    mixing_steps: int,
    blending_method: str,
    rtc_enabled: bool,
    rtc_boundary: int,
) -> np.ndarray:
    """Apply RTC boundary crossfade or standard chunk blending on handoff."""
    if blending_method == "none" or mixing_steps <= 0:
        return new_actions
    if rtc_enabled:
        prev_next_index = int(elapsed_in_prev) + int(prev_chunk_start)
        if (
            prev_next_index + rtc_boundary + mixing_steps <= prev_actions.shape[1]
            and rtc_boundary + mixing_steps <= new_actions.shape[1]
        ):
            return blend_rtc_boundary(
                prev_actions,
                new_actions,
                prev_next_index=prev_next_index,
                boundary=rtc_boundary,
                steps=mixing_steps,
                interp_type=blending_method,
            )
    return blend_actions(
        prev_actions,
        new_actions,
        mixing_steps=mixing_steps,
        query_frequency=elapsed_in_prev,
        prev_chunk_start_index=prev_chunk_start,
        new_chunk_start_index=new_chunk_start,
        interp_type=blending_method,
    )


class SmoothedPolicySession:
    """Remote Pi0.5 session with chunk blending, RTC blending, and temporal aggregation."""

    def __init__(
        self,
        runtime: PolicyRuntime,
        *,
        blending_method: str = "sigmoid",
        mixing_steps: int = 5,
        temporal_agg: bool = False,
        temporal_agg_k: float = TEMPORAL_AGG_DECAY_K,
        action_interp: bool = False,
        masking_strategy: str | None = None,
        masking_config: str | Path | None = None,
    ) -> None:
        if blending_method not in BLENDING_METHODS:
            raise ValueError(f"blending_method must be one of {BLENDING_METHODS}, got {blending_method!r}")
        if mixing_steps < 0:
            raise ValueError("mixing_steps must be nonnegative")
        if temporal_agg:
            if runtime.args.rtc:
                raise ValueError("Temporal aggregation cannot be combined with RTC; disable rtc to use temporal_agg")
            if blending_method != "none":
                raise ValueError("Temporal aggregation replaces chunk blending; set blending_method='none'")

        self.runtime = runtime
        self.args = runtime.args
        self.blending_method = blending_method
        self.mixing_steps = int(mixing_steps)
        self.temporal_agg = bool(temporal_agg)
        self.temporal_agg_k = float(temporal_agg_k)
        self.action_interp = bool(action_interp)

        self.worker = None
        if self.args.async_inference and not self.temporal_agg:
            self.worker = policy_chunks.AsyncOpenPIQueryWorker(
                policy_client=runtime.client,
                timing_stats=runtime.timing,
                query_actions=runtime.query,
                max_delay_s=self.args.async_prefetch_steps / self.args.fps,
            )
            self.worker.start()

        self.aggregator = (
            TemporalAggregator(
                num_queries=int(runtime.horizon),
                max_timesteps=int(self.args.max_timesteps),
                action_dim=int(runtime.action_dim),
                decay_k=self.temporal_agg_k,
            )
            if self.temporal_agg
            else None
        )
        self.gap_interpolator = (
            ActionGapInterpolator(is_double_arm=not runtime.task["single_arm"])
            if self.action_interp
            else None
        )
        self.masker = None
        if masking_strategy:
            from utils.masking import Masker

            self.masker = Masker(masking_strategy, masking_config)
        self.generation = 0
        self.reset()

    def reset(self) -> None:
        self.generation += 1
        if self.worker:
            self.worker.reset()
        if self.aggregator:
            self.aggregator.reset()
        self.actions = None
        self.full_actions = None
        self.pending = None
        self.step = 0
        self.chunk_step = 0
        self.chunk_start = 0
        self.update_step = 0
        self.queries = 0
        self.prev_action = None
        self.prev_prev_action = None

    def close(self) -> None:
        if self.worker:
            self.worker.stop()

    def _merge_with_blending(
        self,
        new_chunk: np.ndarray,
        new_chunk_start: int,
        elapsed_in_prev: int,
        *,
        rtc_boundary: int = 0,
    ) -> np.ndarray:
        prev_plan = self.full_actions if self.full_actions is not None else self.actions
        if prev_plan is None:
            return new_chunk
        return _apply_chunk_blending(
            prev_plan,
            new_chunk,
            elapsed_in_prev=elapsed_in_prev,
            prev_chunk_start=self.chunk_start,
            new_chunk_start=new_chunk_start,
            mixing_steps=self.mixing_steps,
            blending_method=self.blending_method,
            rtc_enabled=bool(self.args.rtc),
            rtc_boundary=rtc_boundary,
        )

    def _policy_inputs(self, observation) -> tuple[dict[str, np.ndarray], np.ndarray]:
        if self.masker is None:
            return observation.images, observation.qpos
        return self.masker.mask(observation.images, observation.qpos)

    def tick(self, observe, prompt: str) -> np.ndarray | None:
        if self.temporal_agg:
            observation = observe()
            images, qpos = self._policy_inputs(observation)
            actions, _, _ = self.runtime.query(
                self.runtime.client,
                images,
                qpos,
                prompt,
                query_index=self.queries,
                rtc_metadata={},
            )
            self.queries += 1
            self.aggregator.add_chunk(self.step, actions)
            return self.aggregator.select(self.step)

        if self.worker:
            self.pending = policy_chunks.poll_async_result(
                self.worker,
                self.pending,
                current_chunk_step=self.chunk_step,
                log_prefix="[Pi0.5]",
                expected_goal_generation=self.generation,
            )
            if self.actions is None and self.pending is not None:
                raw_chunk = np.asarray(self.pending["action_chunk"])
                self.full_actions = raw_chunk.copy()
                self.actions, self.chunk_start = policy_chunks.merge_sync_openpi_chunk(
                    new_action_chunk=raw_chunk,
                    current_chunk_size=self.args.query_frequency,
                )
                self.pending = None
                self.update_step = self.step
            elif self.actions is not None and self.pending is not None:
                prev_plan = self.full_actions if self.full_actions is not None else self.actions
                prev_chunk_start = self.chunk_start
                prev_update_step = self.update_step
                raw_new = np.asarray(self.pending["action_chunk"])
                (
                    pending_after,
                    merged_actions,
                    merged_start,
                    merged_update_step,
                    merged_chunk_step,
                    *_,
                ) = policy_chunks.maybe_merge_async_result(
                    self.args,
                    pending_async_result=self.pending,
                    all_actions=self.actions,
                    t=self.step,
                    current_chunk_step=self.chunk_step,
                    current_chunk_update_t=prev_update_step,
                    current_chunk_start_index=prev_chunk_start,
                    async_prefetch_steps=self.args.async_prefetch_steps,
                    timing_stats=self.runtime.timing,
                    log_prefix="[Pi0.5]",
                )
                if merged_update_step != prev_update_step:
                    rtc_boundary = policy_chunks.rtc_prefix_length(self.pending.get("rtc_metadata"))
                    elapsed = self.step - prev_update_step
                    merged_actions = _apply_chunk_blending(
                        prev_plan,
                        merged_actions,
                        elapsed_in_prev=elapsed,
                        prev_chunk_start=prev_chunk_start,
                        new_chunk_start=merged_start,
                        mixing_steps=self.mixing_steps,
                        blending_method=self.blending_method,
                        rtc_enabled=bool(self.args.rtc),
                        rtc_boundary=rtc_boundary,
                    )
                    self.full_actions = raw_new.copy()
                self.pending = pending_after
                self.actions = merged_actions
                self.chunk_start = merged_start
                self.update_step = merged_update_step
                self.chunk_step = merged_chunk_step

            query = policy_chunks.should_submit_async_query(
                self.worker,
                all_actions=self.actions,
                pending_async_result=self.pending,
                current_chunk_step=self.chunk_step,
                current_chunk_start_index=self.chunk_start,
                async_prefetch_steps=self.args.async_prefetch_steps,
                force_submit=self.actions is None,
            )
        else:
            query = self.actions is None or self.step % self.args.query_frequency == 0

        if query:
            observation = observe()
            images, qpos = self._policy_inputs(observation)
            prefix = policy_chunks.get_unexecuted_chunk_suffix(
                self.full_actions if self.full_actions is not None else self.actions,
                current_chunk_step=self.chunk_step,
                current_chunk_start_index=self.chunk_start,
            )
            rtc = self.runtime.rtc_metadata(prefix)
            if self.worker:
                self.worker.submit(
                    image_dict=images,
                    qpos=qpos,
                    prompt=prompt,
                    query_index=self.queries,
                    submit_chunk_step=self.chunk_step,
                    submit_rollout_index=self.step,
                    rtc_metadata=rtc,
                    goal_generation=self.generation,
                )
            else:
                actions, _, _ = self.runtime.query(
                    self.runtime.client,
                    images,
                    qpos,
                    prompt,
                    query_index=self.queries,
                    rtc_metadata=rtc,
                )
                merged, next_start = policy_chunks.merge_sync_openpi_chunk(
                    new_action_chunk=actions,
                    current_chunk_size=self.args.query_frequency,
                )
                if self.actions is not None:
                    merged = self._merge_with_blending(
                        merged,
                        next_start,
                        elapsed_in_prev=self.step - self.update_step,
                        rtc_boundary=policy_chunks.rtc_prefix_length(rtc),
                    )
                self.full_actions = np.asarray(actions).copy()
                self.actions, self.chunk_start = merged, next_start
                self.update_step, self.chunk_step = self.step, 0
            self.queries += 1

        if self.actions is None:
            return None
        return policy_chunks.select_action_from_openpi_chunk(
            all_actions=self.actions,
            elapsed_from_chunk_update=self.step - self.update_step,
            chunk_start_index=self.chunk_start,
        )

    def build_gap_steps(self, action: np.ndarray, cur_qpos: np.ndarray) -> list[np.ndarray]:
        if self.gap_interpolator is None or self.prev_action is None or self.prev_prev_action is None:
            return []
        return self.gap_interpolator.build_steps(
            action, cur_qpos, self.prev_action, self.prev_prev_action
        )

    def committed(self, action: np.ndarray | None = None) -> None:
        if action is not None:
            self.prev_prev_action = self.prev_action
            self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self.step += 1
        self.chunk_step += 1


def load_pi05_args(config_path: str | Path, cli_args: argparse.Namespace) -> argparse.Namespace:
    """Build a PolicyRuntime-compatible Namespace from YAML + CLI smoothing overrides."""
    from config.paths import INFERENCE_RAW_DIR, ROLLOUT_DIR, DAGGER_DIR
    from config.task_config import select_task_config

    doc = yaml.safe_load(Path(config_path).read_text()) or {}
    task_name = doc.get("task", "piperx_single")
    task = select_task_config(task_name, None)
    if task is None:
        raise ValueError(f"Unknown task {task_name!r}")
    exec_cfg = doc.get("execution", {})
    conn_cfg = doc.get("connection", {})
    blend_cfg = doc.get("blending", {})
    rtc_cfg = doc.get("rtc", {})
    smooth_cfg = doc.get("smoothing", {})
    rec_cfg = doc.get("recording", {})
    opts_cfg = doc.get("options", {})

    blending_method = cli_args.blending_method or blend_cfg.get("method", "sigmoid")
    mixing_steps = (
        cli_args.mixing_steps if cli_args.mixing_steps is not None else int(blend_cfg.get("steps", 5))
    )
    temporal_agg = bool(
        cli_args.temporal_agg if cli_args.temporal_agg is not None else smooth_cfg.get("temporal_agg", False)
    )
    temporal_agg_k = float(smooth_cfg.get("temporal_agg_k", TEMPORAL_AGG_DECAY_K))
    action_interp = bool(
        cli_args.action_interp if cli_args.action_interp is not None else smooth_cfg.get("action_interp", False)
    )
    masking_strategy = (
        cli_args.masking_strategy
        if getattr(cli_args, "masking_strategy", None) is not None
        else smooth_cfg.get("masking_strategy")
    )
    masking_config = (
        cli_args.masking_config
        if getattr(cli_args, "masking_config", None) is not None
        else smooth_cfg.get("masking_config")
    )
    rtc_enabled = bool(cli_args.rtc if cli_args.rtc is not None else rtc_cfg.get("enabled", True))
    if temporal_agg and rtc_enabled:
        raise ValueError("Disable rtc when using temporal_agg")
    if temporal_agg and blending_method != "none":
        blending_method = "none"

    chunk_steps = int(exec_cfg.get("chunk_steps", 50))
    overlap_steps = int(exec_cfg.get("overlap_steps", 30))
    prefix_steps = int(cli_args.rtc_prefix_steps if cli_args.rtc_prefix_steps is not None else rtc_cfg.get("prefix_steps", 20))
    if prefix_steps > OPENPI_MAX_RTC_PREFIX_STEPS:
        raise ValueError(f"rtc prefix_steps {prefix_steps} exceeds OpenPI limit {OPENPI_MAX_RTC_PREFIX_STEPS}")

    return argparse.Namespace(
        task_name=task.name,
        task_config=None,
        _yaml_task=task,
        remote_host=cli_args.remote_host or conn_cfg.get("host", "localhost"),
        remote_port=int(cli_args.remote_port or conn_cfg.get("port", 8000)),
        api_key=cli_args.api_key,
        fps=task.control_freq,
        legacy_policy_fps=None,
        max_inference_delay_ms=None,
        query_frequency=chunk_steps,
        first_chunk_size=int(exec_cfg.get("first_chunk_steps", chunk_steps)),
        async_inference=bool(exec_cfg.get("async_inference", True)) and not temporal_agg,
        async_prefetch_steps=overlap_steps,
        rtc=rtc_enabled,
        rtc_prefix_steps=prefix_steps,
        prompt=cli_args.prompt or opts_cfg.get("prompt", ""),
        num_rollouts=int(cli_args.num_rollouts or opts_cfg.get("num_rollouts", 1)),
        max_timesteps=int(cli_args.max_timesteps or opts_cfg.get("max_timesteps", task.episode_steps)),
        timing=bool(opts_cfg.get("timing", False)),
        timing_interval=int(opts_cfg.get("timing_interval", 10)),
        dry_run_episodes=int(cli_args.dry_run_episodes),
        async_dry_run_steps=int(cli_args.dry_run_steps),
        dry_run_dataset_dir=cli_args.dry_run_dataset_dir,
        live_startup_warmup_steps=int(cli_args.warmup_steps),
        record_rollout=bool(rec_cfg.get("rollouts", False)),
        record_dagger_corrections=bool(rec_cfg.get("dagger", False)),
        eval=bool(rec_cfg.get("eval", False)),
        recording_data_dir=str(INFERENCE_RAW_DIR),
        rollout_dir=str(ROLLOUT_DIR),
        dagger_dir=str(DAGGER_DIR),
        blending_method=blending_method,
        mixing_steps=mixing_steps,
        pi_temporal_agg=temporal_agg,
        temporal_agg_k=temporal_agg_k,
        action_interp=action_interp,
        masking_strategy=masking_strategy,
        masking_config=masking_config,
        start_position_path=cli_args.start_position_path,
    )


def run_dry_run(args: argparse.Namespace) -> None:
    """Exercise smoothed Pi0.5 scheduling against recorded episodes without hardware."""
    client = WebsocketOpenPIClient(args.remote_host, args.remote_port, args.api_key)
    session = None
    try:
        runtime = PolicyRuntime(args, client)
        paths = discover_episodes(args.dry_run_dataset_dir or runtime.task["dataset_dir"], args.fps)
        session = SmoothedPolicySession(
            runtime,
            blending_method=args.blending_method,
            mixing_steps=args.mixing_steps,
            temporal_agg=args.pi_temporal_agg,
            temporal_agg_k=args.temporal_agg_k,
            action_interp=args.action_interp,
            masking_strategy=args.masking_strategy,
            masking_config=args.masking_config,
        )
        for path in paths[: args.dry_run_episodes]:
            session.reset()
            with open_episode(path) as data:
                rows = read_robot_array(data, "observations/qpos", runtime.action_dim)
                for row in range(min(len(rows), args.async_dry_run_steps)):
                    def observe():
                        return SimpleNamespace(qpos=rows[row], images=data.frame_bundle(row, runtime.cameras))

                    action = session.tick(
                        observe, str(row_metadata(data, row).get("openpi_prompt", args.prompt))
                    )
                    if action is not None:
                        if not np.isfinite(action).all():
                            raise ValueError("Dry-run action is non-finite")
                        session.committed(action)
                    if args.async_inference:
                        time.sleep(1.0 / args.fps)
                if session.step == 0:
                    raise RuntimeError("Dry run produced no executable actions")
                print(f"[Pi0.5 Dry Run] {path}: {session.step} actions, {session.queries} queries")
    finally:
        if session:
            session.close()
        client._ws.close()


def run_live(args: argparse.Namespace) -> None:
    """Run smoothed Pi0.5 rollouts on configured hardware (PiperX or ARX via HardwareRuntime)."""
    from camera_utils.capture import capture_image_from_shared_memory
    from contracts import capture_gc
    from getkey import getkey
    from hardware.arm_factory import build_arms
    from hardware.runtime import HardwareRuntime, RuntimeDamping
    from hardware.safe_exit import damping_exit
    from inference import inference_direct as direct
    from inference.policy_warmup import warm_up
    from inference.shared_hardware import resolve_start_action, validate_start_action
    from recording.distributed_recording import DistributedRecorder, EpisodeRequest
    from recording.record_episode_single import wait_for_cameras

    inputs: Queue[str] = Queue()
    input_stop = threading.Event()
    camera_stop = threading.Event()

    def home_requested() -> bool:
        while True:
            try:
                if inputs.get_nowait() == "h":
                    return True
            except Empty:
                return False

    def read_keys() -> None:
        while not input_stop.is_set():
            try:
                inputs.put(getkey().lower())
            except (EOFError, KeyboardInterrupt):
                inputs.put("q")
                return

    def next_key() -> str | None:
        try:
            key = inputs.get_nowait()
        except Empty:
            return None
        if key == "\x03":
            raise KeyboardInterrupt
        return key

    with ExitStack() as cleanup:
        cleanup.callback(input_stop.set)
        client = WebsocketOpenPIClient(args.remote_host, args.remote_port, args.api_key)
        cleanup.callback(client._ws.close)
        policy = PolicyRuntime(args, client)
        robot = args._yaml_task.as_dict()["robot"]
        hardware = HardwareRuntime(build_arms(robot))
        hardware.require_policy(client.server_metadata)
        if hardware.action_dim != policy.action_dim:
            raise ValueError("Policy dimensions differ from configured hardware")
        start, start_options = resolve_start_action(args, policy.task["task_name"], robot["type"], hardware)
        validate_start_action(hardware, start)
        session = SmoothedPolicySession(
            policy,
            blending_method=args.blending_method,
            mixing_steps=args.mixing_steps,
            temporal_agg=args.pi_temporal_agg,
            temporal_agg_k=args.temporal_agg_k,
            action_interp=args.action_interp,
            masking_strategy=args.masking_strategy,
            masking_config=args.masking_config,
        )
        cleanup.callback(session.close)
        camera_thread = threading.Thread(
            target=capture_image_from_shared_memory,
            args=(camera_stop,),
            kwargs={"camera_names": policy.cameras},
            daemon=True,
        )
        camera_thread.start()
        cleanup.callback(lambda: (camera_stop.set(), camera_thread.join(timeout=5)))
        wait_for_cameras(policy.cameras, timeout_sec=15)
        recorder = None
        if args.record_rollout or args.eval:
            recorder = DistributedRecorder(camera_names=policy.cameras, data_dir=args.recording_data_dir)
            cleanup.callback(recorder.close)
        cleanup.callback(hardware.close)
        arms = direct.RuntimeArms(hardware)
        dt = hardware.interval_s
        dual = len(hardware.arms) == 2

        def capture():
            return hardware.capture(policy.cameras, include_images=True)

        with damping_exit(
            lambda: RuntimeDamping(hardware) if hardware.live_arms() else None,
            home_requested=home_requested,
        ):
            arms.connect_and_hold()
            validate_start_action(hardware, start)
            direct.move_to_start_action(arms, {}, start, execution_dt=dt, is_double_arm=dual, **start_options)
            hold = capture().qpos.copy()
            with direct.idle_hold(arms):
                warm = capture()
                warm_up(
                    lambda index: policy.query(
                        client, warm.images, warm.qpos, args.prompt,
                        query_index=index, rtc_metadata=policy.rtc_metadata(),
                    )[1],
                    args.live_startup_warmup_steps,
                    check=(
                        lambda timing: policy_chunks.validate_overlap_budget(
                            args.fps, args.async_prefetch_steps, timing["query_total"] * 1000
                        )
                    )
                    if args.async_inference
                    else None,
                )
            threading.Thread(target=read_keys, daemon=True, name="pi05-rollout-controls").start()
            print("[Pi0.5] Ready. c: start/pause, s: success, f: failure, r: reset, q: quit")
            for rollout_index in range(args.num_rollouts):
                if rollout_index:
                    while True:
                        begin = timing_now()
                        key = next_key()
                        if key in ("q", "x"):
                            return
                        if key in ("c", "r"):
                            break
                        arms.submit(hold)
                        wait_until_loop_deadline(begin, dt)
                    direct.return_to_start_with_open_grippers(
                        arms, {}, start, execution_dt=dt, is_double_arm=dual, **start_options
                    )
                    hold = capture().qpos.copy()
                with direct.idle_hold(arms):
                    capture_gc.collect_idle()
                    session.reset()
                    if recorder:
                        from camera_utils.camera_settings_contract import snapshot_camera_settings

                        metadata = {
                            **hardware.recording_metadata(args.fps),
                            "camera_settings": snapshot_camera_settings(policy.cameras),
                            "task_name": policy.task["task_name"],
                            "openpi_prompt": args.prompt,
                        }
                        recorder.start_episode(
                            EpisodeRequest(
                                session_id=uuid.uuid4().hex,
                                data_dir=args.recording_data_dir,
                                camera_names=policy.cameras,
                                record_rollout=True,
                                record_dagger=False,
                                rollout_dir=args.rollout_dir,
                                rollout_task_name=policy.task["task_name"],
                                dagger_dir="",
                                dagger_task_name="",
                                dagger_camera_names=[],
                                is_double_arm=dual,
                                fps=args.fps,
                                eval_mode=bool(args.eval),
                                metadata=metadata,
                            ),
                            timeout=15,
                        )
                        recorder.pause(2)
                paused, executed, success, quit_requested = True, 0, False, False
                while executed < args.max_timesteps:
                    begin = timing_now()
                    key = next_key()
                    if key in ("q", "x"):
                        quit_requested = True
                        break
                    if key in ("s", "f"):
                        success = key == "s"
                        break
                    if key == "c":
                        paused = not paused
                        if recorder:
                            recorder.pause(2) if paused else recorder.resume()
                    if paused:
                        arms.submit(hold)
                    else:
                        latest_obs = None

                        def observe():
                            nonlocal latest_obs
                            latest_obs = capture()
                            return latest_obs

                        action = session.tick(observe, args.prompt)
                        if action is None:
                            arms.submit(hold)
                        else:
                            if latest_obs is None:
                                latest_obs = capture()
                            for interp_step in session.build_gap_steps(action, latest_obs.qpos):
                                step_begin = timing_now()
                                arms.submit(interp_step[0])
                                wait_until_loop_deadline(step_begin, dt)
                            target = action[0]
                            arms.submit(target)
                            if recorder:
                                recorder.append_sample(
                                    {**latest_obs.recording_fields(), "commanded_action": target}
                                )
                            session.committed(target)
                            executed += 1
                    wait_until_loop_deadline(begin, dt)
                if recorder and recorder.active_session_id:
                    recorder.stop_episode(
                        retained=True,
                        outputs={"rollout": {"requested": True, "decision": "success" if success else "failure"}},
                    )
                if quit_requested:
                    break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "pi_rollout" / "configs" / "piperx_pi05_remote.yaml"),
    )
    parser.add_argument("--remote-host", default=None)
    parser.add_argument("--remote-port", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--blending-method", choices=BLENDING_METHODS, default=None)
    parser.add_argument("--mixing-steps", type=int, default=None)
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rtc-prefix-steps", type=int, default=None)
    parser.add_argument("--temporal-agg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--action-interp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--masking-strategy", default=None)
    parser.add_argument("--masking-config", default=None)
    parser.add_argument("--num-rollouts", type=int, default=None)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--start-position-path", default="default")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-episodes", type=int, default=1)
    parser.add_argument("--dry-run-steps", type=int, default=50)
    parser.add_argument("--dry-run-dataset-dir", default=None)
    cli_args = parser.parse_args()

    args = load_pi05_args(cli_args.config, cli_args)
    if cli_args.dry_run:
        run_dry_run(args)
    else:
        run_live(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
