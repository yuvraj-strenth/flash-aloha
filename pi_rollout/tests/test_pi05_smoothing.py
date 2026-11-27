"""Unit tests for pi_rollout Pi0.5 smoothing primitives and SmoothedPolicySession."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from contracts.standard_image_processing import openpi_image_contract
from inference import inference_direct as direct
from inference.inference_remote import PolicyRuntime
from pi_rollout.pi05_rollout import SmoothedPolicySession, load_pi05_args
from pi_rollout.smoothing import (
    ActionGapInterpolator,
    TemporalAggregator,
    blend_actions,
    blend_rtc_boundary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("method", ["linear", "exponential", "sigmoid"])
def test_blend_actions_matches_direct_inference(method):
    rng = np.random.default_rng(7)
    prev_chunk = rng.normal(size=(1, 50, 7)).astype(np.float32)
    new_chunk = rng.normal(size=(1, 50, 7)).astype(np.float32)

    expected = direct.blend_actions(
        prev_chunk, new_chunk, mixing_steps=6, query_frequency=20,
        prev_chunk_start_index=2, new_chunk_start_index=1, interp_type=method,
    )
    actual = blend_actions(
        prev_chunk, new_chunk, mixing_steps=6, query_frequency=20,
        prev_chunk_start_index=2, new_chunk_start_index=1, interp_type=method,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["linear", "exponential", "sigmoid"])
def test_blend_rtc_boundary_matches_direct_inference_and_preserves_prefix(method):
    rng = np.random.default_rng(11)
    prev_plan = rng.normal(size=(1, 50, 7)).astype(np.float32)
    new_chunk = rng.normal(size=(1, 50, 7)).astype(np.float32)
    boundary = 10

    expected = direct.blend_rtc_boundary(
        prev_plan, new_chunk, prev_next_index=15, boundary=boundary, steps=5, interp_type=method,
    )
    actual = blend_rtc_boundary(
        prev_plan, new_chunk, prev_next_index=15, boundary=boundary, steps=5, interp_type=method,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[:, :boundary], new_chunk[:, :boundary])


def test_blend_rtc_boundary_rejects_short_previous_plan():
    prev_plan = np.zeros((1, 20, 7), dtype=np.float32)
    new_chunk = np.ones((1, 20, 7), dtype=np.float32)
    with pytest.raises(ValueError, match="previous plan has only"):
        blend_rtc_boundary(prev_plan, new_chunk, prev_next_index=10, boundary=8, steps=5)


def test_temporal_aggregator_exponential_weights_and_wraparound():
    agg = TemporalAggregator(num_queries=4, max_timesteps=10, action_dim=7, decay_k=0.005)
    c0 = np.ones((1, 4, 7), dtype=np.float32) * 1.0
    c1 = np.ones((1, 4, 7), dtype=np.float32) * 3.0
    agg.add_chunk(0, c0)
    agg.add_chunk(1, c1)

    selected = agg.select(1)
    assert selected is not None and selected.shape == (1, 7)
    w0 = np.exp(-0.005 * 1.0)
    w1 = np.exp(-0.005 * 0.0)
    expected = (w0 * 1.0 + w1 * 3.0) / (w0 + w1)
    np.testing.assert_allclose(selected, expected, rtol=1e-6, atol=1e-6)


def test_action_gap_interpolator_triggers_only_above_threshold():
    interp = ActionGapInterpolator(is_double_arm=False, threshold_rad=0.03, step_size_rad=0.05)
    qpos = np.zeros(7, dtype=np.float32)
    assert interp.build_steps(np.full(7, 0.005, dtype=np.float32), qpos, qpos, qpos) == []
    steps = interp.build_steps(np.full(7, 0.1, dtype=np.float32), qpos, qpos, qpos)
    assert len(steps) >= 2
    assert steps[0].shape == (1, 7)


class _FakeOpenPIClient:
    def __init__(self, chunk_fn):
        self._chunk_fn = chunk_fn
        self.calls = []
        self._ws = SimpleNamespace(close=lambda: None)
        self.server_metadata = {
            "model_family": "openpi",
            "request_schema": "openpi",
            "task_name": "piperx_single",
            "fps": 100.0,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 50,
            "action_command_style": "absolute",
            "camera_names": ["video_overhead", "video_left"],
            "robot_contract": {
                "robot_contract_version": 1,
                "arms": {"left": {"model": "PiperX"}},
            },
            "image_preprocessing": openpi_image_contract(
                {
                    "video_overhead": {"request_key": "video_overhead", "source_shape": [480, 640, 3]},
                    "video_left": {"request_key": "video_left", "source_shape": [480, 640, 3]},
                }
            ),
            "rtc_training_time": True,
            "rtc_max_prefix_steps": 20,
        }

    def infer(self, request):
        self.calls.append(request)
        actions = self._chunk_fn(len(self.calls) - 1, request)
        return {"actions": actions}

    def infer_with_timing(self, request):
        return self.infer(request), {"total_ms": 1.0, "encode_ms": 0.1, "network_ms": 0.9}


def _make_cli_args(**overrides):
    defaults = dict(
        remote_host=None, remote_port=None, api_key=None, prompt="pick cube",
        blending_method=None, mixing_steps=None, rtc=None, rtc_prefix_steps=None,
        temporal_agg=None, action_interp=None, masking_strategy=None, masking_config=None,
        num_rollouts=1, max_timesteps=200,
        warmup_steps=1, start_position_path="default", dry_run_episodes=1,
        dry_run_steps=50, dry_run_dataset_dir=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _fake_observe():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return SimpleNamespace(
        qpos=np.zeros(7, dtype=np.float32),
        images={"video_overhead": img, "video_left": img},
    )


def test_smoothed_session_sync_and_rtc_blending():
    cfg_path = REPO_ROOT / "pi_rollout" / "configs" / "piperx_pi05_remote.yaml"
    args = load_pi05_args(cfg_path, _make_cli_args(blending_method="sigmoid", mixing_steps=4, rtc=True, rtc_prefix_steps=5))
    args.async_inference = False
    args.query_frequency = 10
    args.first_chunk_size = 10

    def chunk_fn(call_idx, _req):
        return np.full((50, 7), float(call_idx), dtype=np.float32)

    client = _FakeOpenPIClient(chunk_fn)
    runtime = PolicyRuntime(args, client)
    session = SmoothedPolicySession(
        runtime, blending_method="sigmoid", mixing_steps=4, temporal_agg=False
    )
    try:
        outputs = []
        for _ in range(18):
            act = session.tick(_fake_observe, "pick cube")
            assert act is not None
            outputs.append(act[0, 0])
            session.committed(act[0])
        # First 10 steps from chunk 0 (0.0); steps 10..14 pinned prefix from chunk 1 (1.0);
        # steps 15..17 crossfaded between chunk 0 (0.0) and chunk 1 (1.0) -> strictly in (0, 1).
        assert all(abs(v - 0.0) < 1e-6 for v in outputs[:10])
        assert all(0.0 < v < 1.0 for v in outputs[15:18])
    finally:
        session.close()


def test_smoothed_session_async_blending_and_temporal_agg():
    cfg_path = REPO_ROOT / "pi_rollout" / "configs" / "piperx_pi05_remote.yaml"
    args = load_pi05_args(cfg_path, _make_cli_args(temporal_agg=True, rtc=False))
    assert args.async_inference is False

    client = _FakeOpenPIClient(lambda idx, _req: np.full((50, 7), float(idx + 1), dtype=np.float32))
    runtime = PolicyRuntime(args, client)
    session = SmoothedPolicySession(runtime, blending_method="none", temporal_agg=True)
    try:
        for _ in range(5):
            act = session.tick(_fake_observe, "pick cube")
            assert act is not None and np.isfinite(act).all()
            session.committed(act[0])
        assert session.queries == 5
    finally:
        session.close()


def test_smoothed_session_async_worker_blends_on_handoff():
    cfg_path = REPO_ROOT / "pi_rollout" / "configs" / "piperx_pi05_remote.yaml"
    args = load_pi05_args(cfg_path, _make_cli_args(blending_method="linear", mixing_steps=4, rtc=False))
    args.query_frequency = 10
    args.first_chunk_size = 10
    args.async_prefetch_steps = 5

    client = _FakeOpenPIClient(lambda idx, _req: np.full((50, 7), float(idx), dtype=np.float32))
    runtime = PolicyRuntime(args, client)
    session = SmoothedPolicySession(runtime, blending_method="linear", mixing_steps=4)
    try:
        # Step 0 submits async query 0; wait briefly for worker thread to return chunk 0.
        assert session.tick(_fake_observe, "pick cube") is None
        deadline = time.monotonic() + 2.0
        act = None
        while time.monotonic() < deadline and act is None:
            act = session.tick(_fake_observe, "pick cube")
            if act is None:
                time.sleep(0.005)
        assert act is not None
        session.committed(act[0])
        for _ in range(25):
            act = session.tick(_fake_observe, "pick cube")
            assert act is not None and np.isfinite(act).all()
            session.committed(act[0])
            time.sleep(0.002)
        assert session.queries >= 2
    finally:
        session.close()


def test_smoothed_session_applies_masker_without_mutating_observation():
    cfg_path = REPO_ROOT / "pi_rollout" / "configs" / "piperx_pi05_remote.yaml"
    args = load_pi05_args(
        cfg_path,
        _make_cli_args(
            blending_method="none",
            rtc=False,
            masking_strategy="arm_cameras",
            masking_config=str(REPO_ROOT / "calibration" / "masking" / "masking.yaml"),
        ),
    )
    args.async_inference = False
    args.query_frequency = 10
    args.first_chunk_size = 10

    client = _FakeOpenPIClient(lambda idx, _req: np.ones((50, 7), dtype=np.float32))
    runtime = PolicyRuntime(args, client)
    session = SmoothedPolicySession(
        runtime,
        blending_method="none",
        masking_strategy=args.masking_strategy,
        masking_config=args.masking_config,
    )
    try:
        overhead = np.full((480, 640, 3), 200, dtype=np.uint8)
        wrist = np.full((480, 640, 3), 150, dtype=np.uint8)
        obs = SimpleNamespace(
            qpos=np.zeros(7, dtype=np.float32),
            images={"video_overhead": overhead, "video_left": wrist},
        )
        act = session.tick(lambda: obs, "pick cube")
        assert act is not None
        # arm_cameras blanks video_left in the request sent to OpenPI, while leaving obs untouched.
        req = client.calls[0]
        assert np.all(req["video_left"] == 0)
        assert np.any(req["video_overhead"] > 0)
        assert np.all(obs.images["video_left"] == 150)
    finally:
        session.close()
