#!/usr/bin/env python3
"""Local B601 checkpoint execution through the shared arm runtime.

Chunk selection and blending are the direct runner's functions. This module only
turns the resulting absolute target into a MIT command on the B601 adapter.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ACT_ROOT = Path(__file__).resolve().parents[1]
if str(_ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACT_ROOT))

from hardware.runtime import RuntimeDamping
from hardware.safe_exit import damping_exit
from recording.b601_session import load_start_pose, load_task, open_runtime


def executed_action(chunk, t, query_frequency, chunk_start_index, gripper_lookahead=0,
                    is_double_arm=False, first_chunk_size=None):
    """Absolute action at step t. Indexing is the direct runner's get_action_index."""
    from inference.inference_direct import select_action_from_chunk
    first = query_frequency if first_chunk_size is None else first_chunk_size
    action = select_action_from_chunk(
        t, first, query_frequency, chunk_start_index, gripper_lookahead, chunk, is_double_arm)
    return np.asarray(action, dtype=float).reshape(-1)


def merge_chunk(previous, new_chunk, mixing_steps, query_frequency, previous_start, new_start, interp_type):
    from inference.inference_direct import blend_actions, blend_rtc_boundary
    if mixing_steps <= 0 or previous is None:
        return new_chunk
    if interp_type == 'rtc':
        return blend_rtc_boundary(previous, new_chunk, previous_start, new_start, mixing_steps)
    return blend_actions(previous, new_chunk, mixing_steps, query_frequency, previous_start, new_start, interp_type)


def run_rollout(hardware, camera_names, query_chunk, steps, interval_s, query_frequency, *,
                first_chunk_size=None, mixing_steps=0, interp_type='linear', gripper_lookahead=0,
                rtc_prefix=0):
    from inference.inference_direct import is_query_point
    first = query_frequency if first_chunk_size is None else first_chunk_size
    chunk = None
    chunk_start = 0
    sent = []
    for t in range(steps):
        observation = hardware.capture(camera_names, include_images=bool(camera_names))
        if chunk is None or is_query_point(t, first, query_frequency):
            new_chunk = query_chunk(observation)
            if rtc_prefix and chunk is not None:
                window = first if t == first else query_frequency
                previous_next = chunk_start + window - rtc_prefix
                chunk = merge_chunk(chunk, new_chunk, mixing_steps, query_frequency,
                                    previous_next, rtc_prefix, 'rtc')
                chunk_start = rtc_prefix
            else:
                chunk = merge_chunk(chunk, new_chunk, mixing_steps, query_frequency,
                                    chunk_start, 0, interp_type)
                chunk_start = 0
        action = executed_action(chunk, t, query_frequency, chunk_start, gripper_lookahead,
                                 is_double_arm=hardware.action_dim == 14, first_chunk_size=first)
        hardware.submit_target(action, interval_s=interval_s)
        sent.append(action)
    return sent


def _query_from_policy(policy, stats, camera_names):
    import torch
    from inference.inference_direct import process_image
    mean = torch.from_numpy(np.asarray(stats['qpos_mean'], dtype=np.float32))
    std = torch.from_numpy(np.asarray(stats['qpos_std'], dtype=np.float32))
    action_mean = np.asarray(stats['action_mean'], dtype=np.float32)
    action_std = np.asarray(stats['action_std'], dtype=np.float32)

    def query(observation):
        qpos = torch.from_numpy(np.asarray(observation.qpos, dtype=np.float32))
        qpos = ((qpos - mean) / std).unsqueeze(0).cuda()
        image = process_image(observation.images, camera_names=camera_names).cuda()
        normalized = policy.forward_inf(qpos, image).detach().cpu().numpy()
        return normalized * action_std + action_mean

    return query


def _execution(task, checkpoint, inference_config):
    """Chunk rate and blend from a direct profile. The checkpoint still owns the horizon."""
    from config.task_config import checkpoint_chunk_size
    horizon = checkpoint_chunk_size(checkpoint)
    chosen = dict(query_frequency=horizon, first_chunk_size=horizon, mixing_steps=0,
                  interp_type='linear', horizon=horizon, fast_inference=False)
    if inference_config is None:
        return chosen
    from config.inference_config import load_inference_config
    profile_task, profile = load_inference_config(inference_config)
    if profile_task.source != task.source:
        raise ValueError('Inference profile and --task-config must reference the same task YAML')
    if profile['_inference_document']['backend'] != 'direct':
        raise ValueError('B601 local inference requires a direct profile; remote uses --shared-hardware')
    if profile.get('rtc'):
        raise ValueError('B601 local inference does not run the async RTC query; set rtc.enabled: false')
    chosen.update(query_frequency=profile['query_frequency'], first_chunk_size=profile['first_chunk_size'],
                  mixing_steps=profile['mixing_steps'] if profile.get('blend_actions') else 0,
                  interp_type=profile.get('blending_method', 'sigmoid'),
                  fast_inference=bool(profile.get('fast_inference', False)))
    return chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Checkpoint all_configs.json')
    parser.add_argument('--task-config', required=True)
    parser.add_argument('--inference-config', default=None)
    parser.add_argument('--policy_step', type=int, default=-1)
    parser.add_argument('--max_timesteps', type=int, default=None)
    args = parser.parse_args(argv)
    task = load_task(args.task_config)
    with open(args.config) as stream:
        config = json.load(stream)
    execution = _execution(task, config, args.inference_config)
    from inference.inference_direct import load_policy_and_stats
    from config.task_config import require_chunk_length
    ckpt_name = 'policy_best.ckpt' if args.policy_step == -1 else f'policy_step_{args.policy_step}_seed_0.ckpt'
    policy, stats, _policy_config = load_policy_and_stats(
        config, ckpt_name, fast_inference=execution['fast_inference'])
    names = task.as_dict()['cameras']['names']
    hardware = open_runtime(task)
    hardware.require_policy(config, stats=stats)
    start = load_start_pose(task.name)
    ask = _query_from_policy(policy, stats, names)
    try:
        hardware.connect()
        with damping_exit(lambda: RuntimeDamping(hardware) if hardware.live_arms() else None):
            for arm, offset in zip(hardware.arms.values(), (0,)):
                arm.move_toward(start[offset:offset + 7], duration_s=3.0, interval_s=task.dt)
            def query(observation):
                return require_chunk_length(ask(observation), execution['horizon'])
            run_rollout(hardware, names, query, args.max_timesteps or task.episode_steps, task.dt,
                        execution['query_frequency'], first_chunk_size=execution['first_chunk_size'],
                        mixing_steps=execution['mixing_steps'], interp_type=execution['interp_type'])
    finally:
        hardware.close()


if __name__ == '__main__':
    main()
