"""Pure-NumPy smoothing primitives for Pi0.5 remote rollouts.

Mirrors the local inference smoothing methods in ``src/inference/inference_direct.py``:
- Chunk blending (``linear``, ``exponential``, ``sigmoid``)
- RTC boundary crossfading (``blend_rtc_boundary``)
- Temporal aggregation (exponentially weighted rolling average, ``k = 0.005``)
- Inter-chunk Bezier action-gap interpolation (``ActionGapInterpolator``)
"""
from __future__ import annotations

import numpy as np

BLENDING_METHODS = ("none", "linear", "exponential", "sigmoid")
TEMPORAL_AGG_DECAY_K = 0.005
DEFAULT_GAP_THRESHOLD_RAD = 0.03
DEFAULT_GAP_STEP_SIZE_RAD = 0.05


def blend_actions(
    prev_action_chunk: np.ndarray,
    action_chunk: np.ndarray,
    mixing_steps: int,
    query_frequency: int,
    prev_chunk_start_index: int,
    new_chunk_start_index: int,
    interp_type: str = "sigmoid",
) -> np.ndarray:
    """Blend previous and new action chunks aligned to executed indices."""
    if interp_type == "none" or mixing_steps <= 0:
        return action_chunk
    if interp_type not in ("linear", "exponential", "sigmoid"):
        raise ValueError(f"Unknown blending method {interp_type!r}")

    new_chunk = np.asarray(action_chunk, dtype=np.float32).copy()
    prev_chunk = np.asarray(prev_action_chunk, dtype=np.float32)

    prev_blend_start = int(query_frequency) + int(prev_chunk_start_index)
    new_blend_start = int(new_chunk_start_index)
    if prev_blend_start < 0 or prev_blend_start >= prev_chunk.shape[1]:
        return new_chunk

    prev_blend_end = min(prev_blend_start + int(mixing_steps), prev_chunk.shape[1])
    new_blend_end = min(new_blend_start + int(mixing_steps), new_chunk.shape[1])
    actual_mix_len = min(prev_blend_end - prev_blend_start, new_blend_end - new_blend_start)
    if actual_mix_len <= 0:
        return new_chunk

    if interp_type == "linear":
        start = prev_chunk[:, prev_blend_start]
        end = new_chunk[:, new_blend_start + actual_mix_len - 1]
        new_chunk[:, new_blend_start : new_blend_start + actual_mix_len] = np.linspace(
            start, end, num=actual_mix_len, axis=1
        )
    elif interp_type == "exponential":
        for i in range(actual_mix_len):
            weight = 1.0 - (i / actual_mix_len) ** 3
            new_chunk[:, new_blend_start + i] = (
                weight * prev_chunk[:, prev_blend_start + i]
                + (1.0 - weight) * new_chunk[:, new_blend_start + i]
            )
    elif interp_type == "sigmoid":
        for i in range(actual_mix_len):
            weight = 1.0 / (1.0 + np.exp(3.0 * (i / actual_mix_len - 0.5)))
            new_chunk[:, new_blend_start + i] = (
                weight * prev_chunk[:, prev_blend_start + i]
                + (1.0 - weight) * new_chunk[:, new_blend_start + i]
            )
    return new_chunk


def blend_rtc_boundary(
    prev_plan: np.ndarray,
    new_chunk: np.ndarray,
    prev_next_index: int,
    boundary: int,
    steps: int,
    interp_type: str = "sigmoid",
) -> np.ndarray:
    """Crossfade executed actions from ``prev_plan`` into ``new_chunk`` after RTC ``boundary``.

    Actions before ``boundary`` are the pinned RTC prefix and remain untouched.
    """
    if interp_type == "none":
        return new_chunk
    if steps <= 0:
        raise ValueError(f"Blending window {steps} must be positive")
    if (
        prev_next_index + boundary + steps > prev_plan.shape[1]
        or boundary + steps > new_chunk.shape[1]
    ):
        raise ValueError(
            f"Blending {steps} actions after the handoff needs actions up to previous-plan index "
            f"{prev_next_index + boundary + steps}; the previous plan has only {prev_plan.shape[1]}"
        )
    blended = np.asarray(new_chunk, dtype=np.float32).copy()
    prev_arr = np.asarray(prev_plan, dtype=np.float32)
    for k in range(steps):
        index, old_index = boundary + k, prev_next_index + boundary + k
        x = (k + 0.5) / steps
        if interp_type == "linear":
            weight = 1.0 - x
        elif interp_type == "exponential":
            weight = 1.0 - x ** 3
        elif interp_type == "sigmoid":
            weight = 1.0 / (1.0 + np.exp(6.0 * (x - 0.5)))
        else:
            raise ValueError(f"Unknown blending method {interp_type!r}")
        blended[:, index] = (
            weight * prev_arr[:, old_index] + (1.0 - weight) * blended[:, index]
        )
    return blended


class TemporalAggregator:
    """Circular-buffer temporal aggregation over overlapping action chunks."""

    def __init__(
        self,
        num_queries: int,
        max_timesteps: int,
        action_dim: int,
        *,
        decay_k: float = TEMPORAL_AGG_DECAY_K,
    ) -> None:
        if num_queries <= 0 or max_timesteps <= 0 or action_dim <= 0:
            raise ValueError("num_queries, max_timesteps, and action_dim must be positive")
        self.num_queries = int(num_queries)
        self.max_timesteps = int(max_timesteps)
        self.action_dim = int(action_dim)
        self.decay_k = float(decay_k)
        self.reset()

    def reset(self) -> None:
        self.all_time_actions = np.zeros(
            (self.num_queries, self.max_timesteps, self.action_dim), dtype=np.float32
        )
        self.action_timestamps = np.full((self.num_queries,), -1, dtype=np.int64)

    def add_chunk(self, t: int, action_chunk: np.ndarray) -> None:
        """Record a predicted chunk starting at rollout step ``t``."""
        chunk = np.asarray(action_chunk, dtype=np.float32)
        if chunk.ndim == 3:
            chunk = chunk[0]
        slot_idx = int(t) % self.num_queries
        self.all_time_actions[slot_idx].fill(0.0)
        end_idx = min(self.num_queries, chunk.shape[0], self.max_timesteps - int(t))
        if end_idx > 0:
            self.all_time_actions[slot_idx, t : t + end_idx] = chunk[:end_idx, : self.action_dim]
        self.action_timestamps[slot_idx] = int(t)

    def select(self, t: int) -> np.ndarray | None:
        """Return exponentially weighted action for step ``t``, or ``None`` if unavailable."""
        if t < 0 or t >= self.max_timesteps:
            return None
        actions_for_step = []
        timestamps_for_step = []
        for slot_idx in range(self.num_queries):
            chunk_start = int(self.action_timestamps[slot_idx])
            if chunk_start >= 0 and chunk_start <= t < chunk_start + self.num_queries:
                val = self.all_time_actions[slot_idx, t]
                if np.any(val != 0.0):
                    actions_for_step.append(val)
                    timestamps_for_step.append(chunk_start)
        if not actions_for_step:
            return None
        stacked = np.stack(actions_for_step, axis=0)
        time_diffs = float(t) - np.asarray(timestamps_for_step, dtype=np.float64)
        weights = np.exp(-self.decay_k * time_diffs)
        weights = (weights / weights.sum())[:, None]
        return np.sum(stacked * weights, axis=0, dtype=np.float64).astype(np.float32)[None, :]


class ActionGapInterpolator:
    """Quadratic Bezier sub-step generator for large inter-chunk target jumps."""

    def __init__(
        self,
        *,
        is_double_arm: bool = False,
        threshold_rad: float = DEFAULT_GAP_THRESHOLD_RAD,
        step_size_rad: float = DEFAULT_GAP_STEP_SIZE_RAD,
    ) -> None:
        self.joints_to_compare = 12 if is_double_arm else 6
        self.threshold_rad = float(threshold_rad)
        self.step_size_rad = float(step_size_rad)

    def build_steps(
        self,
        new_action: np.ndarray,
        cur_qpos: np.ndarray,
        prev_action: np.ndarray,
        prev_prev_action: np.ndarray,
    ) -> list[np.ndarray]:
        """Return intermediate actions if the arm-joint jump exceeds ``threshold_rad``."""
        target = np.asarray(new_action, dtype=np.float32).reshape(-1)
        qpos = np.asarray(cur_qpos, dtype=np.float32).reshape(-1)
        prev1 = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        prev2 = np.asarray(prev_prev_action, dtype=np.float32).reshape(-1)
        n = min(self.joints_to_compare, target.shape[0], qpos.shape[0])
        dist = float(np.linalg.norm(target[:n] - qpos[:n]))
        if dist <= self.threshold_rad:
            return []
        n_steps = max(1, int(2 * np.ceil(dist / self.step_size_rad)))
        virtual_action = 2.0 * prev1 - prev2
        steps = []
        for i in range(n_steps):
            alpha = i / n_steps
            wp1 = alpha * target + (1.0 - alpha) * virtual_action
            wp2 = (1.0 - alpha) * prev1 + alpha * virtual_action
            steps.append((wp2 + (wp1 - wp2) * alpha)[None, :].astype(np.float32))
        return steps
