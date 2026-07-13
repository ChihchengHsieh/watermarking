from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class SchedulerTaskStats:
    usage_count: int = 0
    last_step: int = -1
    loss_ema: Optional[float] = None
    prev_loss_ema: Optional[float] = None
    reward_ema: float = 0.0
    uncertainty_ema: float = 0.0
    recent_losses: List[float] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)


@dataclass
class BaselineSchedulerConfig:
    mode: str
    seed: int = 0
    ema_beta: float = 0.85
    reward_beta: float = 0.9
    temperature: float = 1.0
    min_weight: float = 1e-4
    warmup_rounds: int = 1
    ucb_c: float = 1.0
    thompson_std: float = 0.25
    recent_window: int = 20
    bass_lr: float = 0.05
    bass_exploration: float = 0.25
    ats_lr: float = 0.05
    ats_exploration: float = 0.10
    derts_noise_penalty: float = 0.75
    derts_coverage_bonus: float = 0.50
    asr_diversity_weight: float = 0.35
    asr_entropy_weight: float = 0.25
    asr_difficulty_weight: float = 0.40
    gcp_tau: float = 0.95
    gcp_alpha: float = 0.50
    gcp_min_rate: float = 0.05
    gcp_max_ratio: float = 10.0


class BaselineTaskSchedulerController:
    """
    Lightweight scheduler baselines for SpiderMark attack-task meta-training.

    Supported modes:
      - hard_task: emphasize tasks with high recent query loss
      - progress: emphasize tasks with large recent learning progress/uncertainty
      - bandit_ucb: choose the task with the largest UCB score
      - bandit_thompson: sample a task from a Gaussian reward posterior
      - ats: adaptive task scheduler inspired by ATS; not an official reproduction
      - bass: contextual-bandit scheduler inspired by BASS; not an official reproduction
      - asr: ASr-style adaptive sampler using diversity, entropy, and difficulty
      - derts_proxy: lightweight DERTS-inspired task-selection proxy; not an official reproduction
      - gcp_proxy: GCP-style exponential task-weight update; not an official reproduction

    The controller uses the same feedback path as the residual scheduler:
    update(task_id, loss=outer_loss, ...).
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: Optional[Sequence[str]] = None,
        config: Optional[BaselineSchedulerConfig] = None,
    ):
        if num_tasks <= 0:
            raise ValueError("BaselineTaskSchedulerController requires at least one task.")
        self.num_tasks = int(num_tasks)
        self.task_names = list(task_names or [str(i) for i in range(num_tasks)])
        self.config = config or BaselineSchedulerConfig(mode="hard_task")
        self.stats: List[SchedulerTaskStats] = [
            SchedulerTaskStats() for _ in range(self.num_tasks)
        ]
        self.step = 0
        self.rng = np.random.default_rng(self.config.seed)
        self._last_probs: Optional[np.ndarray] = None
        self._last_scores: Optional[np.ndarray] = None
        self._last_contexts: Optional[np.ndarray] = None
        self._last_task_id: Optional[int] = None
        self.bass_theta = np.zeros(7, dtype=np.float64)
        self.ats_theta = np.zeros(6, dtype=np.float64)
        self.gcp_weights = np.ones(self.num_tasks, dtype=np.float64)
        self.global_context: Dict[str, float] = {}

    def sample(self) -> Tuple[int, Dict[str, np.ndarray]]:
        warmup_task = self._warmup_task()
        if warmup_task is not None:
            probs = np.full(self.num_tasks, self.config.min_weight, dtype=np.float64)
            probs[warmup_task] = 1.0
            probs = self._normalize(probs)
            scores = probs.copy()
            task_id = warmup_task
        elif self.config.mode == "hard_task":
            scores = self._hard_task_scores()
            probs = self._softmax(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        elif self.config.mode == "progress":
            scores = self._progress_scores()
            probs = self._softmax(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        elif self.config.mode == "bandit_ucb":
            scores = self._ucb_scores()
            probs = self._argmax_probs(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        elif self.config.mode == "bandit_thompson":
            scores = self._thompson_scores()
            probs = self._argmax_probs(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        elif self.config.mode == "ats":
            contexts = self._ats_contexts()
            scores = self._ats_scores(contexts)
            probs = self._softmax(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
            self._last_contexts = contexts
        elif self.config.mode == "bass":
            contexts = self._bass_contexts()
            scores = self._bass_scores(contexts)
            probs = self._softmax(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
            self._last_contexts = contexts
        elif self.config.mode == "asr":
            scores = self._asr_scores()
            probs = self._softmax(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        elif self.config.mode == "derts_proxy":
            scores = self._derts_proxy_scores()
            probs = self._softmax(scores)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        elif self.config.mode == "gcp_proxy":
            scores = self._gcp_scores()
            probs = self._normalize(self.gcp_weights)
            task_id = int(self.rng.choice(np.arange(self.num_tasks), p=probs))
        else:
            raise ValueError(f"Unsupported scheduler mode: {self.config.mode}")

        stat = self.stats[task_id]
        stat.usage_count += 1
        stat.last_step = self.step
        self.step += 1
        self._last_probs = probs
        self._last_scores = scores
        self._last_task_id = task_id
        return task_id, {"probs": probs, "scores": scores}

    def update(
        self,
        task_id: int,
        *,
        loss: Optional[float] = None,
        val_gain: Optional[float] = None,
        fail_rate: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> float:
        task_id = int(task_id)
        stat = self.stats[task_id]
        beta = float(self.config.ema_beta)

        loss_reward = 0.0
        if loss is not None:
            loss = float(loss)
            old_loss = stat.loss_ema
            stat.prev_loss_ema = old_loss
            stat.loss_ema = loss if old_loss is None else beta * old_loss + (1 - beta) * loss
            if old_loss is not None:
                loss_reward = old_loss - stat.loss_ema
                stat.uncertainty_ema = beta * stat.uncertainty_ema + (1 - beta) * abs(
                    loss - old_loss
                )
            stat.recent_losses.append(loss)
            stat.recent_losses = stat.recent_losses[-self.config.recent_window :]

        if reward is None:
            reward = loss_reward
            if val_gain is not None:
                reward += float(val_gain)
            if fail_rate is not None:
                reward -= 0.1 * float(fail_rate)

        reward = float(np.clip(reward, -1.0, 1.0))
        stat.reward_ema = (
            self.config.reward_beta * stat.reward_ema
            + (1 - self.config.reward_beta) * reward
        )
        stat.recent_rewards.append(reward)
        stat.recent_rewards = stat.recent_rewards[-self.config.recent_window :]

        if self.config.mode == "ats":
            contexts = self._last_contexts if self._last_contexts is not None else self._ats_contexts()
            self.ats_theta += self.config.ats_lr * reward * contexts[task_id]
            self.ats_theta *= 0.999
            self.ats_theta = np.clip(self.ats_theta, -3.0, 3.0)
        elif self.config.mode == "bass":
            contexts = self._last_contexts if self._last_contexts is not None else self._bass_contexts()
            self.bass_theta += self.config.bass_lr * reward * contexts[task_id]
            self.bass_theta *= 0.999
            self.bass_theta = np.clip(self.bass_theta, -3.0, 3.0)
        elif self.config.mode == "gcp_proxy":
            self._update_gcp_weight(task_id)
        return reward

    def update_global_context(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if isinstance(value, (int, float, np.floating)):
                self.global_context[key] = float(value)

    def snapshot(self) -> Dict[str, object]:
        return {
            "mode": self.config.mode,
            "step": self.step,
            "last_probs": None if self._last_probs is None else self._last_probs.copy(),
            "last_scores": None if self._last_scores is None else self._last_scores.copy(),
            "ats_theta": self.ats_theta.copy(),
            "bass_theta": self.bass_theta.copy(),
            "gcp_weights": self.gcp_weights.copy(),
            "global_context": dict(self.global_context),
            "tasks": [
                {
                    "task_id": i,
                    "task_name": self.task_names[i],
                    "usage_count": s.usage_count,
                    "last_step": s.last_step,
                    "loss_ema": s.loss_ema,
                    "prev_loss_ema": s.prev_loss_ema,
                    "reward_ema": s.reward_ema,
                    "uncertainty_ema": s.uncertainty_ema,
                    "recent_loss_mean": self._recent_mean(s.recent_losses),
                    "recent_reward_mean": self._recent_mean(s.recent_rewards),
                }
                for i, s in enumerate(self.stats)
            ],
        }

    def _warmup_task(self) -> Optional[int]:
        warmup_budget = max(0, int(self.config.warmup_rounds)) * self.num_tasks
        if self.step >= warmup_budget:
            return None
        return self.step % self.num_tasks

    def _hard_task_scores(self) -> np.ndarray:
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        fallback = float(np.mean(losses)) if losses else 0.0
        return np.asarray(
            [fallback if s.loss_ema is None else float(s.loss_ema) for s in self.stats],
            dtype=np.float64,
        )

    def _progress_scores(self) -> np.ndarray:
        scores = []
        for stat in self.stats:
            if stat.loss_ema is None or stat.prev_loss_ema is None:
                scores.append(stat.uncertainty_ema)
            else:
                scores.append(abs(stat.prev_loss_ema - stat.loss_ema) + stat.uncertainty_ema)
        return np.asarray(scores, dtype=np.float64)

    def _ucb_scores(self) -> np.ndarray:
        total = max(1, sum(s.usage_count for s in self.stats))
        scores = []
        for stat in self.stats:
            bonus = self.config.ucb_c * np.sqrt(np.log(total + 1.0) / (stat.usage_count + 1.0))
            scores.append(float(stat.reward_ema) + float(bonus))
        return np.asarray(scores, dtype=np.float64)

    def _thompson_scores(self) -> np.ndarray:
        scores = []
        for stat in self.stats:
            scale = self.config.thompson_std / np.sqrt(stat.usage_count + 1.0)
            scores.append(float(self.rng.normal(stat.reward_ema, scale)))
        return np.asarray(scores, dtype=np.float64)

    def _ats_contexts(self) -> np.ndarray:
        max_usage = max(1, max(s.usage_count for s in self.stats))
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        loss_center = float(np.mean(losses)) if losses else 0.0
        loss_scale = float(np.std(losses)) if len(losses) > 1 else 1.0
        loss_scale = max(loss_scale, 1e-6)
        global_loss = float(self.global_context.get("meta_loss_recent_avg", loss_center))

        rows = []
        for stat in self.stats:
            loss = loss_center if stat.loss_ema is None else float(stat.loss_ema)
            loss_gap = (loss - global_loss) / loss_scale
            uncertainty = float(np.tanh(stat.uncertainty_ema))
            progress = 0.0
            if stat.loss_ema is not None and stat.prev_loss_ema is not None:
                progress = stat.prev_loss_ema - stat.loss_ema
            under_sampled = 1.0 - (float(stat.usage_count) / float(max_usage))
            rows.append(
                [
                    1.0,
                    float(np.tanh(loss_gap)),
                    uncertainty,
                    float(np.tanh(progress)),
                    under_sampled,
                    float(stat.reward_ema),
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    def _ats_scores(self, contexts: np.ndarray) -> np.ndarray:
        score = contexts @ self.ats_theta
        total = max(1, sum(s.usage_count for s in self.stats))
        explore = np.asarray(
            [
                self.config.ats_exploration
                * np.sqrt(np.log(total + 1.0) / (stat.usage_count + 1.0))
                for stat in self.stats
            ],
            dtype=np.float64,
        )
        return score + explore

    def _bass_contexts(self) -> np.ndarray:
        max_usage = max(1, max(s.usage_count for s in self.stats))
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        loss_center = float(np.mean(losses)) if losses else 0.0
        loss_scale = float(np.std(losses)) if len(losses) > 1 else 1.0
        loss_scale = max(loss_scale, 1e-6)

        rows = []
        for stat in self.stats:
            recent_loss = 0.0 if stat.loss_ema is None else (stat.loss_ema - loss_center) / loss_scale
            progress = 0.0
            if stat.loss_ema is not None and stat.prev_loss_ema is not None:
                progress = stat.prev_loss_ema - stat.loss_ema
            recency = 1.0 if stat.last_step < 0 else min(1.0, float(self.step - stat.last_step) / max(1, self.num_tasks))
            rows.append(
                [
                    1.0,
                    float(np.tanh(recent_loss)),
                    float(np.tanh(progress)),
                    float(stat.reward_ema),
                    float(np.tanh(stat.uncertainty_ema)),
                    float(stat.usage_count) / float(max_usage),
                    recency,
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    def _bass_scores(self, contexts: np.ndarray) -> np.ndarray:
        exploit = contexts @ self.bass_theta
        total = max(1, sum(s.usage_count for s in self.stats))
        explore = np.asarray(
            [
                self.config.bass_exploration
                * np.sqrt(np.log(total + 1.0) / (stat.usage_count + 1.0))
                for stat in self.stats
            ],
            dtype=np.float64,
        )
        return exploit + explore

    def _asr_scores(self) -> np.ndarray:
        """ASr-style score using diversity, entropy, and difficulty.

        The ASr paper/repo defines task sampling around diversity, entropy, and
        difficulty. In this attack-task scheduler we do not have class-level
        episode embeddings, so the local proxy maps those factors to online
        task statistics: under-sampling/recency for diversity, loss volatility
        for entropy, and recent normalized loss for difficulty.
        """
        max_usage = max(1, max(s.usage_count for s in self.stats))
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        loss_center = float(np.mean(losses)) if losses else 0.0
        loss_scale = float(np.std(losses)) if len(losses) > 1 else 1.0
        loss_scale = max(loss_scale, 1e-6)
        total = max(1, sum(s.usage_count for s in self.stats))

        scores = []
        for stat in self.stats:
            under_sampled = 1.0 - (float(stat.usage_count) / float(max_usage))
            recency = 1.0 if stat.last_step < 0 else min(
                1.0, float(self.step - stat.last_step) / max(1, self.num_tasks)
            )
            diversity = 0.5 * under_sampled + 0.5 * recency
            entropy = float(np.tanh(stat.uncertainty_ema))
            if stat.loss_ema is None:
                difficulty = 0.0
            else:
                difficulty = float(np.tanh((float(stat.loss_ema) - loss_center) / loss_scale))
            exploration = np.sqrt(np.log(total + 1.0) / (stat.usage_count + 1.0))
            score = (
                self.config.asr_diversity_weight * diversity
                + self.config.asr_entropy_weight * entropy
                + self.config.asr_difficulty_weight * difficulty
                + 0.05 * exploration
            )
            scores.append(score)
        return np.asarray(scores, dtype=np.float64)

    def _derts_proxy_scores(self) -> np.ndarray:
        """Online proxy for DERTS-style robust task subset selection.

        Official DERTS uses task-pool gradient approximation. This repository
        only has online scalar feedback at scheduler time, so this proxy favors
        representative, under-sampled tasks while penalizing high-uncertainty
        outliers that may behave like noisy tasks.
        """
        max_usage = max(1, max(s.usage_count for s in self.stats))
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        loss_center = float(np.mean(losses)) if losses else 0.0
        loss_scale = float(np.std(losses)) if len(losses) > 1 else 1.0
        loss_scale = max(loss_scale, 1e-6)

        scores = []
        for stat in self.stats:
            if stat.loss_ema is None:
                loss_z = 0.0
            else:
                loss_z = (float(stat.loss_ema) - loss_center) / loss_scale
            representative = -abs(float(np.tanh(loss_z)))
            coverage = 1.0 - (float(stat.usage_count) / float(max_usage))
            robust_penalty = float(np.tanh(stat.uncertainty_ema))
            reward = float(stat.reward_ema)
            score = (
                representative
                + self.config.derts_coverage_bonus * coverage
                + reward
                - self.config.derts_noise_penalty * robust_penalty
            )
            scores.append(score)
        return np.asarray(scores, dtype=np.float64)

    def _gcp_scores(self) -> np.ndarray:
        return np.log(np.maximum(self.gcp_weights, self.config.min_weight))

    def _update_gcp_weight(self, task_id: int) -> None:
        stat = self.stats[int(task_id)]
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        loss_center = float(np.mean(losses)) if losses else 0.0
        loss_scale = float(np.std(losses)) if len(losses) > 1 else 1.0
        loss_scale = max(loss_scale, 1e-6)

        if stat.loss_ema is None:
            utility = float(stat.reward_ema)
        else:
            loss_z = (float(stat.loss_ema) - loss_center) / loss_scale
            utility = float(np.tanh(loss_z + stat.uncertainty_ema + stat.reward_ema))

        weights = np.asarray(self.gcp_weights, dtype=np.float64).copy()
        weights[task_id] = np.power(
            max(weights[task_id], self.config.min_weight),
            float(self.config.gcp_tau),
        ) * np.exp(float(self.config.gcp_alpha) * utility)

        weights = np.maximum(weights, self.config.min_weight)
        weights = weights / max(float(weights.sum()), self.config.min_weight) * self.num_tasks

        min_rate = float(np.clip(self.config.gcp_min_rate, 0.0, 1.0))
        if min_rate > 0:
            probs = weights / float(weights.sum())
            probs = probs * (1.0 - min_rate) + min_rate / float(self.num_tasks)
            weights = probs * self.num_tasks

        max_ratio = max(float(self.config.gcp_max_ratio), 1.0)
        min_w = max(float(weights.min()), self.config.min_weight)
        max_w = float(weights.max())
        if max_w / min_w > max_ratio:
            cap = min_w * max_ratio
            weights = np.minimum(weights, cap)
            weights = weights / max(float(weights.sum()), self.config.min_weight) * self.num_tasks

        self.gcp_weights = weights

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        temp = max(float(self.config.temperature), 1e-6)
        shifted = (scores - np.max(scores)) / temp
        weights = np.exp(shifted)
        return self._normalize(weights)

    def _argmax_probs(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        best = np.flatnonzero(scores == np.max(scores))
        probs = np.full(self.num_tasks, self.config.min_weight, dtype=np.float64)
        probs[best] = 1.0 / float(len(best))
        return self._normalize(probs)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, self.config.min_weight)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            return np.ones(self.num_tasks, dtype=np.float64) / float(self.num_tasks)
        return weights / total

    def _recent_mean(self, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.mean(values))
