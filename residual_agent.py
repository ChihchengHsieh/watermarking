from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ResidualTaskStats:
    """Online feedback kept for one SpiderMark attack task."""

    usage_count: int = 0
    last_step: int = -1
    loss_ema: Optional[float] = None
    prev_loss_ema: Optional[float] = None
    fail_ema: float = 0.0
    reward_ema: float = 0.0
    uncertainty_ema: float = 0.0
    recent_losses: List[float] = field(default_factory=list)
    recent_rewards: List[float] = field(default_factory=list)


@dataclass
class ResidualTaskControllerConfig:
    residual_scale: float = 0.15
    lr: float = 0.05
    ema_beta: float = 0.85
    reward_beta: float = 0.9
    min_weight: float = 1e-4
    temperature: float = 1.0
    seed: int = 0


@dataclass
class LLMTaskControllerConfig(ResidualTaskControllerConfig):
    model: str = "gpt-5.1"
    api_key_env: str = "OPENAI_API_KEY"
    api_url: str = "https://api.openai.com/v1/responses"
    api_format: str = "responses"
    timeout_sec: float = 20.0
    call_interval: int = 100
    recent_window: int = 20
    max_context_tasks: int = 12
    fallback_on_error: bool = True
    log_errors: bool = False
    residual_scale_final: Optional[float] = None
    residual_scale_switch_step: Optional[int] = None


class ResidualTaskController:
    """
    Agentic residual controller over an existing task scheduler.

    The base scheduler remains the stable policy. This controller learns a small
    bounded correction over per-task weights:

        final_w = normalize(base_w + residual_scale * tanh(phi(task)^T theta))

    Feedback is intentionally lightweight: callers can pass query/outer loss,
    validation gain, failure rate, or an explicit reward after the downstream
    meta-training step.
    """

    feature_names = (
        "base_weight",
        "recent_loss",
        "loss_delta",
        "fail_rate",
        "usage",
        "recency",
        "uncertainty",
        "reward",
    )

    def __init__(
        self,
        num_tasks: int,
        task_names: Optional[Sequence[str]] = None,
        config: Optional[ResidualTaskControllerConfig] = None,
    ):
        if num_tasks <= 0:
            raise ValueError("ResidualTaskController requires at least one task.")

        self.num_tasks = int(num_tasks)
        self.task_names = list(task_names or [str(i) for i in range(num_tasks)])
        self.config = config or ResidualTaskControllerConfig()
        self.stats: List[ResidualTaskStats] = [
            ResidualTaskStats() for _ in range(self.num_tasks)
        ]
        self.step = 0
        self.rng = np.random.default_rng(self.config.seed)
        self.theta = np.zeros(len(self.feature_names), dtype=np.float64)
        self.global_context: Dict[str, object] = {}
        self._last_features: Optional[np.ndarray] = None
        self._last_probs: Optional[np.ndarray] = None
        self._last_task_id: Optional[int] = None

    def base_weights(self, mode: str = "uniform", cycle_idx: Optional[int] = None):
        if mode == "cycle":
            weights = np.full(self.num_tasks, self.config.min_weight, dtype=np.float64)
            weights[int(cycle_idx or 0) % self.num_tasks] = 1.0
            return self._normalize(weights)
        if mode != "uniform":
            raise ValueError(f"Unsupported residual base scheduler: {mode}")
        return np.ones(self.num_tasks, dtype=np.float64) / float(self.num_tasks)

    def propose(
        self,
        base_weights: Sequence[float],
        *,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        base = self._normalize(np.asarray(base_weights, dtype=np.float64))
        features = self._build_features(base, self.step if step is None else int(step))
        delta = self.config.residual_scale * np.tanh(features @ self.theta)
        final = self._normalize(np.maximum(base + delta, self.config.min_weight))
        if self.config.temperature != 1.0:
            temp = max(float(self.config.temperature), 1e-6)
            final = self._normalize(np.power(final, 1.0 / temp))

        self._last_features = features
        self._last_probs = final
        return final, {"base_weights": base, "delta": delta, "features": features}

    def sample(
        self,
        base_weights: Sequence[float],
        *,
        step: Optional[int] = None,
        rng=None,
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        probs, info = self.propose(base_weights, step=step)
        random_source = rng if rng is not None else self.rng
        if isinstance(random_source, np.random.Generator):
            task_id = int(random_source.choice(np.arange(self.num_tasks), p=probs))
        elif hasattr(random_source, "choices"):
            task_id = int(random_source.choices(range(self.num_tasks), weights=probs, k=1)[0])
        else:
            task_id = int(np.random.choice(np.arange(self.num_tasks), p=probs))

        self._last_task_id = task_id
        stat = self.stats[task_id]
        stat.usage_count += 1
        stat.last_step = self.step
        self.step += 1
        return task_id, info

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
        beta = self.config.ema_beta

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
            stat.recent_losses = stat.recent_losses[-int(getattr(self.config, "recent_window", 20)) :]

        if fail_rate is not None:
            fail_rate = float(np.clip(fail_rate, 0.0, 1.0))
            stat.fail_ema = beta * stat.fail_ema + (1 - beta) * fail_rate

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
        stat.recent_rewards = stat.recent_rewards[-int(getattr(self.config, "recent_window", 20)) :]

        if self._last_features is not None:
            features = self._last_features[task_id]
        else:
            features = self._build_features(
                self.base_weights("uniform"), self.step
            )[task_id]

        # Tiny online scoring-head update. Positive reward raises similar states;
        # negative reward suppresses them. Clipping keeps this a residual policy.
        self.theta += self.config.lr * reward * features
        self.theta *= 0.999
        self.theta = np.clip(self.theta, -2.0, 2.0)
        return reward

    def update_global_context(self, **kwargs) -> None:
        self.global_context.update(
            {
                key: value
                for key, value in kwargs.items()
                if value is None or isinstance(value, (int, float, str, bool, list, tuple, dict))
            }
        )

    def snapshot(self) -> Dict[str, object]:
        return {
            "theta": self.theta.copy(),
            "feature_names": self.feature_names,
            "tasks": [
                {
                    "task_id": i,
                    "task_name": self.task_names[i],
                    "usage_count": s.usage_count,
                    "loss_ema": s.loss_ema,
                    "fail_ema": s.fail_ema,
                    "reward_ema": s.reward_ema,
                    "uncertainty_ema": s.uncertainty_ema,
                    "recent_loss_mean": self._recent_mean(s.recent_losses),
                    "recent_loss_std": self._recent_std(s.recent_losses),
                    "recent_sample_count": len(s.recent_losses),
                    "recent_reward_mean": self._recent_mean(s.recent_rewards),
                }
                for i, s in enumerate(self.stats)
            ],
            "global_context": dict(self.global_context),
        }

    def _recent_mean(self, values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(np.mean(values))

    def _recent_std(self, values: Sequence[float]) -> Optional[float]:
        if len(values) < 2:
            return 0.0 if values else None
        return float(np.std(values))

    def _build_features(self, base: np.ndarray, step: int) -> np.ndarray:
        max_usage = max(1, max(s.usage_count for s in self.stats))
        losses = [s.loss_ema for s in self.stats if s.loss_ema is not None]
        loss_center = float(np.mean(losses)) if losses else 0.0
        loss_scale = float(np.std(losses)) if len(losses) > 1 else 1.0
        loss_scale = max(loss_scale, 1e-6)

        rows = []
        for i, stat in enumerate(self.stats):
            recent_loss = 0.0
            loss_delta = 0.0
            if stat.loss_ema is not None:
                recent_loss = (stat.loss_ema - loss_center) / loss_scale
            if stat.loss_ema is not None and stat.prev_loss_ema is not None:
                loss_delta = stat.prev_loss_ema - stat.loss_ema

            if stat.last_step < 0:
                recency = 1.0
            else:
                recency = min(1.0, float(step - stat.last_step) / max(1, self.num_tasks))

            rows.append(
                [
                    float(base[i]),
                    float(np.tanh(recent_loss)),
                    float(np.tanh(loss_delta)),
                    float(stat.fail_ema),
                    float(stat.usage_count) / float(max_usage),
                    recency,
                    float(np.tanh(stat.uncertainty_ema)),
                    float(stat.reward_ema),
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.maximum(weights, self.config.min_weight)
        total = float(weights.sum())
        if total <= 0 or not np.isfinite(total):
            return np.ones(self.num_tasks, dtype=np.float64) / float(self.num_tasks)
        return weights / total


class LLMResidualTaskController(ResidualTaskController):
    """
    LLM API-backed task controller.

    The LLM sees a compact task-memory summary and returns residual corrections
    over the base scheduler weights. The correction is still bounded and
    normalized locally, so the LLM acts as an agentic layer instead of replacing
    the scheduler.
    """

    def __init__(
        self,
        num_tasks: int,
        task_names: Optional[Sequence[str]] = None,
        config: Optional[LLMTaskControllerConfig] = None,
    ):
        super().__init__(
            num_tasks=num_tasks,
            task_names=task_names,
            config=config or LLMTaskControllerConfig(),
        )
        self.config: LLMTaskControllerConfig
        self._cached_delta = np.zeros(self.num_tasks, dtype=np.float64)
        self._last_llm_step = -10**9
        self.last_llm_response: Optional[Dict[str, object]] = None
        self.last_llm_error: Optional[str] = None
        self.llm_call_count = 0
        self.llm_total_time_sec = 0.0
        self.last_llm_time_sec = 0.0

    def propose(
        self,
        base_weights: Sequence[float],
        *,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        base = self._normalize(np.asarray(base_weights, dtype=np.float64))
        current_step = self.step if step is None else int(step)
        features = self._build_features(base, current_step)

        active_scale = self._active_residual_scale(current_step)
        local_delta = active_scale * np.tanh(features @ self.theta)
        llm_delta = self._cached_delta
        should_call = (current_step - self._last_llm_step) >= self.config.call_interval

        if should_call:
            try:
                llm_delta = self._query_llm_delta(base, features, current_step)
                self._cached_delta = llm_delta
                self._last_llm_step = current_step
                self.last_llm_error = None
            except Exception as exc:
                self.last_llm_error = str(exc)
                self._last_llm_step = current_step
                if self.config.log_errors:
                    print(f"[LLMResidualTaskController] {exc}")
                if not self.config.fallback_on_error:
                    raise

        delta = local_delta + llm_delta
        final = self._normalize(np.maximum(base + delta, self.config.min_weight))
        if self.config.temperature != 1.0:
            temp = max(float(self.config.temperature), 1e-6)
            final = self._normalize(np.power(final, 1.0 / temp))

        self._last_features = features
        self._last_probs = final
        return final, {
            "base_weights": base,
            "delta": delta,
            "local_delta": local_delta,
            "llm_delta": llm_delta,
            "active_residual_scale": float(active_scale),
            "features": features,
        }

    def snapshot(self) -> Dict[str, object]:
        snap = super().snapshot()
        snap.update(
            {
                "llm_model": self.config.model,
                "llm_call_interval": self.config.call_interval,
                "residual_scale_initial": self.config.residual_scale,
                "residual_scale_final": self.config.residual_scale_final,
                "residual_scale_switch_step": self.config.residual_scale_switch_step,
                "active_residual_scale": self._active_residual_scale(self.step),
                "last_llm_response": self.last_llm_response,
                "last_llm_error": self.last_llm_error,
                "cached_llm_delta": self._cached_delta.copy(),
                "llm_call_count": self.llm_call_count,
                "llm_total_time_sec": self.llm_total_time_sec,
                "last_llm_time_sec": self.last_llm_time_sec,
                "avg_llm_time_sec": (
                    self.llm_total_time_sec / max(1, self.llm_call_count)
                ),
            }
        )
        return snap

    def _query_llm_delta(
        self,
        base: np.ndarray,
        features: np.ndarray,
        step: int,
    ) -> np.ndarray:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.config.api_key_env} is not set; cannot call LLM controller."
            )

        import time

        payload = self._build_llm_payload(base, features, step)
        req = urllib.request.Request(
            self.config.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
        finally:
            self.last_llm_time_sec = time.time() - t0
            self.llm_total_time_sec += self.last_llm_time_sec
            self.llm_call_count += 1

        data = json.loads(raw)
        text = self._extract_output_text(data)
        parsed = json.loads(self._strip_json_markdown(text))
        self.last_llm_response = parsed

        delta = np.asarray(parsed.get("delta_weights", []), dtype=np.float64)
        if delta.shape != (self.num_tasks,):
            raise ValueError(
                f"LLM returned {delta.shape} delta weights, expected {(self.num_tasks,)}."
            )
        active_scale = self._active_residual_scale(step)
        delta = np.clip(delta, -active_scale, active_scale)
        return delta

    def _active_residual_scale(self, step: int) -> float:
        final_scale = self.config.residual_scale_final
        switch_step = self.config.residual_scale_switch_step
        if final_scale is None or switch_step is None:
            return float(self.config.residual_scale)
        if int(step) >= int(switch_step):
            return float(final_scale)
        return float(self.config.residual_scale)

    def _build_llm_payload(
        self,
        base: np.ndarray,
        features: np.ndarray,
        step: int,
    ) -> Dict[str, object]:
        task_rows = []
        for i in range(self.num_tasks):
            stat = self.stats[i]
            task_rows.append(
                {
                    "task_id": i,
                    "task_name": self.task_names[i],
                    "base_weight": round(float(base[i]), 6),
                    "usage_count": stat.usage_count,
                    "loss_ema": stat.loss_ema,
                    "fail_ema": round(float(stat.fail_ema), 6),
                    "reward_ema": round(float(stat.reward_ema), 6),
                    "uncertainty_ema": round(float(stat.uncertainty_ema), 6),
                    "recent_sample_count": len(stat.recent_losses),
                    "recent_loss_mean": self._round_optional(
                        self._recent_mean(stat.recent_losses)
                    ),
                    "recent_loss_std": self._round_optional(
                        self._recent_std(stat.recent_losses)
                    ),
                    "recent_reward_mean": self._round_optional(
                        self._recent_mean(stat.recent_rewards)
                    ),
                    "last_sampled_step": stat.last_step,
                    "features": {
                        name: round(float(value), 6)
                        for name, value in zip(self.feature_names, features[i])
                    },
                }
            )

        active_scale = self._active_residual_scale(step)
        system_prompt = (
            # TODO: Explain SpiderMark
            "You are an LLM-based residual controller for the SpiderMark "
            "meta-training task scheduler. You must not replace the base "
            "scheduler. Your job is to propose residual corrections "
            "delta_weights on top of base_weights. The training code will clip "
            "and normalize your output. With a large active_residual_scale, your deltas "
            "can strongly reshape the final task weights, but they must still be "
            "justified by the compact training signals. Prefer stable corrections. "
            "Increase a task only when the state suggests useful training signal: "
            "persistent high loss, recent failure, high uncertainty, or "
            "under-sampling. Decrease a task when it appears over-sampled, "
            "already improving, or uninformative. Do not chase noisy high loss "
            "by itself. If evidence is weak, return deltas close to zero. Keep "
            "corrections roughly zero-sum so you shift priority rather than "
            "globally inflating all tasks."
        )
        user_prompt = {
            "step": step,
            "residual_scale": active_scale,
            "active_residual_scale": active_scale,
            "residual_scale_initial": self.config.residual_scale,
            "residual_scale_final": self.config.residual_scale_final,
            "residual_scale_switch_step": self.config.residual_scale_switch_step,
            "task_count": self.num_tasks,
            "global_training_state": self._json_ready(self.global_context),
            "tasks": task_rows,
            "instruction": (
                "Return one delta weight per task, in task_id order. Positive "
                "delta means sample this task more often; negative delta means "
                "sample it less often. Keep each value within "
                "[-active_residual_scale, active_residual_scale], and keep the average delta "
                "near zero. Do not output a new scheduler policy; output only "
                "residual corrections. Use base_weight as the stable prior. "
                "Feature meanings: base_weight is the base scheduler prior; "
                "usage is exposure count normalized against other tasks; recency "
                "is how long since the task was sampled; recent_loss is relative "
                "difficulty; loss_delta is improvement when positive; fail_rate "
                "is recent failure signal; uncertainty is noisy or unstable "
                "feedback; reward is recent downstream usefulness. The "
                "global_training_state summarizes the verifier update from the "
                "most recent completed outer iteration, including meta-loss "
                "trend, gradient norm, learning rate, and recent task counts. "
                "Use recent_sample_count, recent_loss_mean, recent_loss_std, "
                "and recent_reward_mean to avoid overreacting to a single noisy "
                "loss. Return only a valid JSON object with delta_weights and "
                "rationale. Do not wrap it in Markdown."
            ),
        }

        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "delta_weights": {
                    "type": "array",
                    "items": {
                        "type": "number",
                        "minimum": -active_scale,
                        "maximum": active_scale,
                    },
                    "minItems": self.num_tasks,
                    "maxItems": self.num_tasks,
                },
                "rationale": {"type": "string", "maxLength": 240},
            },
            "required": ["delta_weights", "rationale"],
        }

        user_content = "JSON task scheduler state:\n" + json.dumps(
            user_prompt, ensure_ascii=True
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        api_format = self.config.api_format.lower()
        if api_format == "chat_completions":
            return {
                "model": self.config.model,
                "messages": messages,
                "temperature": 0.2,
                "top_p": 0.7,
                "max_tokens": 512,
                "stream": False,
            }
        if api_format != "responses":
            raise ValueError(f"Unsupported LLM API format: {self.config.api_format}")

        return {
            "model": self.config.model,
            "input": messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "spidermark_task_scheduler_delta",
                    "strict": True,
                    "schema": schema,
                }
            },
        }

    def _build_openai_payload(
        self,
        base: np.ndarray,
        features: np.ndarray,
        step: int,
    ) -> Dict[str, object]:
        return self._build_llm_payload(base, features, step)

    def _round_optional(self, value: Optional[float], digits: int = 6):
        if value is None:
            return None
        return round(float(value), digits)

    def _json_ready(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): self._json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_ready(v) for v in value]
        return value

    def _extract_output_text(self, data: Dict[str, object]) -> str:
        if isinstance(data.get("output_text"), str):
            return str(data["output_text"])

        choices = data.get("choices", []) or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]

        chunks = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        if chunks:
            return "".join(chunks)
        raise ValueError("LLM response did not contain output text.")

    def _strip_json_markdown(self, text: str) -> str:
        text = text.strip()
        if not text.startswith("```"):
            return text
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
