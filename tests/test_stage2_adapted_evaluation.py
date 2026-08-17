from types import SimpleNamespace

import torch

from scripts.eval_stage2_scheduler_run import (
    EVALUATION_PROTOCOL,
    artifact_matches_protocol,
    balanced_support_indices,
    predictions_are_degenerate,
    resolve_adaptation_config,
)


class LabelDataset:
    def __init__(self):
        self.labels = [0] * 10 + [1] * 10

    def __len__(self):
        return len(self.labels)


def test_adaptation_defaults_come_from_checkpoint():
    args = SimpleNamespace(
        adaptation_support_size=None,
        adaptation_inner_lr=None,
        adaptation_inner_steps=None,
    )
    checkpoint = {
        "config": {"n_support": 12, "inner_lr": 0.025, "inner_steps": 3}
    }

    config = resolve_adaptation_config(args, checkpoint)

    assert config == {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "adaptation_support_size": 12,
        "adaptation_inner_lr": 0.025,
        "adaptation_inner_steps": 3,
    }


def test_old_unadapted_artifacts_are_not_resumed():
    config = {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "adaptation_support_size": 16,
        "adaptation_inner_lr": 0.001,
        "adaptation_inner_steps": 1,
    }
    assert not artifact_matches_protocol({"preds": [1.0, 1.0]}, config)
    assert artifact_matches_protocol({**config, "preds": [0.2, 0.8]}, config)


def test_support_split_is_balanced_and_disjoint():
    import random

    dataset = LabelDataset()
    support, query = balanced_support_indices(dataset, 8, random.Random(7))

    assert len(support) == 8
    assert len(query) == 12
    assert set(support).isdisjoint(query)
    assert [dataset.labels[index] for index in support].count(0) == 4
    assert [dataset.labels[index] for index in support].count(1) == 4


def test_degenerate_prediction_guard():
    assert predictions_are_degenerate([1.0, 1.0, 1.0])
    assert predictions_are_degenerate([float("nan"), 0.5])
    assert not predictions_are_degenerate(torch.tensor([0.1, 0.9]))
