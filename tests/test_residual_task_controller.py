import numpy as np

from residual_agent import (
    LLMResidualTaskController,
    LLMTaskControllerConfig,
    ResidualTaskController,
    ResidualTaskControllerConfig,
)


def test_residual_controller_starts_from_base_policy():
    agent = ResidualTaskController(
        num_tasks=3,
        config=ResidualTaskControllerConfig(seed=1, residual_scale=0.1),
    )
    base = np.array([0.2, 0.3, 0.5])

    final, info = agent.propose(base)

    np.testing.assert_allclose(final.sum(), 1.0)
    np.testing.assert_allclose(final, base)
    np.testing.assert_allclose(info["delta"], np.zeros(3))


def test_residual_controller_feedback_changes_policy():
    agent = ResidualTaskController(
        num_tasks=3,
        config=ResidualTaskControllerConfig(seed=1, residual_scale=0.2, lr=0.5),
    )
    base = agent.base_weights("uniform")
    agent.sample(base)

    agent.update(0, reward=1.0)
    updated, _ = agent.propose(base)

    assert updated.shape == (3,)
    assert np.isclose(updated.sum(), 1.0)
    assert not np.allclose(updated, base)


def test_llm_controller_builds_structured_output_payload():
    agent = LLMResidualTaskController(
        num_tasks=2,
        task_names=["crop", "jpeg"],
        config=LLMTaskControllerConfig(model="gpt-test", call_interval=1),
    )
    base = agent.base_weights("uniform")
    features = agent._build_features(base, step=0)

    payload = agent._build_openai_payload(base, features, step=0)

    assert payload["model"] == "gpt-test"
    assert payload["text"]["format"]["type"] == "json_schema"
    assert payload["text"]["format"]["schema"]["properties"]["delta_weights"]["minItems"] == 2
