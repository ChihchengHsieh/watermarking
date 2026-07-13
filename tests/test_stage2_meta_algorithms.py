import torch
import torch.nn as nn

from meta.meta_module import MetaModule
from scripts.run_stage2_scheduler_training import (
    matching_net_task_step,
    meta_train_step,
    proto_net_task_step,
    r2d2_ridge_task_step,
    reptile_task_step,
    update_adapted_model,
)


class ToyHead(MetaModule):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.tensor([[0.4, -0.2], [-0.3, 0.5]], requires_grad=True))
        self.register_buffer("bias", torch.zeros(2, requires_grad=True))

    def named_leaves(self):
        return [("weight", self.weight), ("bias", self.bias)]

    def forward(self, x):
        return x @ self.weight + self.bias


class ToyMetaModel(MetaModule):
    def __init__(self):
        super().__init__()
        self.register_buffer("backbone_weight", torch.tensor([[0.7, -0.1], [0.2, 0.6]], requires_grad=True))
        self.head = ToyHead()

    def named_leaves(self):
        return [("backbone_weight", self.backbone_weight)]

    def get_features(self, x):
        return x @ self.backbone_weight

    def forward(self, x):
        return self.head(self.get_features(x))


def make_toy_model():
    return ToyMetaModel()


def make_task_batch():
    return {
        "support": {
            "x": torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]]),
            "y": torch.tensor([[0, 1, 0, 1]]),
        },
        "query": {
            "x": torch.tensor([[[0.5, 0.5], [1.0, -1.0], [-0.5, 1.0], [0.2, -0.3]]]),
            "y": torch.tensor([[0, 1, 1, 0]]),
        },
        "task_name": ["toy_attack"],
    }


def named_param_dict(model):
    return {name: param for name, param in model.named_params()}


def test_fomaml_step_returns_differentiable_outer_loss():
    base_model = make_toy_model()
    crit = nn.CrossEntropyLoss()

    outer_loss, task_name, inner_loss = meta_train_step(
        base_model=base_model,
        make_new_model=make_toy_model,
        task_batch=make_task_batch(),
        crit=crit,
        device="cpu",
        include_psnr_l1=False,
        inner_lr=0.1,
        inner_steps=1,
        first_order=True,
    )

    assert task_name == "toy_attack"
    assert inner_loss is not None
    assert outer_loss.requires_grad

    outer_loss.backward()
    grads = [param.grad for param in base_model.params()]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_maml_step_returns_differentiable_outer_loss():
    base_model = make_toy_model()
    crit = nn.CrossEntropyLoss()

    outer_loss, task_name, inner_loss = meta_train_step(
        base_model=base_model,
        make_new_model=make_toy_model,
        task_batch=make_task_batch(),
        crit=crit,
        device="cpu",
        include_psnr_l1=False,
        inner_lr=0.1,
        inner_steps=1,
        first_order=False,
    )

    assert task_name == "toy_attack"
    assert inner_loss is not None
    assert outer_loss.requires_grad

    outer_loss.backward()
    grads = [param.grad for param in base_model.params()]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_anil_inner_update_changes_head_only():
    base_model = make_toy_model()
    adapted = make_toy_model()
    adapted.copy(base_model, same_var=True)
    crit = nn.CrossEntropyLoss()
    task = make_task_batch()
    xs = task["support"]["x"].squeeze(0)
    ys = task["support"]["y"].squeeze(0)

    before = {name: param.detach().clone() for name, param in adapted.named_params()}
    loss = crit(adapted(xs), ys)
    loss.backward()
    update_adapted_model(
        adapted,
        inner_lr=0.1,
        first_order=True,
        trainable_prefixes=("head.",),
    )

    after = named_param_dict(adapted)
    assert torch.allclose(after["backbone_weight"], before["backbone_weight"])
    assert not torch.allclose(after["head.weight"], before["head.weight"])
    assert not torch.allclose(after["head.bias"], before["head.bias"])


def test_reptile_step_returns_adapted_parameter_targets():
    base_model = make_toy_model()
    crit = nn.CrossEntropyLoss()

    query_loss, task_name, adapted_params = reptile_task_step(
        base_model=base_model,
        make_new_model=make_toy_model,
        task_batch=make_task_batch(),
        crit=crit,
        device="cpu",
        include_psnr_l1=False,
        inner_lr=0.1,
        inner_steps=1,
    )

    assert task_name == "toy_attack"
    assert query_loss > 0
    assert set(adapted_params) == {"backbone_weight", "head.weight", "head.bias"}
    for name, param in adapted_params.items():
        assert param.shape == named_param_dict(base_model)[name].shape
        assert torch.isfinite(param).all()


def test_r2d2_ridge_step_returns_differentiable_outer_loss():
    base_model = make_toy_model()
    crit = nn.CrossEntropyLoss()

    outer_loss, task_name = r2d2_ridge_task_step(
        base_model=base_model,
        task_batch=make_task_batch(),
        crit=crit,
        device="cpu",
        include_psnr_l1=False,
        ridge_lambda=1e-2,
    )

    assert task_name == "toy_attack"
    assert outer_loss.requires_grad
    outer_loss.backward()
    grads = [param.grad for param in base_model.params()]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_proto_net_step_returns_differentiable_outer_loss():
    base_model = make_toy_model()
    crit = nn.CrossEntropyLoss()

    outer_loss, task_name = proto_net_task_step(
        base_model=base_model,
        task_batch=make_task_batch(),
        crit=crit,
        device="cpu",
        include_psnr_l1=False,
    )

    assert task_name == "toy_attack"
    assert outer_loss.requires_grad
    outer_loss.backward()
    grads = [param.grad for param in base_model.params()]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)


def test_matching_net_step_returns_differentiable_outer_loss():
    base_model = make_toy_model()
    crit = nn.CrossEntropyLoss()

    outer_loss, task_name = matching_net_task_step(
        base_model=base_model,
        task_batch=make_task_batch(),
        crit=crit,
        device="cpu",
        include_psnr_l1=False,
    )

    assert task_name == "toy_attack"
    assert outer_loss.requires_grad
    outer_loss.backward()
    grads = [param.grad for param in base_model.params()]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in grads)
