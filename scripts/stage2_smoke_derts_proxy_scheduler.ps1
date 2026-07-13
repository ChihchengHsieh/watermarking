param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$code = @'
import numpy as np

from scheduler_baselines import BaselineSchedulerConfig, BaselineTaskSchedulerController

ctrl = BaselineTaskSchedulerController(
    num_tasks=4,
    task_names=["clean", "jpeg", "crop", "occlusion"],
    config=BaselineSchedulerConfig(mode="derts_proxy", seed=7, warmup_rounds=1),
)

seen = []
for step in range(12):
    task_id, info = ctrl.sample()
    probs = info["probs"]
    assert probs.shape == (4,)
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(), 1.0)
    loss = [0.40, 0.75, 0.55, 1.40][task_id]
    ctrl.update(task_id, loss=loss)
    seen.append(task_id)

snap = ctrl.snapshot()
assert snap["mode"] == "derts_proxy"
assert snap["last_probs"] is not None
assert len(snap["tasks"]) == 4
assert sorted(set(seen[:4])) == [0, 1, 2, 3]
print("stage2 derts_proxy scheduler smoke passed")
'@

$code | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "DERTS proxy scheduler smoke failed."
}
