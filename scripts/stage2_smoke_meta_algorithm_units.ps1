param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$code = @'
from tests.test_stage2_meta_algorithms import (
    test_fomaml_step_returns_differentiable_outer_loss,
    test_maml_step_returns_differentiable_outer_loss,
    test_anil_inner_update_changes_head_only,
    test_reptile_step_returns_adapted_parameter_targets,
    test_matching_net_step_returns_differentiable_outer_loss,
    test_proto_net_step_returns_differentiable_outer_loss,
    test_r2d2_ridge_step_returns_differentiable_outer_loss,
)

test_fomaml_step_returns_differentiable_outer_loss()
test_maml_step_returns_differentiable_outer_loss()
test_anil_inner_update_changes_head_only()
test_reptile_step_returns_adapted_parameter_targets()
test_matching_net_step_returns_differentiable_outer_loss()
test_proto_net_step_returns_differentiable_outer_loss()
test_r2d2_ridge_step_returns_differentiable_outer_loss()
print("stage2 meta algorithm unit smoke passed")
'@

$code | & $PythonExe -
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 meta algorithm unit smoke failed."
}
