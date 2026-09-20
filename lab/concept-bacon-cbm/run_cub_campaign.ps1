# CUB 3-seed reproduction campaign (paper numbers). Single GPU, sequential.
# Canonical method = B (recttree W64 + L0 + hardness-anneal/binarize).
#   B  W64 + binarize            x3 seeds  (main result)
#   C  W8  + binarize            x3 seeds  (width study)
#   D  W4  + binarize            x3 seeds  (width study)
#   E  W64 + edge-l0 3e-4 + bin  x3 seeds  (rule-size lever)
#   A  W64 no-binarize           x1 seed   (hardness-anneal ABLATION)
# Ensemble (MoM = confidence router over {B,C,D}) is computed POST-hoc by
#   route_ensemble.py once these finish -- no training here.
#
# Run:  cd lab/concept-bacon-cbm ;  ./run_cub_campaign.ps1
# Logs: results/campaign/<id>_s<seed>.txt   Checkpoints tagged binw_s<seed> / s<seed>.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONWARNINGS = "ignore"
New-Item -ItemType Directory -Force -Path results\campaign | Out-Null

$base = @(
    "--head","recttree","--backbone","inception_v3","--attr312",
    "--concept-lam","0.03","--concept-imbalance","--epochs","200",
    "--rect-depth","3","--max-parents","2","--edge-l0-warmup","40",
    "--coefficients","--negation","--confusion-ovr","--confusion-hardness","1",
    "--early-stop-patience","40","--min-freeze-frac","0.4","--batch-size","16","--workers","4",
    "--init-from","saved\cub_joint_pretrain_inception_v3_k312_iw_reg_150ep.pt"
)
$binw = @("--binarize-lam","0.5","--binarize-warmup","60")

# each run: id, rect-width, edge-l0, use-binarize, seed
$runs = @(
    @{id="B"; w=64; l0="1e-4"; bin=$true;  seed=0},
    @{id="B"; w=64; l0="1e-4"; bin=$true;  seed=1},
    @{id="B"; w=64; l0="1e-4"; bin=$true;  seed=2},
    @{id="E"; w=64; l0="3e-4"; bin=$true;  seed=0},
    @{id="E"; w=64; l0="3e-4"; bin=$true;  seed=1},
    @{id="E"; w=64; l0="3e-4"; bin=$true;  seed=2},
    @{id="C"; w=8;  l0="1e-4"; bin=$true;  seed=0},
    @{id="C"; w=8;  l0="1e-4"; bin=$true;  seed=1},
    @{id="C"; w=8;  l0="1e-4"; bin=$true;  seed=2},
    @{id="D"; w=4;  l0="1e-4"; bin=$true;  seed=0},
    @{id="D"; w=4;  l0="1e-4"; bin=$true;  seed=1},
    @{id="D"; w=4;  l0="1e-4"; bin=$true;  seed=2},
    @{id="A"; w=64; l0="1e-4"; bin=$false; seed=0}   # ablation (no binarize)
)

$i = 0
foreach ($r in $runs) {
    $i++
    $tag = if ($r.bin) { "binw_s$($r.seed)" } else { "s$($r.seed)" }
    $log = "results\campaign\$($r.id)_w$($r.w)_l0$($r.l0)_$tag.txt"
    # idempotent resume: skip runs whose log already recorded a DONE accuracy.
    if ((Test-Path $log) -and (Select-String -Path $log -Pattern "DONE.*acc" -Quiet)) {
        Write-Host "===== [$i/$($runs.Count)] $($r.id) seed $($r.seed) already DONE -> skip ($log) ====="
        continue
    }
    $extra = @("--rect-width","$($r.w)","--edge-l0","$($r.l0)","--seed","$($r.seed)","--tag",$tag)
    if ($r.bin) { $extra += $binw }
    Write-Host "===== [$i/$($runs.Count)] $($r.id) W$($r.w) l0=$($r.l0) bin=$($r.bin) seed=$($r.seed) -> $log ====="
    $cmdargs = $base + $extra
    py -3 -u run_cub_fulltree.py @cmdargs 2>&1 | Tee-Object $log
    Write-Host "----- done $($r.id) seed $($r.seed) (exit $LASTEXITCODE) -----`n"
}
Write-Host "CUB CAMPAIGN COMPLETE. Aggregate with: py -3 aggregate_seeds.py results\campaign"
