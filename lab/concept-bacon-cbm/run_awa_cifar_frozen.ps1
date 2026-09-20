# AwA2 + CIFAR-100 3-seed FROZEN-ENCODER (annotated-mode) reproduction campaign.
# Same config B as run_awa_cifar_campaign.ps1 (recttree W64 + L0 + binarize, 200ep,
# 3 seeds) but with the concept encoder FROZEN during the tree stage (annotated
# mode: supervised concepts kept faithful, only the graded-logic head trains).
# Uses --tag frozen so logs AND checkpoints are DISTINCT from the earlier leaked
# (unfrozen) runs -- fully non-destructive. Idempotent: skips runs whose log has DONE.
#
# Run:  & c:\School\bacon-net\lab\concept-bacon-cbm\run_awa_cifar_frozen.ps1
# Aggregate:  py -3 aggregate_seeds.py results\campaign_awa_frozen  (and _cifar_frozen)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONWARNINGS = "ignore"
$env:CUDA_VISIBLE_DEVICES = "0"

$runs = @(
    @{id="B"; w=64; l0="1e-4"; bin=$true; seed=0},
    @{id="B"; w=64; l0="1e-4"; bin=$true; seed=1},
    @{id="B"; w=64; l0="1e-4"; bin=$true; seed=2}
)
$binw = @("--binarize-lam","0.5","--binarize-warmup","60")

function Run-Campaign($script, $init, $outdir) {
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null
    # freeze_encoder defaults to True in run_awa/run_cifar tree (annotated mode).
    $base = @("tree","--backbone","inception_v3","--init-from",$init,"--epochs","200",
        "--concept-lam","0.03","--max-parents","2","--rect-depth","3","--edge-l0-warmup","40",
        "--coefficients","--negation","--confusion-ovr","--confusion-hardness","1",
        "--early-stop-patience","40","--min-freeze-frac","0.4","--batch-size","32","--workers","4",
        "--tag","frozen")
    $i = 0
    foreach ($r in $runs) {
        $i++
        $log = "$outdir\$($r.id)_w$($r.w)_l0$($r.l0)_frozen_s$($r.seed).txt"
        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "DONE.*acc" -Quiet)) {
            Write-Host "===== [$i/$($runs.Count)] $script frozen seed $($r.seed) already DONE -> skip ====="
            continue
        }
        $extra = @("--rect-width","$($r.w)","--edge-l0","$($r.l0)","--seed","$($r.seed)")
        if ($r.bin) { $extra += $binw }
        Write-Host "===== [$i/$($runs.Count)] $script FROZEN W$($r.w) l0=$($r.l0) seed=$($r.seed) -> $log ====="
        $cmdargs = $base + $extra
        py -3 -u $script @cmdargs 2>&1 | Tee-Object $log
        Write-Host "----- done frozen $($r.id) seed $($r.seed) (exit $LASTEXITCODE) -----`n"
    }
}

Write-Host "########## AwA2 FROZEN TREE CAMPAIGN ##########"
Run-Campaign "run_awa.py" "saved\awa_joint_inception_v3_k85.pt" "results\campaign_awa_frozen"

Write-Host "########## CIFAR-100 FROZEN TREE CAMPAIGN ##########"
$cifarJoint = "saved\cifar_joint_inception_v3_k925.pt"
if (-not (Test-Path $cifarJoint)) {
    Write-Host "!! CIFAR joint pretrain missing ($cifarJoint) -- run run_cifar.py joint first" -ForegroundColor Yellow
} else {
    Run-Campaign "run_cifar.py" $cifarJoint "results\campaign_cifar_frozen"
}

Write-Host "AwA2 + CIFAR FROZEN CAMPAIGNS COMPLETE. Aggregate: py -3 aggregate_seeds.py results\campaign_awa_frozen (and _cifar_frozen)"
