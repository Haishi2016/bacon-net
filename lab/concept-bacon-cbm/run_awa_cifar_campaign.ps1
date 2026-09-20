# AwA2 + CIFAR-100 3-seed reproduction campaigns.
# Canonical B = recttree W64 + L0 + hardness-anneal (binarize), 3 seeds per dataset
# (no width sweep, no ensemble). Idempotent: skips runs whose log already recorded a
# DONE accuracy. save paths auto-encode width/l0/_bin/_s{seed} (run_awa|run_cifar
# tree), so --seed alone distinguishes checkpoints.
#
# Run:  & c:\School\bacon-net\lab\concept-bacon-cbm\run_awa_cifar_campaign.ps1
# Aggregate:  py -3 aggregate_seeds.py results\campaign_awa   (and _cifar)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONWARNINGS = "ignore"

# w64 only, 3 seeds (config B = recttree W64 + L0 + hardness-anneal). No width
# sweep and no ensemble for AwA2/CIFAR: report mean +/- std over the 3 seeds.
$runs = @(
    @{id="B"; w=64; l0="1e-4"; bin=$true; seed=0},
    @{id="B"; w=64; l0="1e-4"; bin=$true; seed=1},
    @{id="B"; w=64; l0="1e-4"; bin=$true; seed=2}
)
$binw = @("--binarize-lam","0.5","--binarize-warmup","60")

function Run-Campaign($script, $init, $outdir) {
    New-Item -ItemType Directory -Force -Path $outdir | Out-Null
    $base = @("tree","--backbone","inception_v3","--init-from",$init,"--epochs","200",
        "--concept-lam","0.03","--max-parents","2","--rect-depth","3","--edge-l0-warmup","40",
        "--coefficients","--negation","--confusion-ovr","--confusion-hardness","1",
        "--early-stop-patience","40","--min-freeze-frac","0.4","--batch-size","32","--workers","4")
    $i = 0
    foreach ($r in $runs) {
        $i++
        $tag = if ($r.bin) { "binw_s$($r.seed)" } else { "s$($r.seed)" }
        $log = "$outdir\$($r.id)_w$($r.w)_l0$($r.l0)_$tag.txt"
        if ((Test-Path $log) -and (Select-String -Path $log -Pattern "DONE.*acc" -Quiet)) {
            Write-Host "===== [$i/$($runs.Count)] $script $($r.id) seed $($r.seed) already DONE -> skip ====="
            continue
        }
        $extra = @("--rect-width","$($r.w)","--edge-l0","$($r.l0)","--seed","$($r.seed)")
        if ($r.bin) { $extra += $binw }
        Write-Host "===== [$i/$($runs.Count)] $script $($r.id) W$($r.w) l0=$($r.l0) bin=$($r.bin) seed=$($r.seed) -> $log ====="
        $cmdargs = $base + $extra
        py -3 -u $script @cmdargs 2>&1 | Tee-Object $log
        Write-Host "----- done $($r.id) seed $($r.seed) (exit $LASTEXITCODE) -----`n"
    }
}

# ---- AwA2 (joint pretrain already done) ----
Write-Host "########## AwA2 TREE CAMPAIGN ##########"
Run-Campaign "run_awa.py" "saved\awa_joint_inception_v3_k85.pt" "results\campaign_awa"

# ---- CIFAR-100: joint pretrain first (if missing), then trees ----
New-Item -ItemType Directory -Force -Path "results\campaign_cifar" | Out-Null
$cifarJoint = "saved\cifar_joint_inception_v3_k925.pt"
if (-not (Test-Path $cifarJoint)) {
    Write-Host "########## CIFAR-100 JOINT PRETRAIN ##########"
    py -3 -u run_cifar.py joint --epochs 60 --batch-size 64 --workers 4 2>&1 | Tee-Object "results\campaign_cifar\_joint.txt"
}
Write-Host "########## CIFAR-100 TREE CAMPAIGN ##########"
Run-Campaign "run_cifar.py" $cifarJoint "results\campaign_cifar"

Write-Host "AwA2 + CIFAR CAMPAIGNS COMPLETE. Aggregate: py -3 aggregate_seeds.py results\campaign_awa (and _cifar)"
