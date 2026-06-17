import json, sys
H = json.load(open(r"C:\School\bacon-net\runs\cub_bacon\history.json"))
print(f"epochs logged: {len(H)}")
# columns
def row(r):
    return (r["epoch"], r["train"]["loss"], r["train"]["task_acc"],
            r["test"]["task_acc"], r["test"]["concept_acc"], r["lr"])

# find key milestones
best = max(H, key=lambda r: r["test"]["task_acc"])
print(f"BEST test task acc: {best['test']['task_acc']:.4f} @ epoch {best['epoch']} "
      f"(train {best['train']['task_acc']:.4f})")

# gap over time (every 10 epochs from 60)
print("\nepoch  trLoss  trTask  teTask  gap   teConcept  lr")
for r in H:
    if r["epoch"] % 10 == 0 or r["epoch"] in (1,):
        e,l,tr,te,tc,lr = row(r)
        print(f"{e:5d}  {l:6.3f}  {tr:6.3f}  {te:6.3f}  {tr-te:+.3f}  {tc:7.3f}  {lr:.5f}")

# when did test peak vs train keep rising? overfit onset
import numpy as np
te = np.array([r["test"]["task_acc"] for r in H])
tr = np.array([r["train"]["task_acc"] for r in H])
ep = np.array([r["epoch"] for r in H])
# test best epoch and plateau
peak = ep[te.argmax()]
# last epoch where test improved by >0.005 over running max
running = np.maximum.accumulate(te)
improved = ep[np.where(np.diff(running) > 0.002)[0] + 1]
print(f"\ntest peak epoch={peak}, test acc still meaningfully improving until epoch ~{improved[-1] if len(improved) else 'n/a'}")
print(f"final 40-epoch test mean (ep141-180): {te[ep>=141].mean():.4f}  train mean: {tr[ep>=141].mean():.4f}")
print(f"final gap (train-test) avg ep141-180: {(tr[ep>=141]-te[ep>=141]).mean():+.4f}")

# concept acc trend (does it degrade as task head pulls?)
tc = np.array([r["test"]["concept_acc"] for r in H])
print(f"concept acc: max {tc.max():.4f} @ ep{ep[tc.argmax()]}, final {tc[-1]:.4f}")
