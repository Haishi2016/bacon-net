import math

# weights: G_c, G_s, G_f  (hierarchy-corrected ~0.325/0.325/0.35)
w = (0.325, 0.325, 0.35)
heads = {
    "ontology": (0.65, 0.76, 0.81),
    "MLP":      (0.39, 0.00, 0.73),
    "linear":   (0.34, 0.00, 0.69),
}


def arith(v):                      # r=1 weighted arithmetic mean (neutral)
    return sum(wi * vi for wi, vi in zip(w, v))


def soft(v):                       # r=0.5 soft conjunction (non-annihilating)
    return (sum(wi * math.sqrt(vi) for wi, vi in zip(w, v))) ** 2


def hard_then_soft(v):             # current: hard_conj(Gc,Gs) then soft_conj(.,Gf)
    inner = math.sqrt(v[0] * v[1])
    return (0.65 * math.sqrt(inner) + 0.35 * math.sqrt(v[2])) ** 2


def headagn(v):                    # drop G_s, soft_conj(Gc, Gf) 0.65/0.35
    return (0.65 * math.sqrt(v[0]) + 0.35 * math.sqrt(v[2])) ** 2


print(f"{'head':<10}{'arith(r1)':>10}{'soft(r.5)':>10}{'hard->soft':>12}{'head-agn':>10}")
for h, v in heads.items():
    print(f"{h:<10}{arith(v):>10.2f}{soft(v):>10.2f}{hard_then_soft(v):>12.2f}{headagn(v):>10.2f}")
