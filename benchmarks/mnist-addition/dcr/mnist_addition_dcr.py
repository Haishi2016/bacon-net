#!/usr/bin/env python3
import argparse, math, random
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch_explain as te
from torch_explain.nn.concepts import ConceptReasoningLayer

# ----------------------------
# Data: MNIST-Addition pairs
# ----------------------------
class MnistAddition(Dataset):
    """
    Returns ((img1, img2), sum_label) where img1,img2 are separate 1x28x28 tensors
    and sum_label ∈ {0..18} is the sum of the two digits' labels.
    """
    def __init__(self, root, train: bool, size_pairs: int, seed: int = 42):
        super().__init__()
        rng = random.Random(seed)
        self.mnist = datasets.MNIST(
            root=root, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        n = len(self.mnist)
        self.pairs = []
        for _ in range(size_pairs):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            _, d1 = self.mnist[i1]
            _, d2 = self.mnist[i2]
            self.pairs.append((i1, i2, int(d1 + d2)))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, s = self.pairs[idx]
        x1, d1 = self.mnist[i1]
        x2, d2 = self.mnist[i2]
        return (x1, x2), s  # label is ONLY the sum

# ----------------------------
# Small CNN feature tower
# ----------------------------
class SmallCnn(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, out_features), nn.ReLU(),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ----------------------------
# Concept encoder per image:
#   - embeds: ConceptEmbedding(features->(10 concepts x emb_size))
#   - logits for 10-way concept distribution (for Gumbel-Softmax)
# ----------------------------
class ImageConceptHead(nn.Module):
    def __init__(self, feat_dim: int, n_concepts=10, emb_size=30):
        super().__init__()
        self.embed = te.nn.ConceptEmbedding(feat_dim, n_concepts, emb_size)
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features, tau: float, hard: bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
        # embeddings & (unused sigmoid preds from ConceptEmbedding)
        c_emb, c_pred_sigmoid = self.embed(features)  # c_emb: [B, 10, emb], c_pred_sigmoid: [B, 10] in (0,1)
        # Turn logits into a *categorical* truth-degree vector via Gumbel-Softmax to enforce near one-hot
        logits = self.logit_head(features)  # [B,10]
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # [B,10], sums to 1
        return c_emb, probs

# ----------------------------
# Full DCR model for MNIST-Addition
# ----------------------------
class DCRAddition(nn.Module):
    def __init__(self, emb_size=30):
        super().__init__()
        self.tower1 = SmallCnn(128)
        self.tower2 = SmallCnn(128)
        self.head1  = ImageConceptHead(128, 10, emb_size)
        self.head2  = ImageConceptHead(128, 10, emb_size)
        # DCR core: executes rules on concept probabilities and uses embeddings to assemble rules
        self.core   = ConceptReasoningLayer(emb_size, n_classes=19)
        # Auxiliary MLP that sees ONLY concept probabilities (to encourage crispness)
        self.aux    = nn.Sequential(
            nn.Linear(20, 30), nn.ReLU(),
            nn.Linear(30, 19), nn.Sigmoid()
        )

    def forward(self, x1, x2, tau: float):
        z1 = self.tower1(x1)  # [B,128]
        z2 = self.tower2(x2)  # [B,128]
        emb1, p1 = self.head1(z1, tau)
        emb2, p2 = self.head2(z2, tau)
        # concat 10+10 concepts
        c_emb  = torch.cat([emb1, emb2], dim=1)     # [B,20,emb]
        c_prob = torch.cat([p1, p2], dim=1)         # [B,20]
        # DCR prediction (multi-class with one-hot target)
        y_pred = self.core(c_emb, c_prob)           # [B,19] in [0,1]
        # Auxiliary task head on concept probabilities only
        y_aux  = self.aux(c_prob)                   # [B,19] in [0,1]
        return y_pred, y_aux, c_emb, c_prob

# ----------------------------
# Utilities
# ----------------------------
def one_hot(y, n_classes):
    return F.one_hot(torch.as_tensor(y), num_classes=n_classes).float()

def group_entropy_loss(c_prob):
    """Entropy penalty per image (two groups of 10) to push near one-hot."""
    eps = 1e-8
    g1 = c_prob[:, :10]; g2 = c_prob[:, 10:]
    ent = -(g1 * (g1+eps).log()).sum(dim=1) -(g2 * (g2+eps).log()).sum(dim=1)
    # Normalize by log(10) so weight is scale-invariant
    return (ent / math.log(10)).mean()

def accuracy_from_probs(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true).float().mean().item()

# ----------------------------
# Training
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", type=str)
    ap.add_argument("--train-pairs", default=30000, type=int)
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=30, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--emb", default=30, type=int, help="concept embedding size {10,20,30,50}")
    ap.add_argument("--tau", default=1.25, type=float, help="Gumbel-Softmax temperature")
    ap.add_argument("--aux", default=0.5, type=float, help="auxiliary loss weight")
    ap.add_argument("--entropy", default=0.01, type=float, help="entropy penalty weight")
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistAddition(args.data, train=True,  size_pairs=args.train_pairs, seed=args.seed)
    test_ds  = MnistAddition(args.data, train=False, size_pairs=args.test_pairs,  seed=999)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DCRAddition(emb_size=args.emb).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr)

    //bce = nn.BCELoss()
    ce = nn.CrossEntropyLoss(weight=class_weights.to(device))

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y_1hot = one_hot(y, 19).to(device)

            y_pred, y_aux, c_emb, c_prob = model(x1, x2, tau=args.tau)

            # Main task (sum) with one-hot BCE on DCR output
            loss_main = bce(y_pred, y_1hot)
            # Auxiliary task on concept probabilities only
            loss_aux  = bce(y_aux,  y_1hot)
            # Encourage each 10-way group to be one-hot (crispness)
            loss_ent  = group_entropy_loss(c_prob)

            loss = loss_main + args.aux*loss_aux + args.entropy*loss_ent

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * x1.size(0)

        # Eval
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_pred, _, c_emb, c_prob = model(x1, x2, tau=args.tau)
                correct += (y_pred.argmax(1) == y).sum().item()
                tot += y.size(0)
            acc = correct / tot

        print(f"Epoch {epoch:02d} | train_loss={running/len(train_ds):.4f} | test_acc={acc*100:.2f}%")

    # --------- Extract global rules (paper-style) ----------
    # Use a slice of test set for readability
    (x1t, x2t), yt = next(iter(test_loader))
    x1t, x2t = x1t.to(device), x2t.to(device)
    with torch.no_grad():
        _, _, c_emb_t, c_prob_t = model(x1t, x2t, tau=args.tau)
    # Ask Deep CoRe layer for global rules over the observed concept space
    global_rules = model.core.explain(c_emb_t, c_prob_t, mode='global')
    print("\n=== Learned global rules (sample) ===")
    # global_rules is a list of human-readable formulas per class;
    # print for sums 16,17,18 if present (paper Appendix H style)
    for k in [16,17,18]:
        if k < len(global_rules) and global_rules[k]:
            print(f"y_{k} <- {global_rules[k]}")
        else:
            print(f"y_{k} <- (no rule extracted on this slice)")
    print("====================================\n")

if __name__ == "__main__":
    main()
