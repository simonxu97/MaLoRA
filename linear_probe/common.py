"""Shared helpers for the CIFAR-100 linear-probing experiments (paper Sec. 4.1, App. H).
"""
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.datasets as dset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FEAT_DIR = ROOT / "features"
SPLIT_DIR = ROOT / "splits"
CKPT_DIR = ROOT / "checkpoints"
OUT_DIR = ROOT / "outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHS = ["vit", "resnet", "convnext", "clip"]


SUPERCLASS_TO_CLASSES = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}
SUPERCLASSES = list(SUPERCLASS_TO_CLASSES)
N_SUPER = len(SUPERCLASSES)
CLASS_TO_SUPER_IDX = {c: SUPERCLASSES.index(sc) for sc, subs in SUPERCLASS_TO_CLASSES.items() for c in subs}
N_PER_CLASS = 500  # CIFAR-100 training images per fine class (n in the theory)


def seed_everything(seed):
    """Seed Python and torch RNGs (LoRA initialization)."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cifar100(train, transform=None, data_dir=DATA_DIR):
    return dset.CIFAR100(str(data_dir), train=train, download=True, transform=transform)


# CIFAR-100 fine-label names in label-id order (the dataset's alphabetical order).
FINE_CLASSES = sorted(CLASS_TO_SUPER_IDX)


def fine_class_names():
    return FINE_CLASSES


# --------------------------------------------------------------------------- #
# Splits: 3 / 1 / 1 fine classes per superclass -> D_pre / D_1 / D_2           #
# --------------------------------------------------------------------------- #
SPLIT_FILES = {"pretrain": "pretrain.json", "tune1": "tune1.json", "tune2": "tune2.json"}


def build_splits(seed=0, out_dir=SPLIT_DIR):
    """Seeded shuffle of the five fine classes of every superclass."""
    random.seed(seed)
    torch.manual_seed(seed)
    pre, f1, f2 = {}, {}, {}
    for sc, subs in SUPERCLASS_TO_CLASSES.items():
        subs = subs.copy()
        random.shuffle(subs)
        pre[sc], f1[sc], f2[sc] = subs[:3], [subs[3]], [subs[4]]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in [("pretrain", pre), ("tune1", f1), ("tune2", f2)]:
        (out_dir / SPLIT_FILES[name]).write_text(json.dumps(obj, indent=2))
    return pre, f1, f2


def load_split(name):
    return json.loads((SPLIT_DIR / SPLIT_FILES[name]).read_text())


def load_features(arch, split="train", feat_dir=FEAT_DIR):
    blob = torch.load(Path(feat_dir) / f"{split}_{arch}.pt")
    return blob["features"], blob["labels"]


def split_indices(labels, split_name):
    """Dataset-order indices of the samples whose fine class is in `split_name`."""
    classes = fine_class_names()
    allowed = {c for v in load_split(split_name).values() for c in v}
    return torch.tensor([i for i, l in enumerate(labels.tolist()) if classes[l] in allowed], dtype=torch.long)


def superclass_ids(labels):
    classes = fine_class_names()
    return torch.tensor([CLASS_TO_SUPER_IDX[classes[l]] for l in labels.tolist()], dtype=torch.long)


def select(feats, labels, split_name, num_classes=N_SUPER):
    """Features and superclass targets of one split, keeping superclasses < num_classes."""
    idx = split_indices(labels, split_name)
    x, y = feats[idx], superclass_ids(labels[idx])
    if num_classes != N_SUPER:
        keep = y < num_classes
        x, y = x[keep], y[keep]
    return x, y


# --------------------------------------------------------------------------- #
# LoRA on a frozen linear classifier                                            #
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, W_pre, rank):
        super().__init__()
        self.W_pre = W_pre.to(DEVICE)
        self.B = nn.Parameter(torch.zeros(W_pre.shape[0], rank))
        self.A = nn.Parameter(torch.zeros(rank, W_pre.shape[1]))
        nn.init.normal_(self.A, std=1e-3)

    def forward(self, x):
        return x @ (self.W_pre + self.B @ self.A).T

    def delta(self):
        return self.B @ self.A


def penalized_loss(model, x, y, reg):
    """Mean cross-entropy + reg * (||B||_F^2 + ||A||_F^2).

    Same objective as the paper's Eq. (5), L + (lambda/2)(||B||_F^2 + ||A||_F^2), with reg = lambda / 2.
    """
    ce = nn.functional.cross_entropy(model(x), y)
    return ce + reg * (torch.norm(model.B, p="fro") ** 2 + torch.norm(model.A, p="fro") ** 2)


# --------------------------------------------------------------------------- #
# Margins and the closed-form quantities of Sec. 3                              #
# --------------------------------------------------------------------------- #
def normalized_min_margin(logits, y, W):
    """gamma(W; D) = min_i (correct logit - best wrong logit) / ||W||_F."""
    good = logits.gather(1, y.view(-1, 1)).squeeze(1)
    logits = logits.clone()
    logits.scatter_(1, y.view(-1, 1), float("-inf"))
    bad = logits.max(1).values
    return ((good - bad).min() / torch.norm(W, p="fro")).item()


def simplex_etf(m, device=DEVICE, dtype=torch.float32):
    I = torch.eye(m, device=device, dtype=dtype)
    ones = torch.ones(m, 1, device=device, dtype=dtype)
    M = torch.sqrt(torch.tensor(m / (m - 1), device=device, dtype=dtype)) * (I - (1.0 / m) * (ones @ ones.T))
    return M / torch.norm(M, "fro")


def etf_classifier(X, m, K=N_SUPER, rho=1.0):
    """Closed-form K-row classifier [M_m; 0] (x) 1_n^T X_ft^T (Prop. 2.1, zero-padded to K rows).

    X must hold m*n rows grouped by class. Only its normalized margin is used (gamma_ft^*).
    """
    n, rem = divmod(X.shape[0], m)
    if rem:
        raise ValueError("X must have m*n rows")
    device, dtype = X.device, X.dtype
    C = simplex_etf(m, device=device, dtype=dtype)
    if K > m:
        C = torch.cat([C, torch.zeros(K - m, m, device=device, dtype=dtype)], dim=0)
    W = torch.kron(C, torch.ones(1, n, device=device, dtype=dtype)) @ X
    scale = rho / torch.sqrt(torch.tensor(m * n * (m - 1), device=device, dtype=dtype))
    return scale * W / torch.norm(W, "fro")


def theory_optimal_lambda(rho_pre, gamma_pre, gamma_ft, K=N_SUPER, m=N_SUPER, n=N_PER_CLASS):
    """lambda* solving the stationarity condition d psi / d rho = 0 (proof of Theorem 3.1, App. D.1)
    at Theta* = rho_pre gamma_pre / gamma_ft.

    Returned in the paper's lambda units (objective with lambda/2), i.e. twice the `reg` of the runs.
    """
    rho_star = rho_pre * gamma_pre / gamma_ft * math.sqrt(m - 1)
    denom_term = math.exp(rho_star / ((m - 1) * math.sqrt(n)))
    cross_term = math.exp(rho_star / (m * (m - 1) * math.sqrt(n)))
    base = 1.0 / (m * math.sqrt(n))
    return base - base * (denom_term - 1.0) / ((K - m) * cross_term + (m - 1) + denom_term)
