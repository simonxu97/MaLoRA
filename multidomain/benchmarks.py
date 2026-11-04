"""The six multi-domain benchmarks of Table 1 (paper Sec. 4.2, App. H).

Each loader returns a HuggingFace Dataset with a "domain" and a "label" column; the
order of its rows matters, because the per-domain splits are seeded index lists.

  * PACS                     flwrlabs/pacs (HF)
  * VLCS, TerraIncognita     DomainBed image folders (see README), read in the order
                             listed in manifests/<name>.txt (the paper runs' order)
  * MNLI                     multi_nli train split, genres as domains
  * Amazon Reviews           amazon_polarity reviews assigned to four pseudo-domains
                             by keyword matching
  * CivilComments            google/civil_comments, domain = dominant toxicity attribute
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from PIL import Image, ImageFile
from transformers import AutoImageProcessor, AutoTokenizer

MANIFEST_DIR = Path(__file__).resolve().parent / "manifests"
DATA_ROOT = Path(os.environ.get("MALORA_DATA", Path(__file__).resolve().parent / "data"))

# Dataset revisions used for the paper runs.
HF_REVISIONS = {
    "flwrlabs/pacs": "394113073258ead631f617d2e13bb377c0715c4b",
    "multi_nli": "da70db2af9d09693783c3320c4249840212ee221",
    "amazon_polarity": "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289",
    "google/civil_comments": "f2970eb3a55777454c94069077cc8d9b5866312d",
}


def _hf(name, **kw):
    return load_dataset(name, revision=HF_REVISIONS[name], **kw)


@dataclass
class Batch:
    pixel_values: torch.Tensor
    labels: torch.Tensor


@dataclass
class TextBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class MultiDomainDataset:
    domains: List[str] = []
    num_labels: int = 0
    is_text = False
    max_length = 128

    def load_data(self) -> HFDataset:
        raise NotImplementedError

    def get_processor(self, base_model):
        return AutoTokenizer.from_pretrained(base_model) if self.is_text else AutoImageProcessor.from_pretrained(base_model)

    def get_collate_fn(self, processor) -> Callable:
        raise NotImplementedError

    def build_domain_to_indices(self, ds) -> Dict[str, List[int]]:
        out = {d: [] for d in self.domains}
        column = ds["domain"]
        feature = ds.features.get("domain")
        names = getattr(feature, "names", None)
        for i, d in enumerate(column):
            d = names[d] if names is not None else d
            if d in out:
                out[d].append(i)
        return out


# --------------------------------------------------------------------------- #
# Vision                                                                        #
# --------------------------------------------------------------------------- #
class PACS(MultiDomainDataset):
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    num_labels = 7

    def load_data(self):
        return _hf("flwrlabs/pacs")["train"]

    def get_collate_fn(self, processor):
        def collate(examples):
            images = [ex["image"] for ex in examples]
            labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
            return Batch(processor(images=images, return_tensors="pt")["pixel_values"], labels)
        return collate


class ImageFolderDataset(MultiDomainDataset):
    """DomainBed-style <root>/<domain>/<class>/<image> folders, loaded lazily from paths."""
    folder = ""
    class_names: List[str] = []

    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_ROOT / self.folder

    def _relative_paths(self):
        manifest = MANIFEST_DIR / f"{self.folder}.txt"
        if manifest.exists():
            return manifest.read_text().split("\n")[:-1]
        print(f"  [warn] {manifest} not found: using sorted file order")
        rel = []
        for domain in self.domains:
            for c in self.class_names:
                p = self.data_dir / domain / c
                if p.is_dir():
                    rel += [f"{domain}/{c}/{f}" for f in sorted(os.listdir(p))
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        return rel

    def load_data(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(f"{self.data_dir} not found; see README for the DomainBed download command")
        class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        data = {"image_path": [], "label": [], "domain": []}
        for rel in self._relative_paths():
            domain, cls, _ = rel.split("/")
            data["image_path"].append(str(self.data_dir / rel))
            data["label"].append(class_to_idx[cls])
            data["domain"].append(domain)
        print(f"Loaded {len(data['image_path'])} images from {self.data_dir}")
        return HFDataset.from_dict(data)

    def get_collate_fn(self, processor):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        def collate(examples):
            images = []
            for ex in examples:
                try:
                    images.append(Image.open(ex["image_path"]).convert("RGB"))
                except OSError as e:
                    raise OSError(f"cannot load image {ex['image_path']}") from e
            labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
            return Batch(processor(images=images, return_tensors="pt")["pixel_values"], labels)
        return collate

    def write_manifest(self):
        """Record the os.listdir order of the local copy (how the paper runs read the folders)."""
        rel = []
        for domain in self.domains:
            for c in self.class_names:
                p = self.data_dir / domain / c
                if p.is_dir():
                    rel += [f"{domain}/{c}/{f}" for f in os.listdir(p)
                            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        MANIFEST_DIR.mkdir(exist_ok=True)
        (MANIFEST_DIR / f"{self.folder}.txt").write_text("".join(r + "\n" for r in rel))
        return len(rel)


class VLCS(ImageFolderDataset):
    folder = "VLCS"
    domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]
    class_names = ["bird", "car", "chair", "dog", "person"]
    num_labels = 5


class TerraIncognita(ImageFolderDataset):
    folder = "terra_incognita"
    domains = ["location_100", "location_38", "location_43", "location_46"]
    class_names = ["bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit", "raccoon", "squirrel"]
    num_labels = 10


# --------------------------------------------------------------------------- #
# Language                                                                      #
# --------------------------------------------------------------------------- #
class TextDataset(MultiDomainDataset):
    is_text = True
    max_length = 256

    def get_collate_fn(self, processor):
        def collate(examples):
            enc = processor([ex["text"] for ex in examples], padding=True, truncation=True,
                            max_length=self.max_length, return_tensors="pt")
            labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
            return TextBatch(enc["input_ids"], enc["attention_mask"], labels)
        return collate


class MNLI(TextDataset):
    """All five training genres of MultiNLI; premise/hypothesis pairs, 3 classes."""
    domains = ["fiction", "government", "slate", "telephone", "travel"]
    num_labels = 3
    max_length = 128

    def load_data(self):
        data = {"premise": [], "hypothesis": [], "label": [], "domain": []}
        for item in _hf("multi_nli")["train"]:
            if item["genre"] in self.domains:
                data["premise"].append(item["premise"])
                data["hypothesis"].append(item["hypothesis"])
                data["label"].append(item["label"])
                data["domain"].append(item["genre"])
        return HFDataset.from_dict(data)

    def get_collate_fn(self, processor):
        def collate(examples):
            enc = processor([ex["premise"] for ex in examples], [ex["hypothesis"] for ex in examples],
                            padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            labels = torch.tensor([ex["label"] for ex in examples], dtype=torch.long)
            return TextBatch(enc["input_ids"], enc["attention_mask"], labels)
        return collate


class AmazonReviews(TextDataset):
    """Binary sentiment with four keyword-defined pseudo-domains.

    Reviews of amazon_polarity (train split, original order) are assigned to the
    first domain whose keyword list matches the lower-cased title+content, until
    each domain holds 2,500 reviews.
    """
    domains = ["books", "electronics", "movies", "kitchen"]
    num_labels = 2
    keywords = {
        "books": ["book", "read", "story", "author", "novel", "page"],
        "electronics": ["electronic", "battery", "power", "device", "tech"],
        "movies": ["movie", "film", "watch", "actor", "director", "scene"],
        "kitchen": ["kitchen", "cook", "food", "recipe", "dish", "meal"],
    }

    def __init__(self, samples_per_domain=2000):
        self.target = samples_per_domain + 500

    def load_data(self):
        counts = {d: 0 for d in self.domains}
        data = {"text": [], "label": [], "domain": []}
        for item in _hf("amazon_polarity", split="train"):
            if all(c >= self.target for c in counts.values()):
                break
            text = (item.get("title", "") + " " + item.get("content", "")).lower()
            for domain, words in self.keywords.items():
                if counts[domain] < self.target and any(w in text for w in words):
                    data["text"].append(item.get("content", item.get("title", "")))
                    data["label"].append(item["label"])
                    data["domain"].append(domain)
                    counts[domain] += 1
                    break
        print(f"  Loaded samples per domain: {counts}")
        return HFDataset.from_dict(data)


class CivilComments(TextDataset):
    """Binary toxicity (toxicity > 0.5); a comment's domain is its highest-scoring
    attribute among the five below, kept only if that score is >= 0.3."""
    domains = ["obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
    num_labels = 2

    def __init__(self, samples_per_domain=2000, attribute_threshold=0.3):
        self.target = samples_per_domain + 400
        self.threshold = attribute_threshold

    def load_data(self):
        counts = {d: 0 for d in self.domains}
        data = {"text": [], "label": [], "domain": []}
        for item in _hf("google/civil_comments", split="train"):
            if all(c >= self.target for c in counts.values()):
                break
            attrs = {d: float(item[d]) for d in self.domains if d in item}
            top, score = max(attrs.items(), key=lambda kv: kv[1])
            if score < self.threshold or counts[top] >= self.target:
                continue
            data["text"].append(item["text"])
            data["label"].append(1 if float(item["toxicity"]) > 0.5 else 0)
            data["domain"].append(top)
            counts[top] += 1
        print(f"  Loaded samples per domain: {counts}")
        return HFDataset.from_dict(data)


DATASETS = {
    "pacs": PACS,
    "vlcs": VLCS,
    "terra_incognita": TerraIncognita,
    "mnli": MNLI,
    "amazon_reviews": AmazonReviews,
    "civil_comments": CivilComments,
}


def get_dataset(name) -> MultiDomainDataset:
    return DATASETS[name]()
