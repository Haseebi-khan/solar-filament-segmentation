"""Dataset loading: COCO annotations, polygon rasterisation, image caching, CV folds.

Terminology
-----------
image stem      : file name without extension, e.g. ``20160213153054Bh``.
annotation set  : one annotator's labelling of one image; its COCO image id is
                  ``<annotator>-<stem>``, e.g. ``010401-20160213153054Bh``.
                  The organisers score every annotation set separately against
                  all predictions for the image, so a stem with three
                  annotators is scored three times.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from pycocotools import mask as mask_util

H = W = 2048

DATA_ROOT = os.environ.get("FILSEG_DATA", "/root/kaggle/data/MAGFiLO_1.0_Kaggle_2026")
WORK_ROOT = os.environ.get("FILSEG_WORK", "/root/kaggle/work")
TRAIN_JSON = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train", "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "test_images")
CACHE_DIR = os.path.join(WORK_ROOT, "cache")


@dataclass
class AnnotationSet:
    set_id: str                 # "<annotator>-<stem>"
    stem: str
    polygons: list = field(default_factory=list)   # list of flat [x0, y0, x1, y1, ...]

    def rles(self) -> list[dict]:
        """Rasterise each polygon exactly like pycocotools' annToMask."""
        out = []
        for poly in self.polygons:
            rle = mask_util.merge(mask_util.frPyObjects([poly], H, W))
            out.append(rle)
        return out


def load_train_annotations(path: str = TRAIN_JSON) -> tuple[list[str], dict[str, list[AnnotationSet]]]:
    """Returns (sorted unique stems, {stem: [AnnotationSet, ...]})."""
    with open(path) as f:
        coco = json.load(f)
    sets: dict[str, AnnotationSet] = {}
    for im in coco["images"]:
        stem = os.path.splitext(im["file_name"])[0]
        sets[im["id"]] = AnnotationSet(set_id=im["id"], stem=stem)
    for ann in coco["annotations"]:
        assert len(ann["segmentation"]) == 1
        sets[ann["image_id"]].polygons.append(ann["segmentation"][0])
    by_stem: dict[str, list[AnnotationSet]] = defaultdict(list)
    for s in sets.values():
        by_stem[s.stem].append(s)
    for v in by_stem.values():
        v.sort(key=lambda s: s.set_id)
    return sorted(by_stem), dict(by_stem)


def list_test_stems(img_dir: str = TEST_IMG_DIR) -> list[str]:
    return sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.lower().endswith((".jpeg", ".jpg")))


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    assert img.shape == (H, W), (path, img.shape)
    return img


def cache_images(stems: list[str], img_dir: str, name: str) -> np.ndarray:
    """Decodes all JPEGs once into a uint8 memmap of shape (N, 2048, 2048)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}_images.npy")
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] == len(stems):
            return arr
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(stems), H, W))
    for i, s in enumerate(stems):
        arr[i] = read_image(os.path.join(img_dir, s + ".jpeg"))
    arr.flush()
    return np.load(path, mmap_mode="r")


def assign_folds(stems: list[str], n_folds: int = 5, seed: int = 42) -> dict[str, int]:
    """Image-level folds: all annotation sets of one image share a fold.

    The test images are a random sample of the same observation pool (median gap
    to the nearest training image ~37 h, same as between training images), so a
    random image-level split mirrors the train/test relationship.
    """
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(stems))
    return {stems[i]: int(k % n_folds) for k, i in enumerate(order)}


def rle_to_string(rle: dict) -> str:
    counts = rle["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else counts


def encode_mask(mask: np.ndarray) -> dict:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
