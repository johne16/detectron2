import random

from .. import DatasetCatalog, MetadataCatalog
from .coco import load_coco_json


def load_coco_json_labeled_fraction(json_file, image_root, dataset_name, labeled_fraction):
    """
    Load a json file with COCO's instances annotation format and keep a fixed
    random fraction of its images. A smaller fraction is a subset of a larger one.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
        labeled_fraction (float): the fraction of images to keep, in (0, 1].

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format.
    """
    dataset_dicts = load_coco_json(json_file, image_root, dataset_name)
    image_ids = sorted(d["image_id"] for d in dataset_dicts)
    random.Random(0).shuffle(image_ids)
    kept_image_ids = set(image_ids[: round(labeled_fraction * len(image_ids))])
    return [d for d in dataset_dicts if d["image_id"] in kept_image_ids]


def register_coco_instances_labeled_fraction(name, metadata, json_file, image_root, labeled_fraction):
    """
    Register a dataset in COCO's json annotation format that keeps a fixed
    random fraction of its images.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
        labeled_fraction (float): the fraction of images to keep, in (0, 1].
    """
    DatasetCatalog.register(
        name, lambda: load_coco_json_labeled_fraction(json_file, image_root, name, labeled_fraction)
    )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )
