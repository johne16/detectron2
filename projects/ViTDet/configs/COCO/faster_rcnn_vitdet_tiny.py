from functools import partial
import os
import yaml

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.labeled_fraction import register_coco_instances_labeled_fraction

from .mask_rcnn_vitdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vit_lr_decay_rate,
)

with open(os.path.join(os.path.dirname(__file__), "faster_rcnn_vitdet_tiny_runs.yaml")) as f:
    _run = yaml.safe_load(f)[os.environ["RUN"]]


_coco_dir = "../datasets/coco"

register_coco_instances_labeled_fraction(
    "coco_2017_train_subset",
    {},
    os.path.join(_coco_dir, "annotations/instances_train2017.json"),
    os.path.join(_coco_dir, "train/data"),
    _run["labeled_fraction"],
)
register_coco_instances(
    "coco_2017_val_subset",
    {},
    os.path.join(_coco_dir, "annotations/instances_val2017.json"),
    os.path.join(_coco_dir, "validation/data"),
)

dataloader.train.dataset.names = "coco_2017_train_subset"
dataloader.test.dataset.names = "coco_2017_val_subset"

train.init_checkpoint = _run["init_checkpoint"]
train.seed = _run["seed"]
train.output_dir = os.path.join(train.output_dir, os.environ["RUN"])
train.eval_period = 500
train.checkpointer.period = 500

# Schedule
# max_iter iters = num_epochs ep * num_images images/ep / total_batch_size images/iter
_num_epochs = 100
_num_images = len(DatasetCatalog.get("coco_2017_train_subset"))
train.max_iter = round(_num_epochs * _num_images / dataloader.train.total_batch_size)

lr_multiplier.scheduler.milestones = [round(train.max_iter * 8 / 9), round(train.max_iter * 26 / 27)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 500 / train.max_iter

model.backbone.net.embed_dim = 192
model.backbone.net.depth = 12
model.backbone.net.num_heads = 3
model.backbone.net.drop_path_rate = 0.1

model.roi_heads.num_classes = 9

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, lr_decay_rate=_run["lr_decay_rate"], num_layers=12)

# remove the mask branch arguments so the ROI heads build box prediction only (Faster R-CNN)
[model.roi_heads.pop(x) for x in ["mask_in_features", "mask_pooler", "mask_head"]]