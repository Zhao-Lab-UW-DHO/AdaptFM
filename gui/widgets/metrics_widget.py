from abc import ABC, abstractmethod
import tifffile as tiff
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QFileDialog

class Metric(ABC):
    name: str = "BaseMetric"

    @abstractmethod
    def compute(self,gt_path,pred_path):
        """Compute the metric between ground truth and prediction"""
        pass


class MetricRegistry:
    _metrics ={}

    @classmethod
    def register(cls, metric_cls):
        cls._metrics[metric_cls.name] = metric_cls
        return metric_cls
    
    @classmethod
    def get_metrics(cls):
        return list(cls._metrics.keys())

    @classmethod
    def create(cls, name, *args, **kwargs):
        return cls._metrics[name](*args, **kwargs) 



@MetricRegistry.register
class DiceScore(Metric):
    name = "Dice Score"

    def compute(self, gt_dir, models_dirs):
        """
        gt_dir: path to ground truth folder
        models_dirs: list of paths to model prediction folders
        """
        gt_dir = Path(gt_dir)
        results = {}

        # get all ground truth files
        gt_files = sorted(gt_dir.glob("*"))  # assumes all images in folder

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*"))

            per_image_dice = []

            # match files by order (or implement matching by name if needed)
            for gt_file, pred_file in zip(gt_files, model_files):
                gt = tiff.imread(gt_file) > 0        # binarize
                pred = tiff.imread(pred_file) > 0    # binarize
                intersection = (gt & pred).sum()
                union = gt.sum() + pred.sum()
                dice = 2 * intersection / union if union > 0 else 1.0
                per_image_dice.append(dice)

            # store results for this model
            results[str(model_dir)] = per_image_dice

        return results
    

from scipy.optimize import linear_sum_assignment
from skimage.io import imread
from skimage.measure import label, regionprops
import numpy as np
from pathlib import Path

# ------------------------------
# Mean Object F1 with greedy matching
# ------------------------------
@MetricRegistry.register
class MeanObjectF1(Metric):
    name = "Mean Object F1"

    def compute(self, gt_dir, models_dirs):
        """
        Compute object-level F1 score per image using greedy matching
        """
        gt_dir = Path(gt_dir)
        gt_files = sorted(gt_dir.glob("*"))
        results = {}

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*"))

            per_image_f1 = []

            for gt_file, pred_file in zip(gt_files, model_files):
                gt_mask = label(imread(gt_file))        # labeled objects
                pred_mask = label(imread(pred_file))

                f1 = self._greedy_object_f1(gt_mask, pred_mask)
                per_image_f1.append(f1)

            results[str(model_dir)] = per_image_f1

        return results

    @staticmethod
    def _greedy_object_f1(gt_mask, pred_mask):
        gt_labels = np.unique(gt_mask)[1:]  # ignore background
        pred_labels = np.unique(pred_mask)[1:]

        if len(gt_labels) == 0 and len(pred_labels) == 0:
            return 1.0
        if len(gt_labels) == 0 or len(pred_labels) == 0:
            return 0.0

        matched = 0
        pred_used = set()
        for gt_id in gt_labels:
            gt_obj = gt_mask == gt_id
            best_iou = 0
            best_pred = None
            for pred_id in pred_labels:
                if pred_id in pred_used:
                    continue
                pred_obj = pred_mask == pred_id
                intersection = np.logical_and(gt_obj, pred_obj).sum()
                union = np.logical_or(gt_obj, pred_obj).sum()
                iou = intersection / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_id
            if best_iou >= 0.5:  # IoU threshold
                matched += 1
                pred_used.add(best_pred)

        precision = matched / len(pred_labels) if pred_labels.size > 0 else 0
        recall = matched / len(gt_labels) if gt_labels.size > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1


# ------------------------------
# Panoptic F1
# ------------------------------
@MetricRegistry.register
class PanopticF1(Metric):
    name = "Panoptic F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir = Path(gt_dir)
        gt_files = sorted(gt_dir.glob("*"))
        results = {}

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*"))

            per_image_pf1 = []

            for gt_file, pred_file in zip(gt_files, model_files):
                gt_mask = label(imread(gt_file))
                pred_mask = label(imread(pred_file))
                pf1 = self._panoptic_f1(gt_mask, pred_mask)
                per_image_pf1.append(pf1)

            results[str(model_dir)] = per_image_pf1

        return results

    @staticmethod
    def _panoptic_f1(gt_mask, pred_mask):
        """
        Compute panoptic F1 score using greedy matching (IoU>0.5)
        """
        gt_labels = np.unique(gt_mask)[1:]
        pred_labels = np.unique(pred_mask)[1:]

        if len(gt_labels) == 0 and len(pred_labels) == 0:
            return 1.0
        if len(gt_labels) == 0 or len(pred_labels) == 0:
            return 0.0

        tp = 0
        fp = len(pred_labels)
        fn = len(gt_labels)

        pred_used = set()
        for gt_id in gt_labels:
            gt_obj = gt_mask == gt_id
            best_iou = 0
            best_pred = None
            for pred_id in pred_labels:
                if pred_id in pred_used:
                    continue
                pred_obj = pred_mask == pred_id
                intersection = np.logical_and(gt_obj, pred_obj).sum()
                union = np.logical_or(gt_obj, pred_obj).sum()
                iou = intersection / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_id
            if best_iou >= 0.5:
                tp += 1
                pred_used.add(best_pred)
                fp -= 1
                fn -= 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1


# ------------------------------
# Boundary F1
# ------------------------------
from skimage.segmentation import find_boundaries

@MetricRegistry.register
class BoundaryF1(Metric):
    name = "Boundary F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir = Path(gt_dir)
        gt_files = sorted(gt_dir.glob("*"))
        results = {}

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*"))

            per_image_bf1 = []

            for gt_file, pred_file in zip(gt_files, model_files):
                gt_mask = imread(gt_file) > 0
                pred_mask = imread(pred_file) > 0
                bf1 = self._boundary_f1(gt_mask, pred_mask)
                per_image_bf1.append(bf1)

            results[str(model_dir)] = per_image_bf1

        return results

    @staticmethod
    def _boundary_f1(gt_mask, pred_mask):
        """
        Compute boundary F1 (precision/recall of boundary pixels)
        """
        gt_bound = find_boundaries(gt_mask, mode="inner")
        pred_bound = find_boundaries(pred_mask, mode="inner")

        tp = np.logical_and(gt_bound, pred_bound).sum()
        fp = np.logical_and(pred_bound, ~gt_bound).sum()
        fn = np.logical_and(gt_bound, ~pred_bound).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return f1


