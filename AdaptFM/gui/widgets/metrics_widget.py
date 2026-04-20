from abc import ABC, abstractmethod
import tifffile as tiff
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QFileDialog
import matplotlib.pyplot as plt
import os
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


def run_parallel(file_pairs, worker_fn, n_processes=None):
    """
    Run worker_fn over file pairs in parallel.

    file_pairs: list[(gt_path, pred_path)]
    worker_fn: function(gt_path, pred_path) -> scalar
    n_processes: int or None
    """

    if n_processes is None:
        n_processes = max(cpu_count() - 1, 1)

    if n_processes == 1:
        return [worker_fn(p) for p in file_pairs]

    with Pool(processes=n_processes) as pool:
        results = pool.map(worker_fn, file_pairs)

    return results

def plot_dice_boxplot(results,gt_dir,name):
    """
    results: dict {model_name: [dice_scores]}
    """

    model_names = [Path(k).name for k in results.keys()]
    dice_values = list(results.values())

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        dice_values,
        tick_labels=model_names,
        showmeans=True
    )

    plt.ylabel(f"{name}")
    plt.xlabel("Model")
    plt.title(f"{name} Distribution per Model")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(os.path.join(gt_dir,f"{name}_boxplot.png"), dpi=300)



def object_iou_matrix_fast(gt_mask, pred_mask):
    """
    Fast computation of IoU matrix between all GT and predicted objects.
    """
    gt_labels = label(gt_mask)
    pred_labels = label(pred_mask)

    num_gt = gt_labels.max()
    num_pred = pred_labels.max()

    # Flatten masks for histogram counting
    gt_flat = gt_labels.ravel()
    pred_flat = pred_labels.ravel()

    # Combine GT and predicted labels into a single index
    # Shift pred labels to make unique pair indices
    combined = gt_flat * (num_pred + 1) + pred_flat
    counts = np.bincount(combined)

    # Map back to 2D IoU matrix
    iou_matrix = np.zeros((num_gt, num_pred), dtype=float)
    
    # Compute intersection for each pair
    for idx, count in enumerate(counts):
        if count == 0:
            continue
        gt_id = idx // (num_pred + 1)
        pred_id = idx % (num_pred + 1)
        if gt_id == 0 or pred_id == 0:
            continue  # background
        iou_matrix[gt_id-1, pred_id-1] = count

    # Compute union
    gt_area = np.bincount(gt_flat, minlength=num_gt+1)[1:]  # skip background
    pred_area = np.bincount(pred_flat, minlength=num_pred+1)[1:]
    union_matrix = gt_area[:, None] + pred_area[None, :] - iou_matrix
    iou_matrix = iou_matrix / np.maximum(union_matrix, 1e-12)

    return iou_matrix, gt_labels, pred_labels

    
class Metric(ABC):
    name: str = "BaseMetric"

    def __init__(self, n_processes=None):
        self.n_processes = n_processes

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

        gt_dir = Path(gt_dir)
        gt_files = sorted(gt_dir.glob("*tiff"))

        results = {}

        for model_dir in models_dirs:

            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*tiff"))

            file_pairs = list(
                zip(gt_files, model_files)
            )

            per_image_dice = run_parallel(
                file_pairs,
                self._dice_worker,
                self.n_processes
            )

            results[str(model_dir)] = per_image_dice

        plot_dice_boxplot(
            results,
            gt_dir,
            self.name
        )

        return results

    @staticmethod  
    def _dice_worker(pair):

        gt_file, pred_file = pair

        gt = tiff.imread(gt_file) > 0
        pred = tiff.imread(pred_file) > 0

        intersection = (gt & pred).sum()
        union = gt.sum() + pred.sum()

        return (
            2 * intersection / union
            if union > 0 else 1.0
        )

    

from scipy.optimize import linear_sum_assignment
from skimage.io import imread
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
        gt_files = sorted(gt_dir.glob("*tiff"))
        results = {}

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*tiff"))
            file_pairs = list(
                zip(gt_files, model_files)
            )

            per_image_f1 = run_parallel(
                file_pairs,
               self. _object_f1_worker,
                self.n_processes
            )

            results[str(model_dir)] = per_image_f1
        
        plot_dice_boxplot(results,gt_dir,self.name)

        return results

    @staticmethod
    def _greedy_object_f1(gt_mask, pred_mask, iou_thresh=0.5):

        iou_matrix, _, _ = object_iou_matrix_fast(
            gt_mask,
            pred_mask
        )

        num_gt, num_pred = iou_matrix.shape

        if num_gt == 0 and num_pred == 0:
            return 1.0
        if num_gt == 0 or num_pred == 0:
            return 0.0

        matched = 0
        pred_used = np.zeros(num_pred, dtype=bool)

        for gt_idx in range(num_gt):

            row = iou_matrix[gt_idx].copy()

            row[pred_used] = -1

            best_pred = np.argmax(row)
            best_iou = row[best_pred]

            if best_iou >= iou_thresh:
                matched += 1
                pred_used[best_pred] = True

        precision = matched / num_pred
        recall = matched / num_gt

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)

        return 0.0
    
    @staticmethod
    def _object_f1_worker(pair):

        gt_file, pred_file = pair

        gt_mask = label(imread(gt_file))
        pred_mask = label(imread(pred_file))

        return MeanObjectF1._greedy_object_f1(
            gt_mask,
            pred_mask
        )


# ------------------------------
# Panoptic F1
# ------------------------------
@MetricRegistry.register
class PanopticF1(Metric):
    name = "Panoptic F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir = Path(gt_dir)
        gt_files = sorted(gt_dir.glob("*tiff"))
        results = {}

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*tiff"))

            file_pairs = list(
                zip(gt_files, model_files)
            )
            per_image_pf1 = run_parallel(
                file_pairs,
                self._panoptic_worker,
                self.n_processes
            )

            results[str(model_dir)] = per_image_pf1
        
        plot_dice_boxplot(results,gt_dir,self.name)

        return results

    @staticmethod
    def _panoptic_f1(gt_mask, pred_mask, iou_thresh=0.5):
        """
        Fast Panoptic F1 using IoU matrix + greedy matching.
        """

        # Compute IoU matrix (fast histogram method)
        iou_matrix, _, _ = object_iou_matrix_fast(
            gt_mask,
            pred_mask
        )

        num_gt, num_pred = iou_matrix.shape

        if num_gt == 0 and num_pred == 0:
            return 1.0
        if num_gt == 0 or num_pred == 0:
            return 0.0

        tp = 0
        fp = num_pred
        fn = num_gt

        pred_used = np.zeros(num_pred, dtype=bool)

        for gt_idx in range(num_gt):

            row = iou_matrix[gt_idx].copy()

            # mask used predictions
            row[pred_used] = -1

            best_pred = np.argmax(row)
            best_iou = row[best_pred]

            if best_iou >= iou_thresh:
                tp += 1
                pred_used[best_pred] = True
                fp -= 1
                fn -= 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)

        return 0.0
    
    @staticmethod
    def _panoptic_worker(pair):

        gt_file, pred_file = pair

        gt_mask = label(imread(gt_file))
        pred_mask = label(imread(pred_file))

        return PanopticF1._panoptic_f1(
            gt_mask,
            pred_mask
        )

# ------------------------------
# Boundary F1
# ------------------------------
from skimage.segmentation import find_boundaries

@MetricRegistry.register
class BoundaryF1(Metric):
    name = "Boundary F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir = Path(gt_dir)
        gt_files = sorted(gt_dir.glob("*tiff"))
        results = {}

        for model_dir in models_dirs:
            model_dir = Path(model_dir)
            model_files = sorted(model_dir.glob("*tiff"))

            file_pairs = list(
                zip(gt_files, model_files)
            )

            per_image_bf1 =  run_parallel(
                file_pairs,
                self._boundary_worker,
                self.n_processes
            )

            results[str(model_dir)] = per_image_bf1
        plot_dice_boxplot(results,gt_dir,self.name)

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
    
    @staticmethod
    def _boundary_worker(pair):

        gt_file, pred_file = pair

        gt_mask = imread(gt_file) > 0
        pred_mask = imread(pred_file) > 0

        return BoundaryF1._boundary_f1(
            gt_mask,
            pred_mask
        )


# ------------------------------
# Counts Comparison
# ------------------------------
#   Assumes that true counts are in a csv file 
#       - one column has filename that matches the name of the prediction
#       - second column has the counts that you want to compare against
#

import pandas as pd
import numpy as np
import os
import tifffile
import matplotlib.pyplot as plt
from skimage.measure import label

@MetricRegistry.register
class CountsComparison(Metric):
    name = "Compare Counts"

    def compute(self, gt_csv, model_preds):

        gt_data = pd.read_csv(gt_csv)
        colors = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(6, 6))
        all_gt = []

        for i, model_dir in enumerate(model_preds):
            color = colors[i % len(colors)]

            gt_counts = []
            pred_counts = []

            for _, row in gt_data.iterrows():
                filename = row['File']
                gt_count = row['Counts']

                pred_path = os.path.join(model_dir, filename)
                if not os.path.exists(pred_path):
                    stem = os.path.splitext(filename)[0]
                    candidates = [f for f in os.listdir(model_dir)
                                if os.path.splitext(f)[0] == stem]
                    if not candidates:
                        print(f"Warning: no prediction found for {filename} in {model_dir}, skipping")
                        continue
                    pred_path = os.path.join(model_dir, candidates[0])

                pred_img = tifffile.imread(pred_path)
                labeled = label(pred_img, connectivity=2)
                pred_count = labeled.max()

                gt_counts.append(gt_count)
                pred_counts.append(pred_count)

            gt_counts = np.array(gt_counts, dtype=float)
            pred_counts = np.array(pred_counts, dtype=float)
            all_gt.extend(gt_counts)

            slope = np.sum(gt_counts * pred_counts) / np.sum(gt_counts ** 2)
            r_value = np.corrcoef(gt_counts, pred_counts)[0, 1]

            model_name = Path(model_dir).name
            ax.scatter(gt_counts, pred_counts, alpha=0.7, color=color,
                    edgecolors='k', linewidths=0.5, label=f"{model_name} (S={slope:.2f}, R={r_value:.3f})")

            x_line = np.linspace(0, max(all_gt) * 1.05, 100)
            ax.plot(x_line, slope * x_line, color=color, linestyle='--')

        ax.set_xlabel('Ground Truth Counts')
        ax.set_ylabel('Predicted Counts')
        ax.set_title('Predicted vs Ground Truth Counts')
        ax.legend()
        plt.tight_layout()

        save_dir = os.path.dirname(gt_csv) if isinstance(gt_csv, str) else str(Path(model_preds[0]).parent)
        plt.savefig(os.path.join(save_dir, 'counts_correlation.png'), dpi=300)
