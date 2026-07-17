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

from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import SimpleITK as sitk
from skimage.measure import label
from skimage.segmentation import find_boundaries
from skimage.io import imread


# ──────────────────────────────────────────────
# File I/O
# ──────────────────────────────────────────────

TIFF_SUFFIXES  = {".tiff", ".tif"}
NIFTI_SUFFIXES = {".gz"}   # catches .nii.gz; the outer suffix is always .gz


def read_mask(path: Path) -> np.ndarray:
    """
    Load a segmentation mask from either a TIFF or NIfTI (.nii.gz) file and
    return it as a NumPy array.  The array dtype is preserved so callers can
    apply their own thresholding.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in TIFF_SUFFIXES:
        return tiff.imread(path)

    if suffix in NIFTI_SUFFIXES:
        # SimpleITK reads .nii, .nii.gz, .mha, etc.
        sitk_img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(sitk_img)  # shape: (Z, Y, X) or (Y, X)
        return arr

    raise ValueError(
        f"Unsupported file format '{suffix}' for path: {path}\n"
        f"Supported formats: {TIFF_SUFFIXES | NIFTI_SUFFIXES}"
    )


def glob_masks(directory: Path):
    """
    Return all TIFF and NIfTI mask files in *directory*, sorted.
    Files are discovered by their suffix; mixed collections are supported.
    """
    directory = Path(directory)
    found = sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in TIFF_SUFFIXES | NIFTI_SUFFIXES
    )
    return found


# ──────────────────────────────────────────────
# Parallel execution
# ──────────────────────────────────────────────

def run_parallel(file_pairs, worker_fn, n_processes=None):
    """
    Run worker_fn over file_pairs in parallel.

    file_pairs:  list[(gt_path, pred_path)]
    worker_fn:   callable(pair) -> scalar
    n_processes: int or None (defaults to cpu_count - 1)
    """
    if n_processes is None:
        n_processes = max(cpu_count() - 1, 1)

    if n_processes == 1:
        return [worker_fn(p) for p in file_pairs]

    with Pool(processes=n_processes) as pool:
        results = pool.map(worker_fn, file_pairs)

    return results


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_dice_boxplot(results, gt_dir, name):
    """
    results: dict {model_name: [scores]}
    """
    model_names = [Path(k).name for k in results.keys()]
    score_values = list(results.values())

    plt.figure(figsize=(10, 6))
    plt.boxplot(score_values, tick_labels=model_names, showmeans=True)
    plt.ylabel(f"{name}")
    plt.xlabel("Model")
    plt.title(f"{name} Distribution per Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(gt_dir, f"{name}_boxplot.png"), dpi=300)


# ──────────────────────────────────────────────
# IoU matrix helper
# ──────────────────────────────────────────────

def object_iou_matrix_fast(gt_mask, pred_mask):
    """
    Fast computation of the IoU matrix between all GT and predicted objects.
    Works on 2-D or 3-D masks.
    """
    gt_labels  = label(gt_mask)
    pred_labels = label(pred_mask)

    num_gt   = int(gt_labels.max())
    num_pred = int(pred_labels.max())

    gt_flat   = gt_labels.ravel()
    pred_flat = pred_labels.ravel()

    combined = gt_flat * (num_pred + 1) + pred_flat
    counts   = np.bincount(combined)

    iou_matrix = np.zeros((num_gt, num_pred), dtype=float)

    for idx, count in enumerate(counts):
        if count == 0:
            continue
        gt_id   = idx // (num_pred + 1)
        pred_id = idx %  (num_pred + 1)
        if gt_id == 0 or pred_id == 0:
            continue  # background
        iou_matrix[gt_id - 1, pred_id - 1] = count

    gt_area   = np.bincount(gt_flat,   minlength=num_gt   + 1)[1:]
    pred_area = np.bincount(pred_flat, minlength=num_pred + 1)[1:]
    union_matrix = gt_area[:, None] + pred_area[None, :] - iou_matrix
    iou_matrix   = iou_matrix / np.maximum(union_matrix, 1e-12)

    return iou_matrix, gt_labels, pred_labels


# ──────────────────────────────────────────────
# Metric base + registry
# ──────────────────────────────────────────────

class Metric(ABC):
    name: str = "BaseMetric"

    def __init__(self, n_processes=None):
        self.n_processes = n_processes

    @abstractmethod
    def compute(self, gt_dir, pred_dirs):
        """Compute the metric between ground truth and one or more prediction dirs."""
        pass


class MetricRegistry:
    _metrics = {}

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


# ──────────────────────────────────────────────
# Dice Score
# ──────────────────────────────────────────────

@MetricRegistry.register
class DiceScore(Metric):
    name = "Dice Score"

    def compute(self, gt_dir, models_dirs):
        gt_dir   = Path(gt_dir)
        gt_files = glob_masks(gt_dir)
        results  = {}

        for model_dir in models_dirs:
            model_dir   = Path(model_dir)
            model_files = glob_masks(model_dir)
            file_pairs  = list(zip(gt_files, model_files))

            per_image_dice = run_parallel(
                file_pairs,
                self._dice_worker,
                self.n_processes,
            )
            results[str(model_dir)] = per_image_dice

        plot_dice_boxplot(results, gt_dir, self.name)
        return results

    @staticmethod
    def _dice_worker(pair):
        gt_file, pred_file = pair

        gt   = read_mask(gt_file)   > 0
        pred = read_mask(pred_file) > 0

        intersection = (gt & pred).sum()
        union        = gt.sum() + pred.sum()

        return 2 * intersection / union if union > 0 else 1.0


# ──────────────────────────────────────────────
# Mean Object F1
# ──────────────────────────────────────────────

@MetricRegistry.register
class MeanObjectF1(Metric):
    name = "Mean Object F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir   = Path(gt_dir)
        gt_files = glob_masks(gt_dir)
        results  = {}

        for model_dir in models_dirs:
            model_dir   = Path(model_dir)
            model_files = glob_masks(model_dir)
            file_pairs  = list(zip(gt_files, model_files))

            per_image_f1 = run_parallel(
                file_pairs,
                self._object_f1_worker,
                self.n_processes,
            )
            results[str(model_dir)] = per_image_f1

        plot_dice_boxplot(results, gt_dir, self.name)
        return results

    @staticmethod
    def _greedy_object_f1(gt_mask, pred_mask, iou_thresh=0.5):
        iou_matrix, _, _ = object_iou_matrix_fast(gt_mask, pred_mask)
        num_gt, num_pred = iou_matrix.shape

        if num_gt == 0 and num_pred == 0:
            return 1.0
        if num_gt == 0 or num_pred == 0:
            return 0.0

        matched   = 0
        pred_used = np.zeros(num_pred, dtype=bool)

        for gt_idx in range(num_gt):
            row = iou_matrix[gt_idx].copy()
            row[pred_used] = -1
            best_pred = np.argmax(row)
            best_iou  = row[best_pred]
            if best_iou >= iou_thresh:
                matched += 1
                pred_used[best_pred] = True

        precision = matched / num_pred
        recall    = matched / num_gt

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    @staticmethod
    def _object_f1_worker(pair):
        gt_file, pred_file = pair

        gt_mask   = label(read_mask(gt_file))
        pred_mask = label(read_mask(pred_file))

        return MeanObjectF1._greedy_object_f1(gt_mask, pred_mask)


# ──────────────────────────────────────────────
# Panoptic F1
# ──────────────────────────────────────────────

@MetricRegistry.register
class PanopticF1(Metric):
    name = "Panoptic F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir   = Path(gt_dir)
        gt_files = glob_masks(gt_dir)
        results  = {}

        for model_dir in models_dirs:
            model_dir   = Path(model_dir)
            model_files = glob_masks(model_dir)
            file_pairs  = list(zip(gt_files, model_files))

            per_image_pf1 = run_parallel(
                file_pairs,
                self._panoptic_worker,
                self.n_processes,
            )
            results[str(model_dir)] = per_image_pf1

        plot_dice_boxplot(results, gt_dir, self.name)
        return results

    @staticmethod
    def _panoptic_f1(gt_mask, pred_mask, iou_thresh=0.5):
        iou_matrix, _, _ = object_iou_matrix_fast(gt_mask, pred_mask)
        num_gt, num_pred = iou_matrix.shape

        if num_gt == 0 and num_pred == 0:
            return 1.0
        if num_gt == 0 or num_pred == 0:
            return 0.0

        tp        = 0
        fp        = num_pred
        fn        = num_gt
        pred_used = np.zeros(num_pred, dtype=bool)

        for gt_idx in range(num_gt):
            row = iou_matrix[gt_idx].copy()
            row[pred_used] = -1
            best_pred = np.argmax(row)
            best_iou  = row[best_pred]
            if best_iou >= iou_thresh:
                tp += 1
                pred_used[best_pred] = True
                fp -= 1
                fn -= 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision + recall > 0:
            return 2 * precision * recall / (precision + recall)
        return 0.0

    @staticmethod
    def _panoptic_worker(pair):
        gt_file, pred_file = pair

        gt_mask   = label(read_mask(gt_file))
        pred_mask = label(read_mask(pred_file))

        return PanopticF1._panoptic_f1(gt_mask, pred_mask)


# ──────────────────────────────────────────────
# Boundary F1
# ──────────────────────────────────────────────

@MetricRegistry.register
class BoundaryF1(Metric):
    name = "Boundary F1"

    def compute(self, gt_dir, models_dirs):
        gt_dir   = Path(gt_dir)
        gt_files = glob_masks(gt_dir)
        results  = {}

        for model_dir in models_dirs:
            model_dir   = Path(model_dir)
            model_files = glob_masks(model_dir)
            file_pairs  = list(zip(gt_files, model_files))

            per_image_bf1 = run_parallel(
                file_pairs,
                self._boundary_worker,
                self.n_processes,
            )
            results[str(model_dir)] = per_image_bf1

        plot_dice_boxplot(results, gt_dir, self.name)
        return results

    @staticmethod
    def _boundary_f1(gt_mask, pred_mask):
        """
        Boundary F1 (precision/recall of boundary pixels/voxels).
        Works for both 2-D and 3-D masks via skimage.segmentation.find_boundaries.
        """
        gt_bound   = find_boundaries(gt_mask,   mode="inner")
        pred_bound = find_boundaries(pred_mask, mode="inner")

        tp = np.logical_and(gt_bound,  pred_bound).sum()
        fp = np.logical_and(pred_bound, ~gt_bound).sum()
        fn = np.logical_and(gt_bound,  ~pred_bound).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )
        return f1

    @staticmethod
    def _boundary_worker(pair):
        gt_file, pred_file = pair

        gt_mask   = read_mask(gt_file)   > 0
        pred_mask = read_mask(pred_file) > 0

        return BoundaryF1._boundary_f1(gt_mask, pred_mask)

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

        save_dir = str(Path(gt_csv).parent) if isinstance(gt_csv, str) else str(Path(model_preds[0]).parent)
        plt.savefig((Path(save_dir) / 'counts_correlation.png'), dpi=300)
