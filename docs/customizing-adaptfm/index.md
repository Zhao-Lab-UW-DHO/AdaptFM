# Customizing AdaptFM

AdaptFM is designed from the ground up with a modular, registry-based architecture. This allows developers, computational biologists, and researchers to easily extend the framework by integrating custom segmentation algorithms, new foundation models, and evaluation benchmarks without modifying core GUI components.

If you plan to contribute your custom components back to the main repository, please review our [Contribution Guide](contributing.md)

---

## Overview of the Extension Architecture

All customizable components in AdaptFM follow a unified pattern:
1. **Inherit** from a defined base class (e.g., `SegmentationAlgorithmSpec`, `FoundationModelSpec`, or `Metric`).
2. **Implement** the required interface methods and parameter schemas.
3. **Register** the class with the corresponding component registry.

Once registered, AdaptFM automatically exposes your algorithm, model, or metric inside the GUI widgets for seamless interactive use and evaluation.

---

## Extensibility Modules

Select a guide below for step-by-step instructions and code templates:

### 1. [Adding New Annotation Algorithms](adding-annotation-algorithms.md)
Learn how to create click-based, bounding-box, or automated 2D/3D segmentation tools by inheriting from `SegmentationAlgorithmSpec` and registering with `SEGMENTATION_REGISTRY`.

### 2. [Adding New Models](adding-models.md)
Learn how to integrate new deep learning architectures or foundation models for inference and fine-tuning. Covers conda environment isolation, dataset preparation hooks (`prepare_dataset`), execution wrappers (`train_wrapper.py` / `inference_wrapper.py`), and registration in `MODEL_REGISTRY`.

### 3. [Adding Custom Benchmarks](custom-benchmarks.md)
Learn how to add custom performance metrics (e.g., Dice score, IoU, Hausdorff distance) to the benchmarking suite by extending the `Metric` base class and registering with `@MetricRegistry.register`.

### 4. [Adding Preprocessing Pipelines](adding-preprocessing.md)
Learn how to add new preprocessing/file conversion pipelines, by writing a custom function and adding it to the preprocessing registry. 

### 5. [Adding Postprocessing Pipelines](adding-postprocessing.md)
Learn how to add a new postprocessing algorithm by duplicating our base PostProc class and writing your own launch code. 