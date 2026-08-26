# Contributing to AdaptFM

Thank you for your interest in contributing to AdaptFM! We welcome community contributions.

---

## How to Contribute

### Adding Standard Customizations

If you are looking to add new foundation models, annotation algorithms, or benchmarking metrics, please refer to our customization guide:

* [Customizing AdaptFM Documentation](../customizing-adaptfm/index.md)
  * [Adding New Models](../customizing-adaptfm/adding-models.md)
  * [Adding New Annotation Algorithms](../customizing-adaptfm/adding-annotation-algorithms.md)
  * [Adding Benchmarks](../customizing-adaptfm/custom-benchmarks.md)
  * [Adding Preprocessing Pipelines](../customizing-adaptfm/adding-preprocessing.md)
  * [Adding Postprocessing Pipelines](../customizing-adaptfm/adding-postprocessing.md)

### Reporting Bugs
If you encounter a bug, installation issue, or unexpected behavior:
* Check existing [GitHub Issues](https://github.com/Zhao-Lab-UW-DHO/AdaptFM/issues) to see if it has already been reported.
* If not, open a new issue using a descriptive title.
* Include details about your operating system, GPU/hardware setup, Python/Conda environment, steps to reproduce, and any relevant traceback error logs if possible.

### Feature Enhancements
If you have an idea for a new feature that doesn't fall within the widget registry implementation pattern documented in the customization documentation:
* Please **open an Issue first** to discuss the proposed feature before writing code or submitting a PR.
* This allows the maintainers to provide feedback, ensure alignment with project goals, and prevent duplicate work.

### Writing Unit Tests
It is recommended that changes/additions come with unit tests in the `tests/` directory using Python's built-in `unittest` framework.

* **Annotation Algorithms**:
  * Create a test file following the naming convention `tests/test_<feature>.py`.
  * Retrieve registered modules directly from registry classes (e.g., `SEGMENTATION_REGISTRY.get("YourAlgorithmName")`).
  * Pull default parameters dynamically using `instance.tunable_params()`.
  * Test execution using synthetic `numpy` arrays (e.g., uint8 3D volume arrays `(Z, Y, X)`).
  * Assert output properties such as shape matching (`result.shape == input.shape`) and valid output types (`np.integer` or `np.bool_`).

* **Foundation Models**:
  * Use `tempfile.TemporaryDirectory()` inside `setUp()` and `tearDown()` to create temporary input/output directory contexts.
  * Generate mock image inputs locally using `tifffile.imwrite()` to test file I/O operations without relying on external assets.
  * Assert that inference execution creates the expected output files on disk (`expected_file.exists()`).
  * **Conda Environment Integration**: If your foundation model relies on a distinct Conda environment, update `run_tests.sh` to include a test block that activates your environment before invoking `python -m unittest tests/test_<model>.py`.

---

## Code Quality & Guidelines

To maintain code reliability, readability, please ensure pull requests adhere to the following standards:

### 1. Type Hints & Standard Docstrings
All public methods, functions, and registry classes should include Python type hints and clear docstrings detailing parameter definitions, return types, and descriptions.

### 2. Aspire to Windows Compatibility
AdaptFM intends to be "Windows friendly" repository. Do not write logic that uses / as a string identifier and use pathlib.Path such that the ruff linter passes all checks on the branch.

## Items to Check Before Submitting Code
1. Check for ruff errors (e.g. pathlib was not used for a file operation) with `ruff check .` in the root of the repository.
  Some errors may be auto fixed with `ruff check --fix .`.
1. Check for the presence of '/' in filepaths using regex `'[^']*/[^']*'|"[^"]*/[^"]*"` (includes false positives).
1. Optionally enforce good code style with `ruff format --isolated <file to format>`.
1. Ensure qt popups/dialogs are correctly given a parent.