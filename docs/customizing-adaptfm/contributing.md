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

### Reporting Bugs
If you encounter a bug, installation issue, or unexpected behavior:
* Check existing [GitHub Issues](https://github.com/Zhao-Lab-UW-DHO/AdaptFM/issues) to see if it has already been reported.
* If not, open a new issue using a descriptive title.
* Include details about your operating system, GPU/hardware setup, Python/conda environment, steps to reproduce, and any relevant traceback error logs if possible.

### Feature Enhancements
If you have an idea for a new feature that doesn't fall within the widget registry implementation pattern documented in the customization documentation:
* Please **open an Issue first** to discuss the proposed feature before writing code or submitting a PR.
* This allows the maintainers to provide feedback, ensure alignment with project goals, and prevent duplicate work.

---

## Code Quality & Guidelines

To maintain code reliability, readability, plase ensure pull requests adhere to the following standards:

### 1. Type Hints & Standard Docstrings
All public methods, functions, and registry classes should include Python type hints and clear docstrings detailing parameter definitions, return types, and descriptions.

### 2. Aspire to Windows Compatability
AdaptFM intends to be "Windows friendly" repository. Do not write logic that uses / as a string identifier and use pathlib.Path such that the ruff linter passes all checks on the branch.

## Items to check before submitting code
1. Check for ruff errors (e.g. pathlib was not used for a file operation) with `ruff check .` in the root of the repository.
  Some errors may be autofixed with `ruff check --fix .`.
1. Check for the presence of '/' in filepaths using regex `'[^']*/[^']*'|"[^"]*/[^"]*"` (includes false positives).
1. Optionally enforce good codestyle with `ruff format --isolated <file to format>`.
1. Ensure qt popups/dialogs are correctly given a parent.