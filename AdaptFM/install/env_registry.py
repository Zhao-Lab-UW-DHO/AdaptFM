"""
env_registry.py
---------------
Declarative registry of every installable environment AdaptFM ships.

Each entry describes *what* an environment is, not *how* to install it —
the install script already encodes the how.  The registry adds the metadata
the GUI needs: where to check for updates, which conda env to probe, etc.

To add a new environment:
  1. Write its install/uninstall entry-point scripts as normal.
  2. Add an EnvironmentSpec entry below.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


UpdateSource = Literal["pypi", "github", "conda", "none"]


@dataclass
class PipPackageSpec:
    """A pip-installed package whose version the GUI should track."""
    import_name: str          # name used in `pip show` / PyPI JSON API
    display_name: str = ""    # human-friendly label (falls back to import_name)
    update_source: UpdateSource = "pypi"
    # For github source:
    github_repo: str = ""     # "owner/repo"
    # For conda source:
    conda_package: str = ""   # conda package name (if different from import_name)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.import_name


@dataclass
class EnvironmentSpec:
    """Everything the GUI needs to know about one installable environment."""

    # --- Identity ---
    key: str                  # unique machine key, e.g. "cellpose_sam"
    display_name: str         # shown in the GUI card header
    description: str          # one-line blurb shown under the name
    conda_env_name: str       # the actual conda env created on disk

    # --- Install / uninstall commands (entry-point names from setup.cfg) ---
    install_command: str      # e.g. "install-cellposesam"
    uninstall_command: str    # e.g. "uninstall-cellposesam"  (can be empty)

    # --- Pip packages to version-track ---
    pip_packages: list[PipPackageSpec] = field(default_factory=list)

    # --- Flags ---
    requires_pytorch_swap: bool = False   # show PyTorch-swap warning in UI
    requires_gpu: bool = True
    optional: bool = False                # mark non-essential extras

    # --- Docs / homepage ---
    docs_url: str = ""


# ---------------------------------------------------------------------------
# Registry — extend this list as you add new models
# ---------------------------------------------------------------------------

ENV_REGISTRY: list[EnvironmentSpec] = [
    EnvironmentSpec(
        key="cellpose_sam",
        display_name="CellposeSAM",
        description="Cellpose segmentation with SAM backbone (requires custom PyTorch build).",
        conda_env_name="cellpose_adapt",
        install_command="adaptfm-install-cellposesam",
        uninstall_command="adaptfm-uninstall cellpose_adapt",
        requires_pytorch_swap=True,
        requires_gpu=True,
        pip_packages=[
            PipPackageSpec(
                import_name="cellpose",
                display_name="Cellpose",
                update_source="pypi",
            ),
            PipPackageSpec(
                import_name="torch",
                display_name="PyTorch",
                update_source="pypi",
            ),
        ],
        docs_url="https://github.com/MouseLand/cellpose",
    ),
    # ------------------------------------------------------------------ #
    # Add more environments here, e.g.:
    #
    EnvironmentSpec(
        key="microSAM",
        display_name="microSAM",
        description="Segment Anything for Microscopy",
        conda_env_name="micro-sam_adapt",
        install_command="adaptfm-install-microsam",
        uninstall_command="adaptfm-uninstall micro-sam_adapt",
        requires_pytorch_swap=False,
        pip_packages=[
            PipPackageSpec("micro-sam", display_name="microSAM",
                           update_source="conda",
                           conda_package="micro_sam"),
        ],
        docs_url='https://computational-cell-analytics.github.io/micro-sam/micro_sam.html'
    ),



    EnvironmentSpec(
        key="nnUNet",
        display_name="nnUNet",
        description="A self-configuring method for deep learning-based biomedical image segmentation",
        conda_env_name="nnUNet_adapt",
        install_command="adaptfm-install-nnunet",
        uninstall_command="adaptfm-uninstall nnUNet_adapt",
        requires_gpu=True,
        pip_packages=[
            PipPackageSpec("nnunetv2",
                           display_name="nnUNet",
                           update_source="pypi"),
            PipPackageSpec(
                import_name="torch",
                display_name="PyTorch",
                update_source="pypi",
            )],
        docs_url="https://github.com/MIC-DKFZ/nnUNet"
    ),


    EnvironmentSpec(
        key="sammed3d",
        display_name="SAM-Med3D",
        description="SAM-Med3D: Towards General-purpose Segmentation Models for Volumetric Medical Images",
        conda_env_name="sammed3d_adapt",
        install_command="adaptfm-install-sammed3d",
        uninstall_command="adaptfm-uninstall",
        requires_gpu=True,
        pip_packages=[
            PipPackageSpec("segment_anything",
                           display_name="SAM-Med3D",
                           update_source="github",
                           github_repo="https://github.com/uni-medical/SAM-Med3D.git")
        ],
        docs_url="https://github.com/uni-medical/sam-med3d"
    ),

    EnvironmentSpec(
        key="CellSAM",
        display_name='CellSAM',
        description="A foundation model for cell segmentation",
        conda_env_name="cellsam_adapt",
        install_command="adaptfm-install-cellsam",
        uninstall_command="adaptfm-uninstall",
        requires_gpu=True,
        pip_packages=[
            PipPackageSpec("cellSAM",
                           display_name="CellSAM",
                           update_source="github",
                           github_repo="git+https://github.com/vanvalenlab/cellSAM.git")
        ],
        docs_url="https://vanvalenlab.github.io/cellSAM/"
        )

    # ------------------------------------------------------------------ #
]

# Quick lookup by key
ENV_REGISTRY_MAP: dict[str, EnvironmentSpec] = {e.key: e for e in ENV_REGISTRY}