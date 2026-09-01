"""
inference_widget.py
-------------------
Inference workflow window.

Flow
----
1. User selects model, dataset folder, output folder, GPU.
2. User selects a model checkpoint file.
3. Click Run → single QProcess running the model's conda-wrapped
   inference command, with live stdout in the log panel.
4. Terminate kills the process at any time.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QLabel

from AdaptFM.gui.widgets.model_widget import (
    _AMBER,
    _GREEN,
    _RED,
    ModelWorkflowWidget,
)
from AdaptFM.postprocessing.post_proc_registry import POSTPROC_REGISTRY


class PostProcessingWidget(ModelWorkflowWidget):
    WINDOW_TITLE = "Post Processing"
    SKIP_PARAMS = True

    def __init__(self, dataset_manager):
        self.registry = POSTPROC_REGISTRY
        self.registry_title = "Post Processing"

        super().__init__(dataset_manager)

        self.output_dir: Path | None = None
        self.dataset_dir: Path | None = None

    def _build(self):
        super()._build()

        # Patch model section
        self._model_combo.clear()
        self._model_combo.addItems(list(POSTPROC_REGISTRY.keys()))

        # Rename label
        for lbl in self.widget.findChildren(QLabel):
            if lbl.text() == "Model":
                lbl.setText("Post Processing")
                break

        # Hide params AFTER base class fires its initial model-selected event
        QTimer.singleShot(0, lambda: self._set_param_section_visible(False))

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def _run_workflow(self):
        if self.dataset_dir is None:
            self._log_line("⚠  Please select a dataset folder.", color=_AMBER)
            return

        params = self.collect_params()
        gpu = self._gpu_spin.value()
        params["gpu"] = gpu

        env_extra = {"CUDA_VISIBLE_DEVICES": str(gpu)}

        post_process_cmd = [
            "python",
            f"{self.model.module_path}",
            "--input_dir",
            self.dataset_dir,
            "--output_dir",
            self.output_dir,
        ]

        full_cmd = self.model._wrap_with_conda(post_process_cmd)

        self._start_process(
            full_cmd[0],
            full_cmd[1:],
            label=f"Postprocessing with  [{self.model.name}]",
            env_extra=env_extra,
            on_done=self._on_postproc_done,
        )

    def _on_postproc_done(self, exit_code: int):
        self._set_busy(False)
        if exit_code == 0:
            self._log_line(
                f"\n✓  Post processing complete. Output: {self.output_dir}",
                color=_GREEN,
                bold=True,
            )
        else:
            self._log_line(
                f"\n✗  Post processing failed (exit {exit_code}).",
                color=_RED,
                bold=True,
            )
