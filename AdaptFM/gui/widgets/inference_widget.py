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
from typing import Optional

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog,
)

from AdaptFM.gui.widgets.model_widget import (
    ModelWorkflowWidget, _section_label, _secondary_btn_style,
    _RED, _AMBER, _GREEN, _TEXT, _MUTED,
)


class InferenceWidget(ModelWorkflowWidget):
    WINDOW_TITLE = "Inference"
    SKIP_PARAMS  = True   # inference widgets don't expose tunable params

    def __init__(self, dataset_manager):
        self.checkpoint_path: Optional[Path] = None
        self.output_dir: Optional[Path] = None
        # checkpoint label ref — set in _extra_controls, used in _on_model_changed
        self._checkpoint_lbl: Optional[QLabel] = None
        super().__init__(dataset_manager)

    # ------------------------------------------------------------------
    # Extra controls: checkpoint selector, inserted above Run by the base
    # ------------------------------------------------------------------

    def _extra_controls(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(_section_label("Model checkpoint"))

        row = QHBoxLayout()
        self._checkpoint_lbl = QLabel("No checkpoint selected")
        self._checkpoint_lbl.setStyleSheet(
            f"color: {_MUTED}; font-size: 11px;"
        )
        self._checkpoint_lbl.setWordWrap(True)
        row.addWidget(self._checkpoint_lbl, stretch=1)

        btn = QPushButton("Browse…")
        btn.setFixedWidth(80)
        btn.setStyleSheet(_secondary_btn_style())
        btn.clicked.connect(self._select_checkpoint)
        row.addWidget(btn)

        layout.addLayout(row)
        return container

    # ------------------------------------------------------------------
    # Checkpoint selection
    # ------------------------------------------------------------------

    def _select_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self.widget, "Select model checkpoint"
        )
        if path:
            self.checkpoint_path = Path(path)
            self._checkpoint_lbl.setText(str(self.checkpoint_path))
            self._checkpoint_lbl.setStyleSheet(
                f"color: {_TEXT}; font-size: 11px;"
            )

    def _on_model_changed(self):
        """Clear checkpoint whenever the model changes."""
        self.checkpoint_path = None
        if self._checkpoint_lbl is not None:
            self._checkpoint_lbl.setText("No checkpoint selected")
            self._checkpoint_lbl.setStyleSheet(
                f"color: {_MUTED}; font-size: 11px;"
            )

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def _run_workflow(self):
        if self.dataset_dir is None:
            self._log_line("⚠  Please select a dataset folder.", color=_AMBER)
            return
        
        conda_env = Path(self.model.conda_env)
        if not conda_env.exists():
            raise RuntimeError(
            f"{self.model.conda_env} not installed. "
            "You must first install the environment with the environment manager before using."
            )


        params = self.collect_params()
        gpu    = self._gpu_spin.value()
        params["gpu"] = gpu

        env_extra = {"CUDA_VISIBLE_DEVICES": str(gpu)}

        inference_cmd  = self.model.inference_command(
            dataset_dir=self.dataset_dir,
            checkpoint=self.checkpoint_path,
            output_dir=self.output_dir,
        )
        full_cmd = self.model._wrap_with_conda(inference_cmd)

        self._start_process(
            full_cmd[0],
            full_cmd[1:],
            label=f"Inference  [{self.model.name}]",
            env_extra=env_extra,
            on_done=self._on_inference_done,
        )

    def _on_inference_done(self, exit_code: int):
        self._set_busy(False)
        if exit_code == 0:
            self._log_line(
                f"\n✓  Inference complete. Output: {self.output_dir}",
                color=_GREEN, bold=True,
            )
        else:
            self._log_line(
                f"\n✗  Inference failed (exit {exit_code}).",
                color=_RED, bold=True,
            )