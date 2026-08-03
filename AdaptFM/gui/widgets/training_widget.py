
"""
training_widget.py
------------------
Training workflow window.

Flow
----
1. User selects model, dataset folder, output folder, GPU.
2. User optionally loads / edits parameters.
3. Click Run →
     a. model.prepare_dataset()  (Python, on a QThread)
     b. For nnUNet: preprocessing step via QProcess, then training via QProcess
     c. All other models: single training QProcess
4. Live stdout streams into the log panel.
5. "Terminate" kills the process (and its process group) at any time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from qtpy.QtCore import QThread, Signal, QObject
from qtpy.QtWidgets import QWidget, QLabel

from AdaptFM.gui.widgets.model_widget import (
    ModelWorkflowWidget, _section_label, _RED, _AMBER, _GREEN,
)
from AdaptFM.model.nnUNetV2Spec import NNUNetV2ModelSpec


# ---------------------------------------------------------------------------
# Background worker: prepare_dataset() can do heavy file I/O
# ---------------------------------------------------------------------------

class _PrepareWorker(QObject):
    finished = Signal(object)   # dataset_info (any type the model returns)
    error    = Signal(str)

    def __init__(self, model, dataset_manager, output_dir, params):
        super().__init__()
        self._model          = model
        self._dataset_manager = dataset_manager
        self._output_dir     = output_dir
        self._params         = params

    def run(self):
        try:
            result = self._model.prepare_dataset(
                self._dataset_manager,
                self._output_dir,
                params=self._params,
            )
            self.finished.emit(result)
        except TypeError:
            # Some models don't accept params kwarg — try without
            try:
                result = self._model.prepare_dataset(
                    self._dataset_manager,
                    self._output_dir,
                )
                self.finished.emit(result)
            except Exception as exc:
                self.error.emit(str(exc))
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# TrainingWidget
# ---------------------------------------------------------------------------

class TrainingWidget(ModelWorkflowWidget):
    WINDOW_TITLE = "Training"
    SKIP_PARAMS  = False

    def __init__(self, dataset_manager):
        self._prep_thread: Optional[QThread] = None
        self._prep_worker: Optional[_PrepareWorker] = None
        self.output_dir: Optional[Path] = None
        super().__init__(dataset_manager)

    # ------------------------------------------------------------------
    # Workflow
    # ------------------------------------------------------------------

    def _run_workflow(self):
        # Already validated by base: model set, output_dir set, no process running
        if self.dataset_dir is None:
            self._log_line("⚠  Please select a dataset folder.", color=_AMBER)
            return

        params  = self.collect_params()
        gpu     = self._gpu_spin.value()
        params["gpu"] = gpu

        self._log_line(
            f"Preparing dataset for {self.model.name}…",
            color="#aaa",
        )
        self._set_busy(True)

        # Parent thread to widget to keep it alive past this method's
        # return. deleteLater() + None-clear clean up after it finishes.
        self._prep_thread = QThread(self.widget)
        self._prep_worker = _PrepareWorker(
            self.model, self.dataset_manager, self.output_dir, dict(params)
        )
        self._prep_worker.moveToThread(self._prep_thread)
        self._prep_thread.started.connect(self._prep_worker.run)
        self._prep_worker.finished.connect(
            lambda info: self._on_dataset_ready(info, params, gpu)
        )
        self._prep_worker.error.connect(self._on_prepare_error)
        self._prep_worker.finished.connect(self._prep_thread.quit)
        self._prep_worker.error.connect(self._prep_thread.quit)
        self._prep_thread.finished.connect(self._prep_thread.deleteLater)
        self._prep_thread.finished.connect(lambda: setattr(self, '_prep_thread', None))
        self._prep_thread.start()

    def _on_prepare_error(self, msg: str):
        self._set_busy(False)
        self._log_line(f"[dataset error] {msg}", color=_RED)

    def _on_dataset_ready(self, dataset_info, params: dict, gpu: int):
        self._log_line("✓  Dataset ready.", color=_GREEN)

        env_extra = {"CUDA_VISIBLE_DEVICES": str(gpu)}

        # Build the conda-wrapped command list
        training_cmd  = self.model.training_command(dataset_info, params, self.output_dir)
        full_train_cmd = self.model._wrap_with_conda(training_cmd)

        if isinstance(self.model, NNUNetV2ModelSpec):
            # nnUNet: preprocess → train (sequential chain)
            preprocess_cmd = self.model.preprocessing_command(
                dataset_dir=dataset_info,
                params=params,
                output_dir=self.output_dir,
            )
            full_pre_cmd = self.model._wrap_with_conda(preprocess_cmd)

            self._run_chain(
                steps=[
                    (full_pre_cmd[0],   full_pre_cmd[1:],   "Preprocessing"),
                    (full_train_cmd[0], full_train_cmd[1:], "Training"),
                ],
                env_extra=env_extra,
                on_all_done=self._on_training_done,
            )
        else:
            self._start_process(
                full_train_cmd[0],
                full_train_cmd[1:],
                label=f"Training  [{self.model.name}]",
                env_extra=env_extra,
                on_done=self._on_training_done,
            )

    def _on_training_done(self, exit_code: int):
        self._set_busy(False)
        if exit_code == 0:
            self._log_line(
                f"\n✓  Training complete. Output: {self.output_dir}",
                color=_GREEN, bold=True,
            )
        else:
            self._log_line(
                f"\n✗  Training failed (exit {exit_code}).",
                color=_RED, bold=True,
            )

    # ------------------------------------------------------------------
    # Terminate override: also stop any in-flight prep thread
    # ------------------------------------------------------------------

    def _on_terminate_clicked(self):
        # If dataset prep is still running, stop it first
        if self._prep_thread and self._prep_thread.isRunning():
            self._prep_thread.quit()
            self._prep_thread.wait(2000)
            self._log_line("■  Dataset preparation cancelled.", color=_RED)
            self._set_busy(False)
            return
        # Otherwise delegate to base (kills QProcess)
        super()._on_terminate_clicked()