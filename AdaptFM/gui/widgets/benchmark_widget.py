from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog, QListWidget
from qtpy.QtWidgets import QSpinBox
import subprocess
import json
import multiprocessing
import sys
from AdaptFM.gui.widgets.metrics_widget import MetricRegistry

class BenchmarkWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Metric dropdown
        self.metric_dropdown = QComboBox()
        self.metric_dropdown.addItems(MetricRegistry.get_metrics())
        self.layout.addWidget(QLabel("Select Metric"))
        self.layout.addWidget(self.metric_dropdown)

        # Ground truth selection
        self.gt_button = QPushButton("Select Ground Truth")
        self.gt_button.clicked.connect(self.select_ground_truth)
        self.layout.addWidget(self.gt_button)
        self.gt_dir = None

        # Model selection
        self.models_button = QPushButton("Add Model Predictions")
        self.models_button.clicked.connect(self.add_model)
        self.layout.addWidget(self.models_button)
        self.models_dirs = []

        # List of models added
        self.models_list = QListWidget()
        self.layout.addWidget(self.models_list)

                # Remove selected model button
        self.remove_model_button = QPushButton("Remove Selected Model")
        self.remove_model_button.clicked.connect(self.remove_model)
        self.layout.addWidget(self.remove_model_button)

        # Compute button
        self.compute_button = QPushButton("Compute Metric")
        self.compute_button.clicked.connect(self.compute_metrics)
        self.layout.addWidget(self.compute_button)
        self.layout.addWidget(QLabel("Number of Processes"))

        self.nproc_spinbox = QSpinBox()

        self.nproc_spinbox.setMinimum(1)

        self.nproc_spinbox.setMaximum(
            multiprocessing.cpu_count()
        )

        self.nproc_spinbox.setValue(
           1
        )

        self.layout.addWidget(self.nproc_spinbox)

    def select_ground_truth(self):
        self.gt_dir = QFileDialog.getExistingDirectory(None, "Select ground truth")
        print(f"Selected GT: {self.gt_dir}")

    def add_model(self):
        model_dir = QFileDialog.getExistingDirectory(None, "Select model predictions")
        if model_dir:
            self.models_dirs.append(model_dir)
            self.models_list.addItem(model_dir)

    def remove_model(self):
            selected = self.models_list.currentRow()
            if selected >= 0:
                self.models_list.takeItem(selected)
                self.models_dirs.pop(selected)
            


    def compute_metrics(self):
        metric_name = self.metric_dropdown.currentText()

        if not self.gt_dir or not self.models_dirs:
            print("Select ground truth and at least one model!")
            return

        models_json = json.dumps(self.models_dirs)
        
        num_processes = str(
        self.nproc_spinbox.value()
        )

        cmd = [
            sys.executable,
            "-m","AdaptFM.gui.run_metric_subprocess",
            "--metric", metric_name,
            "--gt_dir", self.gt_dir,
            "--models_json", models_json,
            "--num_processes",num_processes
        ]

        print("Launching metric subprocess...")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )


