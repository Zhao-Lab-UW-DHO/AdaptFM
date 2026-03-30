from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QFileDialog, QListWidget
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

        # Compute button
        self.compute_button = QPushButton("Compute Metric")
        self.compute_button.clicked.connect(self.compute_metrics)
        self.layout.addWidget(self.compute_button)

    def select_ground_truth(self):
        self.gt_dir = QFileDialog.getExistingDirectory(None, "Select ground truth")
        print(f"Selected GT: {self.gt_dir}")

    def add_model(self):
        model_dir = QFileDialog.getExistingDirectory(None, "Select model predictions")
        if model_dir:
            self.models_dirs.append(model_dir)
            self.models_list.addItem(model_dir)

    def compute_metrics(self):
        metric_name = self.metric_dropdown.currentText()
        metric = MetricRegistry.create(metric_name)

        if not self.gt_dir or not self.models_dirs:
            print("Select ground truth and at least one model!")
            return

        # just pass paths to the metric
        result = metric.compute(self.gt_dir, self.models_dirs)
        
