import napari
from gui_scripts.volume.volume_manager import VolumeManager 
from gui_scripts.segmentation.manager import SegmentationManager 
from gui_scripts.dataset.dataset_manager import DatasetManager
from gui_scripts.gui.widgets.seg_widget import SegmentationWidget
from gui_scripts.gui.widgets.save_widget import SaveWidget
from gui_scripts.gui.widgets.session_widget import SessionWidget
from gui_scripts.session.annotation_session import AnnotationSession
from gui_scripts.gui.widgets.inference_widget import InferenceWidget
from gui_scripts.gui.widgets.training_widget import TrainingWidget
from gui_scripts.gui.widgets.benchmark_widget import BenchmarkWidget
from gui_scripts.model.registry import MODEL_REGISTRY
from qtpy.QtWidgets import QAction


def main():
    viewer = napari.Viewer()

    # Core managers
    vm = VolumeManager()
    sm = SegmentationManager(vm)
    session = AnnotationSession([])
    dm = DatasetManager()

    # Model registry (shared by training + inference)
    model_registry = MODEL_REGISTRY
    # or: model_registry = MODEL_REGISTRY
    viewer.window.add_dock_widget(
    SessionWidget(viewer, session, vm, sm).widget,
    area="right"
    )
    # Existing widgets
    viewer.window.add_dock_widget(
        SegmentationWidget(viewer, sm).widget,
        area="right"
    )

    viewer.window.add_dock_widget(
        SaveWidget(viewer, sm).widget,
        area="right"
    )


    # --- NEW: Training ---
    menu = viewer.window._qt_window.menuBar().addMenu("Models")

    train_action = QAction("Training", viewer.window._qt_window)
    infer_action = QAction("Inference", viewer.window._qt_window)

    menu.addAction(train_action)
    menu.addAction(infer_action)

    # lazy-create floating widgets
    train_widget = TrainingWidget(dataset_manager=dm).widget
    infer_widget = InferenceWidget(dataset_manager=dm).widget

    train_action.triggered.connect(train_widget.show)
    infer_action.triggered.connect(infer_widget.show)

# Add Benchmark menu
    menu = viewer.window._qt_window.menuBar().addMenu("Benchmark")
    benchmark_action = QAction("Run Benchmark", viewer.window._qt_window)
    menu.addAction(benchmark_action)

    # Lazy-create widget
    benchmark_widget = BenchmarkWidget()

    # Show widget when menu action triggered
    benchmark_action.triggered.connect(benchmark_widget.show)


    napari.run()


if __name__ == "__main__":
    main()
