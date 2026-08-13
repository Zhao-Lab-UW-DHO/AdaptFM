import napari
from AdaptFM.volume.volume_manager import VolumeManager 
from AdaptFM.segmentation.manager import SegmentationManager 
from AdaptFM.dataset.dataset_manager import DatasetManager
from AdaptFM.gui.widgets.seg_widget import SegmentationWidget
from AdaptFM.gui.widgets.save_widget import SaveWidget
from AdaptFM.gui.widgets.session_widget import SessionWidget
from AdaptFM.session.annotation_session import AnnotationSession
from AdaptFM.gui.widgets.inference_widget import InferenceWidget
from AdaptFM.gui.widgets.training_widget import TrainingWidget
from AdaptFM.gui.widgets.benchmark_widget import BenchmarkWidget
from AdaptFM.gui.widgets.env_manager_dialog import EnvironmentManagerDialog
from AdaptFM.gui.widgets.post_proc_widget import PostProcessingWidget
from AdaptFM.model.registry import MODEL_REGISTRY
from AdaptFM.gui.napari_utils import reorder_docks,restore_or_focus_widget,restore_all_widgets
from qtpy.QtWidgets import QAction, QScrollArea, QFrame
from qtpy.QtCore import Qt

def make_scrollable(widget):
    """Wraps a QWidget or magicgui widget in a Qt scroll area."""
    native_widget = widget.native if hasattr(widget, "native") else widget
    
    scroll = QScrollArea()
    scroll.setWidget(native_widget)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return scroll


def main():
    viewer = napari.Viewer()
    viewer.title = "AdaptFM"

    # Core managers
    vm = VolumeManager()
    sm = SegmentationManager(vm)
    session = AnnotationSession([])
    dm = DatasetManager()

    # Model registry (shared by training + inference)
    model_registry = MODEL_REGISTRY


    # Canonical widget definitions defining the exact top-to-bottom layout order
    widget_specs = [
        {
            "name": "AdaptFM Image Manager",
            "create_fn": lambda: SessionWidget(viewer, session, vm, sm).widget,
            "dock": None,
        },
        {
            "name": "AdaptFM Annotation",
            "create_fn": lambda: make_scrollable(SegmentationWidget(viewer, sm).widget),
            "dock": None,
        },
        {
            "name": "AdaptFM Save Image",
            "create_fn": lambda: SaveWidget(viewer, sm).widget,
            "dock": None,
        },
    ]

    # ---------------------------------------------------------
    # Create the default widgets at application startup
    # ---------------------------------------------------------

    for spec in widget_specs:
        spec["dock"] = viewer.window.add_dock_widget(
            spec["create_fn"](),
            area="right",
            name=spec["name"]
        )

    # Force canonical ordering:
    # Session -> Annotation -> Save
    reorder_docks(viewer,
                    widget_specs)

    # --- Restore Widgets Menu ---
    restore_menu = viewer.window._qt_window.menuBar().addMenu("Restore Widgets")

    for idx, spec in enumerate(widget_specs):
        action = QAction(spec["name"], viewer.window._qt_window)
        action.triggered.connect(lambda checked=False, i=idx: restore_or_focus_widget(i,widget_specs,viewer))
        restore_menu.addAction(action)

    restore_menu.addSeparator()
    restore_all_action = QAction("Restore All Widgets", viewer.window._qt_window)
    restore_all_action.triggered.connect(lambda: restore_all_widgets(viewer,widget_specs))
    restore_menu.addAction(restore_all_action)

    # --- Models Menu ---
    menu = viewer.window._qt_window.menuBar().addMenu("Models")

    train_action = QAction("Training", viewer.window._qt_window)
    infer_action = QAction("Inference", viewer.window._qt_window)

    menu.addAction(train_action)
    menu.addAction(infer_action)

    train_widget = TrainingWidget(dataset_manager=dm).widget
    infer_widget = InferenceWidget(dataset_manager=dm).widget

    train_action.triggered.connect(train_widget.show)
    infer_action.triggered.connect(infer_widget.show)

    menu = viewer.window._qt_window.menuBar().addMenu("Post Process")
    post_process_action = QAction("Run Post Processing",viewer.window._qt_window)
    menu.addAction(post_process_action)

    post_proc_widget = PostProcessingWidget()

    post_process_action.triggered.connect(post_proc_widget.show)

    # --- Benchmark Menu ---
    benchmark_menu = viewer.window._qt_window.menuBar().addMenu("Benchmark")
    benchmark_action = QAction("Run Benchmark", viewer.window._qt_window)
    benchmark_menu.addAction(benchmark_action)

    # Lazy-create widget
    benchmark_widget = BenchmarkWidget().widget

    # Show widget when menu action triggered
    benchmark_action.triggered.connect(benchmark_widget.show)

    # --- Environments Menu ---
    env_menu = viewer.window._qt_window.menuBar().addMenu("Environments")
    env_action = QAction("Manage Environments…", viewer.window._qt_window)
    env_menu.addAction(env_action)
 
    _env_dialog: list[EnvironmentManagerDialog] = []
 
    def _open_env_manager():
        if not _env_dialog:
            dlg = EnvironmentManagerDialog(parent=viewer.window._qt_window)
            _env_dialog.append(dlg)
        _env_dialog[0].show()
        _env_dialog[0].raise_()
        _env_dialog[0].activateWindow()
 
    env_action.triggered.connect(_open_env_manager)

    napari.run()


if __name__ == "__main__":
    main()