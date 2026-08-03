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
from AdaptFM.model.registry import MODEL_REGISTRY
from qtpy.QtWidgets import QAction


def main():
    viewer = napari.Viewer()
    viewer.title ="AdaptFM"

    # Core managers
    vm = VolumeManager()
    sm = SegmentationManager(vm)
    session = AnnotationSession([])
    dm = DatasetManager()

    # Model registry (shared by training + inference)
    model_registry = MODEL_REGISTRY

    def qt_widget_obj_exists(dock_obj) -> bool:
        if dock_obj is None:
            return False
        try:
            dock_obj.objectName()
            return True
        except (RuntimeError, AttributeError):
            return False
    

    def create_sesh_dock():
        return viewer.window.add_dock_widget(
            SessionWidget(viewer, session, vm, sm).widget,
            area="right", name='AdaptFM Image Manager'
        )
    def create_seg_dock():
        return viewer.window.add_dock_widget(
            SegmentationWidget(viewer, sm).widget,
            area="right", name='AdaptFM Annotation'
        )
    def create_save_dock():
        return viewer.window.add_dock_widget(
            SaveWidget(viewer, sm).widget,
            area="right", name='AdaptFM Save Image'
        )
    
    save_dock = create_save_dock()
    seg_dock = create_seg_dock()
    sesh_dock = create_sesh_dock()
    
    side_docs = {
        create_sesh_dock: sesh_dock,
        create_seg_dock: seg_dock,
        create_save_dock: save_dock,
    }

    def restore_docks():
        for factory, dock in side_docs.items():
            if qt_widget_obj_exists(dock):
                dock.setVisible(True)
                dock.show()
                dock.raise_()
            else:
                side_docs[factory] = factory()

    restore_action = QAction("Restore AdaptFM Sidewidgets", viewer.window._qt_window)
    restore_action.triggered.connect(restore_docks)
    viewer.window.main_menu.addAction(restore_action)


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

    # ------------------------------------------------------------------ #
    # Environments menu  ← NEW
    # ------------------------------------------------------------------ #
    env_menu = viewer.window._qt_window.menuBar().addMenu("Environments")
    env_action = QAction("Manage Environments…", viewer.window._qt_window)
    env_menu.addAction(env_action)
 
    # Lazy-create: dialog is parented to the main window so it stays on top
    _env_dialog: list[EnvironmentManagerDialog] = []   # mutable cell
 
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
