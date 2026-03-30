from qtpy.QtWidgets import QWidget, QVBoxLayout
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog
import os
from glob import glob
from gui_scripts.gui.napari_utils import update_or_create_image,  update_or_create_labels

class SessionWidget:
    def __init__(self, viewer, session, vm, sm):
        self.viewer = viewer
        self.session = session
        self.vm = vm
        self.sm = sm

        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        self._build(layout)

    def _load_path(self, path):
        self._clear_auto_seg()
        img, _ = self.vm.load_image(path)

        update_or_create_image(
            self.viewer,
            "Original",
            img,
            colormap="blue",
        )

    def _build(self, layout):
        @magicgui(call_button="Open image")
        def open_image():
            path, _ = QFileDialog.getOpenFileName(
                None,
                "Select image",
                ""
            )
            if not path:
                return

            self.session.set_images([path])
            self._load_path(path)

        @magicgui(call_button="Open folder")
        def open_folder():
            folder = QFileDialog.getExistingDirectory(
                None,
                "Select folder"
            )
            if not folder:
                return

            paths = sorted(glob(os.path.join(folder, "*.tif*")))
            if not paths:
                return

            self.session.set_images(paths)
            self._load_path(paths[0])

        @magicgui(call_button="Previous image")
        def prev_image():
            path = self.session.prev()
            if path:
                self._load_path(path)

        @magicgui(call_button="Next image")
        def next_image():
            path = self.session.next()
            if path:
                self._load_path(path)

        # add widgets to layout
        for w in (open_image, open_folder, prev_image, next_image):
            layout.addWidget(w.native)

    def _clear_auto_seg(self):
        if "auto_seg" in self.viewer.layers:
            self.viewer.layers.remove("auto_seg")

