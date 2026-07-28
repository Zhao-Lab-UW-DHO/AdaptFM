from pathlib import Path

from qtpy.QtWidgets import QWidget, QVBoxLayout
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog
import os
from glob import glob
from AdaptFM.gui.napari_utils import update_or_create_image,  update_or_create_labels
from qtpy.QtWidgets import QVBoxLayout, QSizePolicy

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
        self.widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _load_path(self, path):
        self._clear_auto_seg()
        img, _ = self.vm.load_image(path)

        layername = "Original"
        update_or_create_image(
            self.viewer,
            layername,
            img,
            colormap="gray",
        )
        self.viewer.layers[layername].metadata = {"filename_base": Path(path).stem}

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

        # add widgets to layout
        layout.addWidget(open_image.native)

    def _clear_auto_seg(self):
        layers_to_remove = [
            layer for layer in self.viewer.layers 
            if layer.name.endswith("_AdaptFMseg")
        ]
        for layer in layers_to_remove:
            self.viewer.layers.remove(layer)
