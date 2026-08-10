from qtpy.QtWidgets import QWidget, QVBoxLayout
from magicgui import magicgui
from qtpy.QtWidgets import QFileDialog
from napari.layers import Image
from napari.layers.image._image_utils import guess_labels
import os
from glob import glob
from pathlib import Path
from AdaptFM.gui.napari_utils import update_or_create_image,  update_or_create_labels

class SessionWidget:
    def __init__(self, viewer, session, vm, sm):
        self.viewer = viewer
        self.session = session
        self.vm = vm
        self.sm = sm

        self._internal_layer_names = {"Original", "auto_seg"}

        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        self._build(layout)

        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)

    def _load_path(self, path):
        self._clear_auto_seg()
        img, _ = self.vm.load_image(path)

        if guess_labels(img)=="labels":
            label_name = Path(path).name
            update_or_create_labels(self.viewer,label_name,img)
            return

        update_or_create_image(
            self.viewer,
            "Original",
            img,
            colormap="gray",
        )
    def _on_layer_inserted(self, event):
        layer = event.value
        if not isinstance(layer, Image):
            return
        if layer.name in self._internal_layer_names:
            return  # created by our own update_or_create_image / auto-seg flow

        path = layer.source.path if layer.source is not None else None
        if not path:
            return  # not backed by a readable file (e.g. pasted/generated array)

        # Sync VolumeManager to this file, same as the button path.
        self._clear_auto_seg()
        self._clear_original()
        self.vm.load_image(path)
        self.session.set_images([path])

        # Keep the existing "current image" convention working for any other
        # widget that looks up viewer.layers["Original"].
        layer.name = "Original"


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
        if "auto_seg" in self.viewer.layers:
            self.viewer.layers.remove("auto_seg")

    def _clear_original(self):
        if "Original" in self.viewer.layers:
            self.viewer.layers.remove("Original")