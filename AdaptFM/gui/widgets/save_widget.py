from magicgui import magicgui
from pathlib import Path
from napari.layers import Labels
from napari.utils.notifications import (
    show_info,
    show_error,
)

class SaveWidget:
    def __init__(self, viewer, segmentation_manager):
        self.viewer = viewer
        self.sm = segmentation_manager
        self.viewer.layers.events.inserted.connect(self._setup_original_listener)
        self._build_widget()
        self._setup_original_listener()

    def _setup_original_listener(self, event=None):
        if "Original" in self.viewer.layers:
            layer = self.viewer.layers["Original"]
            layer.events.metadata.connect(self._update_filename)
            self._update_filename()

    def _update_filename(self, event=None):
        if "Original" in self.viewer.layers:
            base = self.viewer.layers["Original"].metadata.get("filename_base")
            if base:
                self.widget["filename_base"].value = base
        

    def _build_widget(self):
        @magicgui(
            save_dir={"mode": "d", "label": "Save location"},
            filename_base={"tooltip": "(Optional) Custom filename else volume_YYYYMMDD_HHMMSS", "label": "Filename"},
            segmentation_layer={"label": "Segmentation Layer"},
            call_button="Save",
            save_image={"label": "Save image alongside segmentation (Recommended)"}
        )
        def widget(
            save_dir=Path().resolve(),
            filename_base="",
            segmentation_layer: Labels = None,
            save_image: bool = True
        ):
            
            if segmentation_layer is None:
                show_info("Error: No segmentation layer selected.")
                return
            
            if len([
                layer for layer in self.viewer.layers 
                if layer.name.endswith("_AdaptFMseg")
            ]) == 0:
                raise RuntimeError("No segmentation layer found")

            try:
                _, seg_path = self.sm.save_for_training(
                                save_dir=str(save_dir),
                                filename_base=filename_base or None,
                                manual_seg=segmentation_layer.data,
                                save_image=save_image
                        )

                show_info(
                    f"Saved successfully!\nSegmentation: {Path(seg_path).name}"
                )

            except Exception as e:
                show_error(f"Save failed:\n{e}")
            
        self.widget = widget

        # Keep the shared manager in sync so other widgets (e.g. SegmentationWidget)
        # can read the current selection even after a dock restore recreates this widget
        widget.save_dir.changed.connect(self._sync_save_state)
        widget.filename_base.changed.connect(self._sync_save_state)
        widget.save_image.changed.connect(self._sync_save_state)

        self._sync_save_state()

    def _sync_save_state(self, *_):
        self.sm.selected_output_folder = str(self.widget.save_dir.value)
        self.sm.selected_filename_base = self.widget.filename_base.value
        self.sm.selected_save_image = self.widget.save_image.value

