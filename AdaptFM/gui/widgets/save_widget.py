from magicgui import magicgui
from pathlib import Path
from napari.layers import Labels

class SaveWidget:
    def __init__(self, viewer, segmentation_manager):
        self.viewer = viewer
        self.sm = segmentation_manager
        self._build_widget()

    def _build_widget(self):
        @magicgui(
            save_dir={"mode": "d", "label": "Save location"},
            filename_base={"tooltip": "(Optional) Custom filename"},
            segmentation_layer={"label": "Segmentation Layer"},
            call_button="Save",
            save_image={"label": "(Recommended) Save image alongside segmentation"}
        )
        def widget(
            save_dir=Path(".").resolve(),
            filename_base="",
            segmentation_layer: Labels = None,
            save_image: bool = True
        ):
            
            if segmentation_layer is None:
                print("Error: No segmentation layer selected.")
                return
            
            if len([
                layer for layer in self.viewer.layers 
                if layer.name.endswith("_AdaptFMseg")
            ]) == 0:
                raise RuntimeError("No segmentation layer found")

            manual_seg = self.viewer.layers["auto_seg"].data

            self.sm.save_for_training(
                save_dir=str(save_dir),
                filename_base=filename_base or None,
                manual_seg=segmentation_layer,
            )

        self.widget = widget
