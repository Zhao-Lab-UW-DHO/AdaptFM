from magicgui import magicgui
from pathlib import Path

class SaveWidget:
    def __init__(self, viewer, segmentation_manager):
        self.viewer = viewer
        self.sm = segmentation_manager
        self._build_widget()

    def _build_widget(self):
        @magicgui(
            save_dir={"mode": "d"},
            filename_base={"tooltip": "Optional base name"},
            call_button="Save for training",
        )
        def widget(
            save_dir=Path(),
            filename_base="",
        ):
            if "auto_seg" not in self.viewer.layers:
                raise RuntimeError("No segmentation layer found")

            manual_seg = self.viewer.layers["auto_seg"].data

            self.sm.save_for_training(
                save_dir=str(save_dir),
                filename_base=filename_base or None,
                manual_seg=manual_seg,
            )

        self.widget = widget
