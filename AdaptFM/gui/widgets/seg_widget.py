from magicgui import magicgui
from magicgui.widgets import Container, Label
from napari import Viewer
from AdaptFM.segmentation.registry import SEGMENTATION_REGISTRY
from magicgui import widgets, magicgui
import dask.array as da


class SegmentationWidget:
    def __init__(self, viewer,segmentation_manager):
        self.viewer = viewer
        self.current_algo = None
        self.run_button = None
        self.param_widgets = {}
        self._build_widget()


    def _build_widget(self):
        # Dropdown for selecting segmentation algorithm
        @magicgui(
            segmentation_algorithm={"choices": SEGMENTATION_REGISTRY.names()},
            call_button="Select Algorithm"
        )
        def algo_selector(segmentation_algorithm: str):
            self._on_algorithm_selected(segmentation_algorithm)

        self.algo_selector = algo_selector

        # Container for dynamically generated param widgets
        self.param_container = Container()
        self.param_container.native.hide()  # hide initially

        # Layout into a dock
        self.widget = self.algo_selector.native
        self.viewer.window.add_dock_widget(self.widget, area="right")

    def _on_algorithm_selected(self, algo_name: str):
        self.current_algo = SEGMENTATION_REGISTRY.get(algo_name)
        
        self._load_tunable_params()

    def _load_tunable_params(self):
        # clear old widgets
        for w in self.param_widgets.values():
            self.param_container.native.layout().removeWidget(w.native)
            w.native.deleteLater()
        self.param_widgets.clear()

        schema = self.current_algo.tunable_params()

        # dynamically create widgets for each param
        for name, spec in schema.items():
            w = self._make_param_widget(name, spec)
            self.param_widgets[name] = w
            self.param_container.native.layout().addWidget(w.native)

        self.param_container.native.show()
        from qtpy.QtCore import Qt

        w = self.param_container.native
        w.setWindowFlags(w.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Window)
        w.show()


        # Add "Run auto-seg" button dynamically
        if self.run_button is None:
            @magicgui(call_button="Run auto-segmentation")
            def run_button():
                if "Original" not in self.viewer.layers:
                    raise RuntimeError("No image loaded")


                params = {k: w.control.value for k, w in self.param_widgets.items()}
                volume = self.viewer.layers["Original"].data

                if isinstance(volume, da.Array):
                    print('converting')
                    volume = volume.compute()
                seg = self.current_algo.run(volume, params)

                if "auto_seg" in self.viewer.layers:
                    self.viewer.layers["auto_seg"].data = seg
                else:
                    self.viewer.add_labels(seg, name="auto_seg",
                                           colormap={1:'white',
                                                     None: 'white'})

            self.run_button = run_button
            self.param_container.native.layout().addWidget(run_button.native)


    def _make_param_widget(self, name: str, spec: dict):
        """
        Convert spec dict to a labeled magicgui widget.
        """
        param_type = spec.get("type", "float")
        default = spec.get("default")

        if param_type == "float":
            control = widgets.FloatSpinBox(
                value=float(default),
                min=spec.get("min", 0.0),
                max=spec.get("max", 1.0),
                step=spec.get("step", 0.01),
            )
        elif param_type == "int":
            control = widgets.SpinBox(
                value=int(default),
                min=int(spec.get("min", 0)),
                max=int(spec.get("max", 100)),
                step=int(spec.get("step", 1)),
            )
        elif param_type == "bool":
            control = widgets.CheckBox(
                value=bool(default),
            )
        else:
            control = widgets.LineEdit(
                value=str(default),
            )

        # Label + widget container
        labeled = Container(
            widgets=[
                Label(value=name),
                control,
            ],
            layout="horizontal",
        )

        # expose value cleanly
        labeled.value = lambda: control.value
        labeled.control = control

        return labeled
