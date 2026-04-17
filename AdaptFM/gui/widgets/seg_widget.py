from magicgui import magicgui
from magicgui.widgets import Container, Label
from napari import Viewer
from AdaptFM.segmentation.registry import SEGMENTATION_REGISTRY
from magicgui import widgets, magicgui
import dask.array as da
import numpy as np


class SegmentationWidget:
    def __init__(self, viewer,segmentation_manager):
        self.viewer = viewer
        self.current_algo = None
        self.run_button = None
        self.param_widgets = {}
        # SAM2 interactive state
        self._sam2_obj_id = 1
        self._sam2_labels_layer = None
        self._sam2_click_callback = None  # reference so we can disconnect it

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
        self._teardown_sam2()  # no-op if previous algo was batch

        self.current_algo = SEGMENTATION_REGISTRY.get(algo_name)
        
        if getattr(self.current_algo, "mode", "batch") == "interactive":
            self._build_interactive_panel()

            self._load_tunable_params()
        else:
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


    def _build_interactive_panel(self):
        layout = self.param_container.native.layout()

            #     # ── Segment click mode ON/OFF ──────────────────────────────
        self._sam2_click_active = widgets.CheckBox(
            value=False,
            label="Segment click mode",
        )
        layout.addWidget(self._sam2_click_active.native)

        self._sam2_status = widgets.Label(value="Status: not initialised")
        layout.addWidget(self._sam2_status.native)

        @magicgui(call_button="Initialise (encode all slices)")
        def init_btn():
            self._sam2_initialise()
        self._sam2_init_btn = init_btn
        layout.addWidget(init_btn.native)

        # ── Text prompt row  (SAM3 only — hidden for SAM2) ────────────────
        is_sam3 = getattr(self.current_algo, "supports_text_prompts", False)

        self._sam3_text_input = widgets.LineEdit(
            value="",
            label="Text concept",
            tooltip='e.g. "nucleus", "organoid", "cell membrane"',
        )
        self._sam3_text_obj_spinner = widgets.SpinBox(
            value=1, min=1, max=99, label="Text obj ID"
        )

        @magicgui(call_button="Segment by text (current slice)")
        def text_prompt_btn():
            self._sam3_run_text_prompt()
        self._sam3_text_btn = text_prompt_btn

        for w in [self._sam3_text_input, self._sam3_text_obj_spinner, text_prompt_btn]:
            layout.addWidget(w.native)
            w.native.setVisible(is_sam3)   # hide entirely for SAM2

        # ── Click mode + object ID (shared) ──────────────────────────────
        self._sam2_click_mode = widgets.ComboBox(
            choices=["Foreground", "Background"], value="Foreground", label="Click mode"
        )
        self._sam2_obj_spinner = widgets.SpinBox(
            value=1, min=1, max=1000, label="Click obj ID"
        )
        layout.addWidget(self._sam2_click_mode.native)
        layout.addWidget(self._sam2_obj_spinner.native)

        @magicgui(call_button="Propagate through volume")
        def prop_btn():
            self._sam2_propagate()
        self._sam2_prop_btn = prop_btn
        layout.addWidget(prop_btn.native)

        @magicgui(call_button="Reset current object")
        def reset_obj_btn():
            obj_id = self._sam2_obj_spinner.value
            self.current_algo.reset_object(obj_id)
            self._sam2_refresh_labels()
            self._sam2_status.value = f"Status: object {obj_id} cleared"

        @magicgui(call_button="Reset all")
        def reset_all_btn():
            self.current_algo.reset_all()
            self._sam2_refresh_labels()
            self._sam2_status.value = "Status: all objects cleared"

        self._sam2_reset_obj_btn = reset_obj_btn
        self._sam2_reset_all_btn = reset_all_btn
        layout.addWidget(reset_obj_btn.native)
        layout.addWidget(reset_all_btn.native)

        self._sam2_extra_widgets = [
            self._sam2_status, init_btn,
            self._sam3_text_input, self._sam3_text_obj_spinner, text_prompt_btn,
            self._sam2_click_mode, self._sam2_obj_spinner,
            prop_btn, reset_obj_btn, reset_all_btn,self._sam2_click_active
        ]


    def _sam3_run_text_prompt(self):
        """Called by the 'Segment by text' button."""
        if not self.current_algo.is_initialized:
            self._sam2_status.value = "Status: initialise first"
            return

        text = self._sam3_text_input.value.strip()
        if not text:
            self._sam2_status.value = "Status: enter a text prompt first"
            return

        obj_id = self._sam3_text_obj_spinner.value
        z      = self.viewer.dims.current_step[0]   # current slice in viewer
        params = {k: w.control.value for k, w in self.param_widgets.items()}

        self._sam2_status.value = f'Status: running text prompt "{text}" on slice {z}…'
        self.param_container.native.repaint()

        try:
            self.current_algo.add_text_prompt(
                z=z, text=text, obj_id=obj_id, params=params
            )
            self._sam2_refresh_labels()
            self._sam2_status.value = (
                f'Status: text prompt done — "{text}", obj {obj_id}, slice {z}. '
                "Now click Propagate."
            )
        except Exception as e:
            self._sam2_status.value = f"Status: ERROR — {e}"
    # ------------------------------------------------------------------
    # SAM2 actions
    # ------------------------------------------------------------------

    def _sam2_get_params(self) -> dict:
        return {k: w.control.value for k, w in self.param_widgets.items()}

    def _sam2_initialise(self):
        if "Original" not in self.viewer.layers:
            self._sam2_status.value = "Status: ERROR — no 'Original' layer found"
            return

        volume = self.viewer.layers["Original"].data
        if isinstance(volume, da.Array):
            volume = volume.compute()

        self._sam2_status.value = "Status: encoding slices… (may take a moment)"
        self.param_container.native.repaint()  # force UI refresh before blocking call

        try:
            self.current_algo.initialize(volume, self._sam2_get_params())
        except Exception as e:
            self._sam2_status.value = f"Status: ERROR — {e}"
            return

        # Create (or replace) the labels layer
        blank = self.current_algo.get_label_volume()
        if self._sam2_labels_layer is not None and self._sam2_labels_layer in self.viewer.layers:
            self.viewer.layers.remove(self._sam2_labels_layer)

        self._sam2_labels_layer = self.viewer.add_labels(blank, name="SAM2 labels")
        self._sam2_status.value = (
            f"Status: ready — {self.current_algo.n_slices} slices encoded. "
            "Left-click = add prompt, right-click = background"
        )

        # Wire up click callback on the labels layer
        self._sam2_connect_clicks()

    def _sam2_connect_clicks(self):
        """
        Intercept mouse clicks on the viewer canvas while SAM2 is active.
        We use the viewer's mouse_drag_callbacks so we get position in
        world coords regardless of which layer the user clicks on.
        """
        # Disconnect any previous callback first
        self._sam2_disconnect_clicks()

        def on_click(viewer, event):
            if not self._sam2_click_active.value:  # <-- skip if not active
                return
            if event.type != "mouse_press":
                return
            if not self.current_algo.is_initialized:
                return
            

            # Only act in 2-D display mode (slice view)
            if viewer.dims.ndisplay != 2:
                self._sam2_status.value = "Status: switch to 2D view to add prompts"
                return

            # World → data coordinates
            # viewer.dims.current_step gives (z, ...) for the current slice
            coords = self.viewer.layers["Original"].world_to_data(event.position)
            if len(coords) < 3:
                return

            z   = int(np.clip(round(coords[0]), 0, self.current_algo.n_slices - 1))
            row = int(coords[1])   # y in SAM2 terms
            col = int(coords[2])   # x in SAM2 terms

            # Left-click = foreground, right-click = background
            if event.button == 1:
                label = 1
            elif event.button == 2:
                label = 0
            else:
                return

            # Override with the toggle widget if the user prefers keyboard-free control
            if self._sam2_click_mode.value == "Background":
                label = 0

            obj_id = self._sam2_obj_spinner.value

            try:
                self.current_algo.add_prompt(
                    z=z, x=col, y=row,
                    label=label,
                    obj_id=obj_id,
                    params=self._sam2_get_params(),
                )
                self._sam2_refresh_labels()
                self._sam2_status.value = (
                    f"Status: prompt added — slice {z}, obj {obj_id}, "
                    f"{'FG' if label else 'BG'} ({col}, {row})"
                )

                if label == 1:
                    self._sam2_obj_spinner.value = obj_id + 1

            except Exception as e:
                self._sam2_status.value = f"Status: ERROR — {e}"

        self._sam2_click_callback = on_click
        self.viewer.mouse_drag_callbacks.append(on_click)

    def _sam2_disconnect_clicks(self):
        if self._sam2_click_callback is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._sam2_click_callback)
            except ValueError:
                pass
            self._sam2_click_callback = None


    def _sam2_propagate(self):
        if not self.current_algo.is_initialized:
            self._sam2_status.value = "Status: initialise first"
            return

        params    = self._sam2_get_params()
        direction = params.get("propagation_direction", "both")

        # No longer need obj_id — propagates everything at once
        self._sam2_status.value = f"Status: propagating all objects ({direction})…"
        self.param_container.native.repaint()

        try:

            self.current_algo.propagate(direction=direction, params=params)
            self.current_algo.reset_inference_state()

            self._sam2_refresh_labels()
            self._sam2_status.value = "Status: propagation done"
        except Exception as e:
            self._sam2_status.value = f"Status: ERROR — {e}"


    def _sam2_refresh_labels(self):
        """Push the updated label volume back to the napari layer."""
        if self._sam2_labels_layer is None:
            return
        self._sam2_labels_layer.data = self.current_algo.get_label_volume()
        self._sam2_labels_layer.refresh()

    def _teardown_sam2(self):
        """
        Called whenever the user switches away from a SAM2 algorithm.
        Disconnects click callbacks and removes the extra UI widgets.
        """
        self._sam2_disconnect_clicks()

        if hasattr(self, "_sam2_extra_widgets"):
            layout = self.param_container.native.layout()
            for w in self._sam2_extra_widgets:
                layout.removeWidget(w.native)
                w.native.deleteLater()
            self._sam2_extra_widgets = []


