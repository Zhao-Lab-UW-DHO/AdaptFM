from pathlib import Path

import dask.array as da
import numpy as np
import tifffile as tiff
from magicgui import magicgui, widgets
from magicgui.widgets import Container, Label
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QFileDialog, QMessageBox, QSizePolicy

from AdaptFM.gui.napari_utils import qt_widget_obj_exists
from AdaptFM.segmentation.registry import SEGMENTATION_REGISTRY
from AdaptFM.volume.volume_manager import VolumeManager


class SegmentationWidget:
    def __init__(self, viewer, segmentation_manager):
        self.viewer = viewer
        self.current_algo = None
        self.run_button = None
        self.run_folder_button = None
        self.volume_manager = None
        self.param_widgets = {}
        self.sm = segmentation_manager
        # SAM2 interactive state
        self._sam2_obj_id = 1
        self._sam2_labels_layer = None
        self._sam2_click_callback = None  # reference so we can disconnect it

        self._build_widget()

    def _build_widget(self):
        # Dropdown for selecting segmentation algorithm
        @magicgui(
            segmentation_algorithm={
                "choices": SEGMENTATION_REGISTRY.names(),
                "label": "Algorithm",
            },
            auto_call=True,
        )
        def algo_selector(segmentation_algorithm: str):
            self._on_algorithm_selected(segmentation_algorithm)

        algo_selector.label = (
            ""  # otherwise puts the text algo selector on the dropdown
        )
        self.algo_selector = algo_selector

        self.param_container = (
            Container()
        )  # Container for dynamically generated param widgets
        self.main_container = Container(
            widgets=[self.algo_selector, self.param_container]
        )  # bundle with the selector
        self.widget = self.main_container.native

        # run the param retrieval on startup
        self._on_algorithm_selected(self.algo_selector.segmentation_algorithm.value)

    def _on_algorithm_selected(self, algo_name: str):
        self._teardown_sam2()  # no-op if previous algo was batch

        self.current_algo = SEGMENTATION_REGISTRY.get(algo_name)

        if getattr(self.current_algo, "mode", "batch") == "interactive":
            self._build_interactive_panel()

            if self.run_button is not None:
                self.run_button.visible = False
                self.run_folder_button.visible = False
            self._load_tunable_params()
        else:
            if self.run_button is not None:
                self.run_button.visible = True
                self.run_folder_button.visible = True  # sadly, unhiding this puts it at the top of the parameter options. not sure how to fix
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

        # w = self.param_container.native
        # w.setWindowFlags(w.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Window)
        # w.show()

        # w.setWindowTitle("Annotation")
        # Add "Run auto-seg" button dynamically
        if self.run_button is None:

            @magicgui(call_button="Run auto-segmentation")
            def run_button():
                if "Original" not in self.viewer.layers:
                    raise RuntimeError("No image loaded")

                params = {k: w.control.value for k, w in self.param_widgets.items()}
                volume = self.viewer.layers["Original"].data
                algo = self.current_algo

                self._set_ui_enabled(False)

                @thread_worker
                def _run_in_thread(volume, algo, params):
                    if isinstance(volume, da.Array):
                        print("converting")
                        volume = volume.compute()
                    seg = algo.run(volume, params)
                    return seg

                def _on_success(seg):
                    self._set_ui_enabled(True)
                    if seg is None:
                        print(f"Warning: {algo.name} returned None.")
                        return

                    if algo.name in self.viewer.layers:
                        self.viewer.layers[algo.name].data = seg
                    else:
                        self.viewer.add_labels(
                            seg,
                            name=f"{algo.name}_AdaptFMseg",
                            colormap={1: "white", None: "white"},
                        )

                def _on_error(e):
                    self._set_ui_enabled(True)
                    print(f"Error running segmentation: {e}")

                worker = _run_in_thread(volume, algo, params)
                worker.returned.connect(_on_success)
                worker.errored.connect(_on_error)
                worker.start()

            self.run_button = run_button
            self.param_container.native.layout().addWidget(run_button.native)

        else:
            # Re-append to layout to guarantee it stays below dynamically added parameter widgets
            self.param_container.native.layout().removeWidget(self.run_button.native)
            self.param_container.native.layout().addWidget(self.run_button.native)

        if self.run_folder_button is None:

            @magicgui(call_button="Run auto-segmentation on folder")
            def run_folder_button():
                folder = QFileDialog.getExistingDirectory(
                    self.widget, "Select folder containing images", ""
                )
                if not folder:
                    show_info("No folder selected.")
                    return
                # Collect image files
                image_paths = sorted(
                    [
                        p
                        for p in Path(folder).iterdir()
                        if p.is_file()
                        and p.name.lower().endswith((".tif", ".tiff", ".nii.gz"))
                    ]
                )
                if not image_paths:
                    show_info("No images found in folder.")
                    return
                algo = self.current_algo
                params = {k: w.control.value for k, w in self.param_widgets.items()}

                # Use the save widget's chosen output location if one has been set;
                # otherwise fall back to the old behavior (subfolder inside input folder)
                out_dir = getattr(self.sm, "selected_output_folder", None)

                if out_dir:
                    out_dir = str(Path(out_dir).resolve())

                if not out_dir or not Path(out_dir).is_dir():
                    show_info("You must select an output folder in the AdaptFM Save Image dock.")
                    return
                save_images = getattr(self.sm, "selected_save_image", True)

                # Build the save-images message
                if save_images:
                    save_msg = "Original images will be saved with segmentations."
                else:
                    save_msg = "Original images will NOT be saved with segmentations."

                msg = (
                    f"Images will be processed from:\n"
                    f"{folder}\n\n"
                    f"and saved to:\n"
                    f"{out_dir}\n\n"
                    f"{save_msg}\n\n"
                    f"Would you like to proceed?\n"
                    "Select No and modify the AdaptFM Save Widget to change save settings."
                )

                reply = QMessageBox.question(
                    self.widget,
                    "Confirm Batch Processing",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                )

                if reply != QMessageBox.Yes:
                    show_info("Batch processing cancelled.")
                    return

                self._set_ui_enabled(False)
                self.volume_manager = VolumeManager()

                @thread_worker
                def _run_batch(paths, algo, params, out_dir):
                    results = []
                    Path(out_dir).mkdir(parents=True, exist_ok=True)

                    # Pull "save image alongside segmentation" from the save widget's
                    # last-synced value; default True to match the save widget's own default
                    # before the user has touched the checkbox.
                    save_image = getattr(self.sm, "selected_save_image", True)

                    def _strip_known_ext(filename):
                        for ext in (".nii.gz", ".tif", ".tiff"):
                            if filename.lower().endswith(ext):
                                return filename[: -len(ext)]
                        return Path(filename).stem

                    for path in paths:
                        try:
                            img, _ = self.volume_manager.load_image(path)
                            if isinstance(img, da.Array):
                                img = img.compute()
                            seg = algo.run(img, params)

                            orig_name = Path(path).name
                            base = _strip_known_ext(orig_name)
                            seg_name = f"{base}_seg.tiff"
                            seg_path = str(Path(out_dir) / seg_name)

                            try:
                                tiff.imwrite(seg_path, seg.astype(seg.dtype))
                            except Exception as e:
                                print(f"Failed to save {seg_path}: {e}")

                            if save_image:
                                img_name = f"{base}.tiff"
                                img_path = str(Path(out_dir) / img_name)
                                try:
                                    tiff.imwrite(img_path, img)
                                except Exception as e:
                                    print(f"Failed to save {img_path}: {e}")

                            results.append((path, seg))
                        except Exception as e:
                            results.append((path, None))
                            print(f"Error processing {path}: {e}")
                    return results

                def _on_success(results):
                    self._set_ui_enabled(True)
                    show_info("Batch segmentation complete!")
                    for path, seg in results:
                        if seg is None:
                            print(f"Failed: {path}")
                        else:
                            print(f"Success: {path}")
                    print("Done.")

                def _on_error(e):
                    self._set_ui_enabled(True)
                    print(f"Batch error: {e}")

                worker = _run_batch(image_paths, algo, params, out_dir)
                worker.returned.connect(_on_success)
                worker.errored.connect(_on_error)
                worker.start()

            self.run_folder_button = run_folder_button
            self.param_container.native.layout().addWidget(run_folder_button.native)
        else:
            self.param_container.native.layout().removeWidget(
                self.run_folder_button.native
            )
            self.param_container.native.layout().addWidget(
                self.run_folder_button.native
            )
            self.param_container.native.show()

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

        elif param_type == "choice":
            control = widgets.ComboBox(
                choices=spec.get("options", []),
                value=default,
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
        self._sam2_status.native.setWordWrap(
            True
        )  # attempt send text down rather than out (that autoresizes undesirably)
        self._sam2_status.native.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )  # more enforcement of no dock size change
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
            tooltip='e.g. "nucleus", "organoid", "cell membrane"',
        )

        self._sam3_text_input_row = Container(
            widgets=[
                Label(value="Text concept"),
                self._sam3_text_input,
            ],
            layout="horizontal",
        )

        self._sam3_text_obj_spinner = widgets.SpinBox(
            value=1,
            min=1,
            max=99,
        )

        self._sam3_text_obj_spinner_row = Container(
            widgets=[
                Label(value="Text obj ID"),
                self._sam3_text_obj_spinner,
            ],
            layout="horizontal",
        )

        @magicgui(call_button="Segment by text (current slice)")
        def text_prompt_btn():
            self._sam3_run_text_prompt()

        self._sam3_text_btn = text_prompt_btn

        for w in [
            self._sam3_text_input_row,
            self._sam3_text_obj_spinner_row,
            text_prompt_btn,
        ]:
            layout.addWidget(w.native)
            w.native.setVisible(is_sam3)

        # ── Click mode + object ID (shared) ──────────────────────────────
        self._sam2_obj_spinner = widgets.SpinBox(
            value=1,
            min=1,
            max=1000,
        )

        self._sam2_obj_spinner_row = Container(
            widgets=[
                Label(value="Click Object ID"),
                self._sam2_obj_spinner,
            ],
            layout="horizontal",
        )

        layout.addWidget(self._sam2_obj_spinner_row.native)

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
            self._sam2_status,
            init_btn,
            self._sam3_text_input,
            self._sam3_text_obj_spinner,
            text_prompt_btn,
            self._sam2_obj_spinner,
            self._sam2_obj_spinner_row,
            prop_btn,
            reset_obj_btn,
            reset_all_btn,
            self._sam2_click_active,
            self._sam3_text_obj_spinner_row,
            self._sam3_text_input_row,
        ]

    def _set_ui_enabled(self, enabled: bool):
        """While threading occurs, the user changing/running other elements (like algorithm selection
        tearing down SAM variables) should be prevented
        """
        self.algo_selector.enabled = enabled

        if self.run_button is not None:
            self.run_button.enabled = enabled
            self.run_folder_button.enabled = enabled

        sam_buttons = [
            "_sam2_init_btn",
            "_sam2_prop_btn",
            "_sam3_text_btn",
        ]

        for attr_name in sam_buttons:
            btn = getattr(self, attr_name, None)
            if qt_widget_obj_exists(btn):  # this function will return false on None
                btn.enabled = enabled

        for param_widget in self.param_widgets.values():
            param_widget.control.enabled = enabled

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
        z = self.viewer.dims.current_step[0]  # current slice in viewer
        params = {k: w.control.value for k, w in self.param_widgets.items()}

        self._sam2_status.value = f'Status: running text prompt "{text}" on slice {z}…'
        self.param_container.native.repaint()

        try:
            self.current_algo.add_text_prompt(
                z=z, text=text, obj_id=obj_id, params=params
            )
            self._sam2_refresh_labels()
            self._sam2_status.value = (
                f'Status: text prompt done\n— "{text}" --,\n obj {obj_id}, slice {z}.\n'
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

        self._sam2_status.value = "Status: encoding slices… (may take a moment)"
        self._set_ui_enabled(False)
        self.param_container.native.repaint()  # force UI refresh before blocking call

        @thread_worker
        def _threaded_init_worker(volume):
            if isinstance(volume, da.Array):
                volume = volume.compute()

            try:
                self.current_algo.initialize(volume, self._sam2_get_params())
                blank = self.current_algo.get_label_volume()
                return blank
            except Exception as e:
                self._sam2_status.value = f"Status: ERROR — {e}"
                return

        def _on_success(blank):
            self._set_ui_enabled(True)
            if (
                self._sam2_labels_layer is not None
                and self._sam2_labels_layer in self.viewer.layers
            ):
                self.viewer.layers.remove(self._sam2_labels_layer)

            self._sam2_labels_layer = self.viewer.add_labels(
                blank, name=f"{self.current_algo.name}_AdaptFMseg"
            )
            self._sam2_status.value = (
                f"Status: ready — {self.current_algo.n_slices} slices encoded.\n"
                "Left-click = add prompt, right-click = background"
            )
            self._sam2_connect_clicks()

        def _on_error(e):
            self._set_ui_enabled(True)
            self._sam2_status.value = f"Status: ERROR — {e}"
            print(f"Error running segmentation: {e}")

        worker = _threaded_init_worker(volume)
        worker.returned.connect(_on_success)
        worker.errored.connect(_on_error)
        worker.start()

    def _sam2_disconnect_clicks(self):
        if self._sam2_click_callback is not None:
            try:
                self.viewer.mouse_drag_callbacks.remove(self._sam2_click_callback)
            except ValueError:
                pass
            self._sam2_click_callback = None

    def _sam2_connect_clicks(self):
        self._sam2_disconnect_clicks()
        self._sam2_box_start = None

        BOX_MIN_DRAG_PX = 5  # tune this threshold as needed

        def on_drag(viewer, event):
            if not self._sam2_click_active.value:
                return
            if not self.current_algo.is_initialized:
                return
            if viewer.dims.ndisplay != 2:
                self._sam2_status.value = "Status: switch to 2D view to add prompts"
                return

            # Only start tracking on left or right click
            if event.button not in (1, 2):
                return

            coords = self.viewer.layers["Original"].world_to_data(event.position)
            if len(coords) < 3:
                return

            z0 = int(np.clip(round(coords[0]), 0, self.current_algo.n_slices - 1))
            r0 = int(coords[1])
            c0 = int(coords[2])

            # Right-click: handle immediately, no drag tracking needed
            if event.button == 2:
                obj_id = self._sam2_obj_spinner.value
                try:
                    self.current_algo.add_prompt(
                        z=z0,
                        x=c0,
                        y=r0,
                        label=0,
                        obj_id=obj_id,
                        params=self._sam2_get_params(),
                    )
                    self._sam2_refresh_labels()
                    self._sam2_status.value = (
                        f"Status: BG click — slice {z0}, obj {obj_id}, ({c0}, {r0})"
                    )
                except Exception as e:
                    self._sam2_status.value = f"Status: ERROR — {e}"
                return

            # ── Left click: yield to receive move/release events ──────────
            yield  # napari now streams subsequent events into this generator

            while event.type == "mouse_move":
                coords = self.viewer.layers["Original"].world_to_data(event.position)

                yield

            # mouse_release
            coords = self.viewer.layers["Original"].world_to_data(event.position)
            if len(coords) < 3:
                return

            z1 = int(np.clip(round(coords[0]), 0, self.current_algo.n_slices - 1))
            r1 = int(coords[1])
            c1 = int(coords[2])

            drag_dist = np.hypot(c1 - c0, r1 - r0)
            obj_id = self._sam2_obj_spinner.value

            try:
                if drag_dist < BOX_MIN_DRAG_PX:
                    label = 1
                    self.current_algo.add_prompt(
                        z=z0,
                        x=c0,
                        y=r0,
                        label=label,
                        obj_id=obj_id,
                        params=self._sam2_get_params(),
                    )
                    self._sam2_refresh_labels()
                    self._sam2_status.value = f"Status: {'FG' if label else 'BG'} click — slice {z0}, obj {obj_id}, ({c0}, {r0})"
                    if label == 1:
                        self._sam2_obj_spinner.value = obj_id + 1
                else:
                    x_min, x_max = sorted([c0, c1])
                    y_min, y_max = sorted([r0, r1])
                    self.current_algo.add_box_prompt(
                        z=z0,
                        x0=x_min,
                        y0=y_min,
                        x1=x_max,
                        y1=y_max,
                        obj_id=obj_id,
                        params=self._sam2_get_params(),
                    )
                    self._sam2_refresh_labels()
                    self._sam2_status.value = f"Status: box — slice {z0}, obj {obj_id}, [{x_min},{y_min} → {x_max},{y_max}]"
                    self._sam2_obj_spinner.value = obj_id + 1
            except Exception as e:
                self._sam2_status.value = f"Status: ERROR — {e}"

        self._sam2_click_callback = on_drag
        self.viewer.mouse_drag_callbacks.append(on_drag)

    def _sam2_propagate(self):
        if not self.current_algo.is_initialized:
            self._sam2_status.value = "Status: initialise first"
            return

        params = self._sam2_get_params()
        direction = params.get("propagation_direction", "both")

        # No longer need obj_id — propagates everything at once
        self._sam2_status.value = f"Status: propagating all objects ({direction})…"
        self._set_ui_enabled(False)
        self.param_container.native.repaint()  # force UI refresh before blocking call

        @thread_worker
        def _threaded_propagate_worker(direction_val, params_val):
            try:
                self.current_algo.propagate(direction=direction_val, params=params_val)
                self.current_algo.reset_inference_state()
                return True
            except Exception as e:
                self._sam2_status.value = f"Status: ERROR — {e}"
                print(f"Error running propagation: {e}")
                return None

        def _on_success(result):
            self._set_ui_enabled(True)

            if result is not None:
                self._sam2_refresh_labels()
                self._sam2_status.value = "Status: propagation done"

        def _on_error(e):
            self._set_ui_enabled(True)
            self._sam2_status.value = f"Status: ERROR — {e}"
            print(f"Error running propagation: {e}")

        worker = _threaded_propagate_worker(direction, params)
        worker.returned.connect(_on_success)
        worker.errored.connect(_on_error)
        worker.start()

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

        self._sam2_init_btn = None
        self._sam2_prop_btn = None
        self._sam3_text_btn = None
