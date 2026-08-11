from itertools import product
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QFrame

def update_or_create_image(viewer, name, data, layer_type="image", **kwargs):
    if name in viewer.layers:
        viewer.layers[name].data = data
    else:
        if layer_type == "image":
            viewer.add_image(data, name=name, **kwargs)
        elif layer_type == "labels":
            viewer.add_labels(data, name=name, **kwargs)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


def update_or_create_labels(
    viewer,
    name: str,
    data,
    *,
    colormap=None,
    opacity: float = 1.0,
    preserve_properties: bool = True,
):
    """
    Create or update a napari Labels layer.

    Parameters
    ----------
    viewer : napari.Viewer
    name : str
        Layer name
    data : np.ndarray
        Label image
    colormap : dict or None
        Label -> color mapping (e.g. {1: "white"})

    opacity : float
        Layer opacity
    preserve_properties : bool
        Preserve metadata (e.g. user-edited labels) when updating
    """

    # Ensure integer labels (napari requirement)
    if data.dtype.kind not in ("i", "u"):
        data = data.astype("uint16")

    if name in viewer.layers:
        layer = viewer.layers[name]

        # Preserve existing properties if requested
        old_props = layer.properties if preserve_properties else None

        layer.data = data

        if colormap is not None:
            layer.color = colormap

        layer.opacity = opacity

        if preserve_properties and old_props is not None:
            layer.properties = old_props

    else:
        viewer.add_labels(
            data,
            name=name,
            colormap=colormap,
            opacity=opacity,
        )



def expand_param_grid( user_values):
    keys = []
    values = []

    for k, v in user_values.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)
        else:
            keys.append(k)
            values.append([v])

    for combo in product(*values):
        yield dict(zip(keys, combo))


def qt_widget_obj_exists(dock_obj) -> bool:
    if dock_obj is None:
        return False
    try:
        dock_obj.objectName()
        return True
    except (RuntimeError, AttributeError):
        return False

# --- FIX 2: Enforce top-to-bottom placement using splitDockWidget ---
def reorder_docks(viewer,
                    widget_specs):
    """Force Session -> Annotation -> Save from top to bottom."""

    qt_window = viewer.window._qt_window

    # Get all existing docks in canonical order
    active_docks = []

    for spec in widget_specs:
        dock = spec["dock"]

        if qt_widget_obj_exists(dock):
            active_docks.append(dock)

    if not active_docks:
        return

    # Make sure all docks are visible and non-floating
    for dock in active_docks:
        dock.setFloating(False)
        dock.show()

    # Put every dock into the right dock area first.
    #
    # This is important: splitDockWidget() should operate on
    # docks that are already part of the Qt dock layout.
    for dock in active_docks:
        qt_window.addDockWidget(
            Qt.RightDockWidgetArea,
            dock
        )

    # Now explicitly construct the vertical hierarchy:
    #
    # Session
    #   |
    # Annotation
    #   |
    # Save
    #
    for i in range(1, len(active_docks)):
        qt_window.splitDockWidget(
            active_docks[i - 1],
            active_docks[i],
            Qt.Vertical
        )

    # Make sure the docks are actually visible
    for dock in active_docks:
        dock.show()



def highlight_dock(dock):
    """Temporarily draw a highlight directly over the dock contents."""

    if not qt_widget_obj_exists(dock):
        return

    dock.setFloating(False)
    dock.show()
    dock.setVisible(True)
    dock.raise_()

    target = dock.widget()

    if target is None:
        return

    # Make sure the target has been laid out before getting its geometry
    target.adjustSize()
    target.updateGeometry()

    # Remove an existing highlight if one exists
    old_frame = target.findChild(QFrame, "AdaptFM_HighlightFrame")
    if old_frame is not None:
        old_frame.deleteLater()

    # Create an overlay frame that lives DIRECTLY on the widget
    frame = QFrame(target)
    frame.setObjectName("AdaptFM_HighlightFrame")

    frame.setAttribute(Qt.WA_TransparentForMouseEvents, True)
    frame.setAttribute(Qt.WA_StyledBackground, False)

    # Transparent inside, cyan border
    frame.setStyleSheet("""
        QFrame#AdaptFM_HighlightFrame {
            background: transparent;
            border: 3px solid #00d2ff;
        }
    """)

    # Cover the actual widget, not its parent/container
    QTimer.singleShot(
        0,
        lambda: frame.setGeometry(target.rect())
    )
    frame.raise_()
    frame.show()
    frame.update()

    def remove_highlight():
        if qt_widget_obj_exists(target):
            try:
                frame.deleteLater()
            except RuntimeError:
                pass

    QTimer.singleShot(1200, remove_highlight)

# Master handler for restoring or focusing a specific widget
def restore_or_focus_widget(spec_idx: int,
                            widget_specs :dict,
                            viewer ):
    spec = widget_specs[spec_idx]
    dock = spec["dock"]

    if qt_widget_obj_exists(dock):
        dock.setFloating(False)
        dock.setVisible(True)
        dock.show()

    else:
        spec["dock"] = viewer.window.add_dock_widget(
            spec["create_fn"](),
            area="right",
            name=spec["name"]
        )

        dock = spec["dock"]

    reorder_docks(viewer,widget_specs)

    # Give Qt time to finish the dock layout
    QTimer.singleShot(
        100,
        lambda d=dock: highlight_dock(d)
    )

def restore_all_widgets(viewer,widget_specs):
    # Restore/recreate every widget
    for spec in widget_specs:
        dock = spec["dock"]

        if not qt_widget_obj_exists(dock):
            spec["dock"] = viewer.window.add_dock_widget(
                spec["create_fn"](),
                area="right",
                name=spec["name"]
            )
        else:
            dock.setFloating(False)
            dock.setVisible(True)
            dock.show()

    # Rebuild the canonical order
    reorder_docks(viewer,widget_specs)

    # Highlight all widgets
    for spec in widget_specs:
        dock = spec["dock"]

        if qt_widget_obj_exists(dock):
            highlight_dock(dock)

