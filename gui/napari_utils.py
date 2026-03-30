from itertools import product

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