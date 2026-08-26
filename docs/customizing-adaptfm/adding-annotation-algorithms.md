# Adding New Annotation Algorithms to AdaptFM

In addition to the existing annotation algorithms, AdaptFM also allows users to add new ones. The steps are:

1. Navigate to AdaptFM > segmentation > Seg_Alg.py
2. Create a new class that inherits from the "SegmentationAlgorithmSpec" base class
3. Provide your class a 'name'. Define a 'tunable_params' method that creates a dictionary of any adjustable parameters, their type, and defaults.
4. Define a 'run' method that takes 'volume' and 'params' parameters. The method should contain the annotation algorithm itself.
5. Once complete, navigate to AdaptFM > segmentation > registry.py. Import your newly created class and register it with SEGMENTATION_REGISTRY
6. The algorithm should now appear in the 'segmentation algorithm' widget.

Example:

```python
class Watershed3D(SegmentationAlgorithmSpec):
    name = "Watershed3D"

    def tunable_params(self):
        return {
            "min_distance": {"type": "int", "default": 5, "min": 1, "max": 50},
            "remove_small_objects": {
                "type": "int",
                "default": 50,
                "min": 0,
                "max": 10000,
            },
        }

    def run(self, volume, params):
        import numpy as np
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        from skimage.morphology import remove_small_objects

        distance = ndi.distance_transform_edt(volume)
        local_maxi = peak_local_max(
            distance, min_distance=params["min_distance"], labels=volume
        )
        markers = np.zeros_like(volume, dtype=int)
        for i, coord in enumerate(local_maxi, 1):
            markers[tuple(coord)] = i

        labels = watershed(-distance, markers, mask=volume)
        labels = remove_small_objects(labels, max_size=params["remove_small_objects"])
        return labels.astype(int)
```

Once you have written your new class, navigate to AdaptFM > segmentation > registry.py and import your newly created class and register with SEGMENTATION_REGISTRY

```python

from AdaptFM.segmentation.Seg_Alg import (
    CannyEdge3D,
    Felzenszwalb3D,
    FrequencySegmentation,
    NucLogGabor,
    NucLogGaborGPU,
    OrganoidSegmentation,
    OtsuThreshold3D,
    SAM2ClickAndPropagate,
    SAM3TextAndPropagate,
    SauvolaThreshold3D,
    SegmentationAlgorithmSpec,
    Watershed3D # <- IMPORT THE NEW CLASS
)


class SegmentationRegistry:
    def __init__(self):
        self._algorithms: dict[str, SegmentationAlgorithmSpec] = {}

    def register(self, algo: SegmentationAlgorithmSpec):
        if algo.name in self._algorithms:
            raise ValueError(f"Segmentation algorithm '{algo.name}' already registered")
        self._algorithms[algo.name] = algo

    def get(self, name: str) -> SegmentationAlgorithmSpec:
        return self._algorithms[name]

    def names(self):
        return sorted(self._algorithms.keys())

    def items(self):
        return self._algorithms.items()


SEGMENTATION_REGISTRY = SegmentationRegistry()
SEGMENTATION_REGISTRY.register(NucLogGabor())
SEGMENTATION_REGISTRY.register(FrequencySegmentation())
SEGMENTATION_REGISTRY.register(OrganoidSegmentation())
SEGMENTATION_REGISTRY.register(NucLogGaborGPU())
SEGMENTATION_REGISTRY.register(Felzenszwalb3D())
SEGMENTATION_REGISTRY.register(CannyEdge3D())
SEGMENTATION_REGISTRY.register(SauvolaThreshold3D())
SEGMENTATION_REGISTRY.register(OtsuThreshold3D())
SEGMENTATION_REGISTRY.register(SAM2ClickAndPropagate())
SEGMENTATION_REGISTRY.register(SAM3TextAndPropagate())
SEGMETATION_REGISTRY.register(Watershed3D()) # <-- REGISTER HERE

```