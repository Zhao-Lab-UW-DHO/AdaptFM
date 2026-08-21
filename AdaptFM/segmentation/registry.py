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
