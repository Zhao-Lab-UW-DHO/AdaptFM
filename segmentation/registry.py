
from typing import Dict
from gui_scripts.segmentation.Seg_Alg import SegmentationAlgorithmSpec,NucLogGabor,FrequencySegmentation,OrganoidSegmentation,NucLogGaborGPU,Felzenszwalb3D,CannyEdge3D,Watershed3D,SauvolaThreshold3D,OtsuThreshold3D


class SegmentationRegistry:
    def __init__(self):
        self._algorithms: Dict[str, SegmentationAlgorithmSpec] = {}

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
SEGMENTATION_REGISTRY.register(Watershed3D())
SEGMENTATION_REGISTRY.register(SauvolaThreshold3D())
SEGMENTATION_REGISTRY.register(OtsuThreshold3D())


    

