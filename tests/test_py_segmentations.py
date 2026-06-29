from AdaptFM.segmentation.registry import SEGMENTATION_REGISTRY
import numpy as np
import unittest
# print(SEGMENTATION_REGISTRY.names())

def get_vol():
    return np.random.randint(0, 255, size=(3, 32, 32), dtype=np.uint8)


class TestNucLogGabor(unittest.TestCase):

    def test_will_run(self):
        instance = SEGMENTATION_REGISTRY.get('NucLogGabor')
        params = {k: v["default"] for k, v in instance.tunable_params().items()}
        
        result = instance.run(get_vol(), params)
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestOtsuThreshold3D(unittest.TestCase):

    def test_will_run(self):
        instance = SEGMENTATION_REGISTRY.get('OtsuThreshold3D')
        params = {k: v["default"] for k, v in instance.tunable_params().items()}
        
        result = instance.run(get_vol(), params)
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))



#(
#     NucLogGabor, FrequencySegmentation, OrganoidSegmentation, NucLogGaborGPU,
#     OtsuThreshold3D, SauvolaThreshold3D, CannyEdge3D, Felzenszwalb3D
# )

# Z_dim, Y_dim, X_dim
# MOCK_VOLUME = Z, Y, X


# class NucLogGabor(SegmentationAlgorithmSpec):
#     name = "NucLogGabor"

#     def tunable_params(self):
#         return {
#             "percentile": {"type": "float", "default": 90, "min": 0, "max": 100},
#             "max_freq": {"type": "float", "default": .01, "min": .001, "max": 1.0},
#             "frequency_step": {"type": "float", "default": .01, "min": 0, "max": 1.0},
#             "sigma": {"type": "float", "default": 0.3, "min": 0.1, "max": 100.0},
#             "remove_background": {"type": "bool", "default": True},
#         }

#     def run(self, volume, params):
#         return run_nuclear_segmentation(
#             volume,
#             percentile=params["percentile"],
#             max_freq=params["max_freq"],
#             frequency_step=params["frequency_step"],
#             sigma=params["sigma"],
#             remove_background=params["remove_background"],
#         )


# class FrequencySegmentation(SegmentationAlgorithmSpec):

#     name = "HighFreqSeg"

#     def tunable_params(self):
#         return {
#             "percentile": {"type": "float", "default": 90, "min": 0, "max": 10000.0},
#             "minimum size": {"type": "int", "default": 0, "min": 0},
#         }

#     def run(self, volume, params):
#         return run_single_cell_segmentation(
#             volume,
#             percentile=params["percentile"],
#             minimum_size=params['minimum size'],
#         )
    
# class OrganoidSegmentation(SegmentationAlgorithmSpec):

#     name ='OrganoidSeg'


#     def tunable_params(self):
#         return {
#             "minimum size": {"type": "int", "default": 0.0, "min": 0.0,"max":10000.0},
#             "sigma": {"type": "float", "default": 0.0, "min":0.0,"max":100.0},
#             "n_jobs": {'type': "int", "default": 1,"min":"-1","max": 256}
#         }

#     def run(self, volume, params):
#         return run_organoid_segmentation(
#             volume,
#             minimum_size=params["minimum size"],
#             sigma=params['sigma'],
#         )
    

    
# class NucLogGaborGPU(SegmentationAlgorithmSpec):
#     name = "NucLogGaborGPU"

#     def tunable_params(self):
#         return {
#             "percentile": {"type": "float", "default": 90, "min": 0, "max": 100},
#             "max_freq": {"type": "int", "default": 20, "min": 1, "max": 1000},
#             "frequency_step": {"type": "int", "default": 1, "min": 0, "max": 100},
#             "sigma": {"type": "float", "default": 0.3, "min": 0.1, "max": 100},
#             "remove_background": {"type": "bool", "default": True},
#             "chunk_size": {'type': "int", 'default': 5, "min":1, "max":1000},
#             "GPU": {'type': "int", 'default':0,'min':0,'max':100},

#         }

#     def run(self, volume, params):
#         return run_nuclear_segmentation_gpu_chunked(
#             volume,
#             percentile=params["percentile"],
#             max_freq=params["max_freq"],
#             frequency_step=params["frequency_step"],
#             sigma=params["sigma"],
#             remove_background=params["remove_background"],
#             chunk_size = params['chunk_size'],
#             gpu_id = params['GPU'],
#         )
    

# ###
# ## classical segmentation algorithms
# class OtsuThreshold3D(SegmentationAlgorithmSpec):
#     name = "OtsuThreshold3D"

#     def tunable_params(self):
#         return {
#             "remove_small_objects": {"type": "int", "default": 50, "min": 0, "max": 10000},
#         }

#     def run(self, volume, params):
#         from skimage.filters import threshold_otsu
#         from skimage.morphology import remove_small_objects

#         thresh = threshold_otsu(volume)
#         mask = volume > thresh
#         mask = remove_small_objects(mask, min_size=params["remove_small_objects"])
#         return mask.astype(int)


# class SauvolaThreshold3D(SegmentationAlgorithmSpec):
#     name = "SauvolaThreshold3D"

#     def tunable_params(self):
#         return {
#             "window_size": {"type": "int", "default": 25, "min": 3, "max": 101},
#             "k": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
#             "remove_small_objects": {"type": "int", "default": 50, "min": 0, "max": 10000},
#         }

#     def run(self, volume, params):
#         import numpy as np
#         from skimage.filters import threshold_sauvola
#         from skimage.morphology import remove_small_objects

#         mask = np.zeros_like(volume, dtype=bool)
#         for z in range(volume.shape[0]):
#             thresh = threshold_sauvola(volume[z], window_size=params["window_size"], k=params["k"])
#             mask[z] = volume[z] > thresh
#         mask = remove_small_objects(mask, min_size=params["remove_small_objects"])
#         return mask.astype(int)
    

    
# class CannyEdge3D(SegmentationAlgorithmSpec):
#     name = "CannyEdge3D"

#     def tunable_params(self):
#         return {
#             "sigma": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
#             "low_threshold": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
#             "high_threshold": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
#         }

#     def run(self, volume, params):
#         import numpy as np
#         from skimage.feature import canny

#         mask = np.zeros_like(volume, dtype=bool)
#         for z in range(volume.shape[0]):
#             mask[z] = canny(volume[z],
#                             sigma=params["sigma"],
#                             low_threshold=params["low_threshold"],
#                             high_threshold=params["high_threshold"])
#         return mask.astype(int)
    

# class Felzenszwalb3D(SegmentationAlgorithmSpec):
#     name = "Felzenszwalb3D"

#     def tunable_params(self):
#         return {
#             "scale": {"type": "float", "default": 100.0, "min": 10.0, "max": 1000.0},
#             "sigma": {"type": "float", "default": 0.5, "min": 0.0, "max": 5.0},
#             "min_size": {"type": "int", "default": 50, "min": 0, "max": 10000},
#         }

#     def run(self, volume, params):
#         import numpy as np
#         from skimage.segmentation import felzenszwalb
#         from skimage.morphology import remove_small_objects

#         labels = np.zeros_like(volume, dtype=int)
#         for z in range(volume.shape[0]):
#             slice_labels = felzenszwalb(volume[z], scale=params["scale"], sigma=params["sigma"])
#             labels[z] = slice_labels
#         labels = remove_small_objects(labels, min_size=params["min_size"])
#         return labels.astype(int)