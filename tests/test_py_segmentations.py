from AdaptFM.segmentation.registry import SEGMENTATION_REGISTRY
import numpy as np
import unittest

# tests for the non NN based algorithms returned via:
# print(SEGMENTATION_REGISTRY.names())
# not 'SAM2 Click & Propagate', 'SAM3 Text-Based',

# The volume data given to the segmentation functions comes from napari's viewer.layers[<image_layer>].data
# here assumes the data is volume Z, Y, X, is a numpy array (it's sometimes a dask array but computed before given to the run function in that case)
# and is of time uint8 for these test cases
def get_vol():
    Z_dim, Y_dim, X_dim = 3, 32, 32
    return np.random.randint(0, 255, size=(Z_dim, Y_dim, X_dim), dtype=np.uint8)

def test_runner_return_result(seg_registry_name):
    instance = SEGMENTATION_REGISTRY.get(seg_registry_name)

    # we use the defaults for testing the segmentation's run function
    params = {k: v["default"] for k, v in instance.tunable_params().items()}
        
    return instance.run(get_vol(), params)


class TestNucLogGabor(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("NucLogGabor")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestOtsuThreshold3D(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("OtsuThreshold3D")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestHighFreqSeg(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("HighFreqSeg")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestOrganoidSeg(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("OrganoidSeg")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestNucLogGaborGPU(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("NucLogGaborGPU")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestSauvolaThreshold3D(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("SauvolaThreshold3D")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestCannyEdge3D(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("CannyEdge3D")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))

class TestFelzenszwalb3D(unittest.TestCase):

    def test_will_run(self):
        result = test_runner_return_result("Felzenszwalb3D")
        self.assertEqual(result.shape, get_vol().shape)
        self.assertTrue(issubclass(result.dtype.type, (np.integer, np.bool_)))