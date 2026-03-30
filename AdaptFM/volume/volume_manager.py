import os
import tifffile as tiff
import dask.array as da
import numpy as np
import SimpleITK as sitk

class VolumeManager:
    def __init__(
        self,
        chunk_size=(16, 256, 256),
        eager_threshold_mb=500,
        default_axes="ZYX",
        default_spacing=None,  # e.g. (z, y, x) in microns
    ):
        self.chunk_size = chunk_size
        self.eager_threshold_mb = eager_threshold_mb
        self.default_axes = default_axes
        self.default_spacing = default_spacing

        self.lazy_arr = None
        self.eager_arr = None
        self.mode = None
        self.path = None
        self.metadata = {}

    # -------------------------
    # Public API
    # -------------------------

    def load_image(self, path):
        self.path = path

        mb_est = os.path.getsize(path) / (1024 ** 2)

        if mb_est < self.eager_threshold_mb:
            self.mode = "eager"
            self.eager_arr = self._load_eager(path)
            self.lazy_arr = None
            arr = self.eager_arr
        else:
            self.mode = "lazy"
            self.lazy_arr = self._load_lazy(path)
            self.eager_arr = None
            arr = self.lazy_arr

        self.metadata = self._build_metadata(arr)

        return arr, self.metadata

    def get_array(self):
        return self.lazy_arr if self.mode == "lazy" else self.eager_arr

    def get_eager(self):
        """Return an eager NumPy array, computing from lazy if needed."""
        if self.eager_arr is not None:
            return self.eager_arr

        self.eager_arr = self.lazy_arr.compute()
        return self.eager_arr

    # -------------------------
    # Internal helpers
    # -------------------------


    def _load_eager(self, path):

        if path.endswith('.tiff'):
            return tiff.imread(path)
        elif path.endswith('.nii.gz'):
            return sitk.GetArrayFromImage(sitk.ReadImage(path))

    def _load_lazy(self, path):

        if path.endswith('.tiff'):
            zarr_store = tiff.imread(path, aszarr=True)
            return da.from_zarr(zarr_store, chunks=self.chunk_size)
        
        elif path.endswith('.nii.gz'):
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
            return da.from_array(arr, chunks=self.chunk_size)

    def _build_metadata(self, arr):
        """
        Build a napari- and ML-friendly metadata dictionary.
        """
        shape = arr.shape
        dtype = arr.dtype

        metadata = {
            "path": self.path,
            "mode": self.mode,
            "is_lazy": self.mode == "lazy",
            "shape": shape,
            "ndim": arr.ndim,
            "dtype": str(dtype),
            "axes": self.default_axes,
            "spacing": self.default_spacing,
        }

        return metadata


