from AdaptFM.segmentation.fourier.nuc_seg import run_nuclear_segmentation
from AdaptFM.segmentation.fourier.cell_seg import run_single_cell_segmentation
from AdaptFM.segmentation.fourier.org_seg import run_organoid_segmentation
from AdaptFM.segmentation.fourier.nuc_seg_gpu import run_nuclear_segmentation_gpu_chunked
from AdaptFM.segmentation.checkpoint_utils import check_sam2_installed 
from abc import ABC

class SegmentationAlgorithmSpec(ABC):
    name: str
    mode: str = "batch"  # "batch" | "interactive" — default keeps all existing algos unchanged

    def tunable_params(self) -> dict:
        """
        Return parameter schema:
        {
          "sigma": {"type": "float", "default": 0.3, "min": 0.1, "max": 2.0},
          "remove_background": {"type": "bool", "default": True},
        }
        """
        pass

    def run(self, volume, params: dict):
        """Run segmentation and return label image"""
        pass


class NucLogGabor(SegmentationAlgorithmSpec):
    name = "NucLogGabor"

    def tunable_params(self):
        return {
            "percentile": {"type": "float", "default": 90, "min": 0, "max": 100},
            "max_freq": {"type": "float", "default": .01, "min": .001, "max": 1.0},
            "frequency_step": {"type": "float", "default": .01, "min": 0, "max": 1.0},
            "sigma": {"type": "float", "default": 0.3, "min": 0.1, "max": 100.0},
            "remove_background": {"type": "bool", "default": True},
        }

    def run(self, volume, params):
        return run_nuclear_segmentation(
            volume,
            percentile=params["percentile"],
            max_freq=params["max_freq"],
            frequency_step=params["frequency_step"],
            sigma=params["sigma"],
            remove_background=params["remove_background"],
        )
    

class FrequencySegmentation(SegmentationAlgorithmSpec):

    name = "HighFreqSeg"

    def tunable_params(self):
        return {
            "percentile": {"type": "float", "default": 90, "min": 0, "max": 10000.0},
            "minimum size": {"type": "int", "default": 0, "min": 0},
        }

    def run(self, volume, params):
        return run_single_cell_segmentation(
            volume,
            percentile=params["percentile"],
            minimum_size=params['minimum size'],
        )
    
class OrganoidSegmentation(SegmentationAlgorithmSpec):

    name ='OrganoidSeg'


    def tunable_params(self):
        return {
            "minimum size": {"type": "int", "default": 0.0, "min": 0.0,"max":10000.0},
            "sigma": {"type": "float", "default": 0.0, "min":0.0,"max":100.0},
            "n_jobs": {'type': "int", "default": 1,"min":"-1","max": 256}
        }

    def run(self, volume, params):
        return run_organoid_segmentation(
            volume,
            minimum_size=params["minimum size"],
            sigma=params['sigma'],
        )
    

    
class NucLogGaborGPU(SegmentationAlgorithmSpec):
    name = "NucLogGaborGPU"

    def tunable_params(self):
        return {
            "percentile": {"type": "float", "default": 90, "min": 0, "max": 100},
            "max_freq": {"type": "int", "default": 20, "min": 1, "max": 1000},
            "frequency_step": {"type": "int", "default": 1, "min": 0, "max": 100},
            "sigma": {"type": "float", "default": 0.3, "min": 0.1, "max": 100},
            "remove_background": {"type": "bool", "default": True},
            "chunk_size": {'type': "int", 'default': 5, "min":1, "max":1000},
            "GPU": {'type': "int", 'default':0,'min':0,'max':100},

        }

    def run(self, volume, params):
        return run_nuclear_segmentation_gpu_chunked(
            volume,
            percentile=params["percentile"],
            max_freq=params["max_freq"],
            frequency_step=params["frequency_step"],
            sigma=params["sigma"],
            remove_background=params["remove_background"],
            chunk_size = params['chunk_size'],
            gpu_id = params['GPU'],
        )
    

###
## classical segmentation algorithms
class OtsuThreshold3D(SegmentationAlgorithmSpec):
    name = "OtsuThreshold3D"

    def tunable_params(self):
        return {
            "remove_small_objects": {"type": "int", "default": 50, "min": 0, "max": 10000},
        }

    def run(self, volume, params):
        from skimage.filters import threshold_otsu
        from skimage.morphology import remove_small_objects

        thresh = threshold_otsu(volume)
        mask = volume > thresh
        mask = remove_small_objects(mask, min_size=params["remove_small_objects"])
        return mask.astype(int)


class SauvolaThreshold3D(SegmentationAlgorithmSpec):
    name = "SauvolaThreshold3D"

    def tunable_params(self):
        return {
            "window_size": {"type": "int", "default": 25, "min": 3, "max": 101},
            "k": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
            "remove_small_objects": {"type": "int", "default": 50, "min": 0, "max": 10000},
        }

    def run(self, volume, params):
        import numpy as np
        from skimage.filters import threshold_sauvola
        from skimage.morphology import remove_small_objects

        mask = np.zeros_like(volume, dtype=bool)
        for z in range(volume.shape[0]):
            thresh = threshold_sauvola(volume[z], window_size=params["window_size"], k=params["k"])
            mask[z] = volume[z] > thresh
        mask = remove_small_objects(mask, min_size=params["remove_small_objects"])
        return mask.astype(int)
    

    
class CannyEdge3D(SegmentationAlgorithmSpec):
    name = "CannyEdge3D"

    def tunable_params(self):
        return {
            "sigma": {"type": "float", "default": 1.0, "min": 0.1, "max": 10.0},
            "low_threshold": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0},
            "high_threshold": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
        }

    def run(self, volume, params):
        import numpy as np
        from skimage.feature import canny

        mask = np.zeros_like(volume, dtype=bool)
        for z in range(volume.shape[0]):
            mask[z] = canny(volume[z],
                            sigma=params["sigma"],
                            low_threshold=params["low_threshold"],
                            high_threshold=params["high_threshold"])
        return mask.astype(int)
    

class Felzenszwalb3D(SegmentationAlgorithmSpec):
    name = "Felzenszwalb3D"

    def tunable_params(self):
        return {
            "scale": {"type": "float", "default": 100.0, "min": 10.0, "max": 1000.0},
            "sigma": {"type": "float", "default": 0.5, "min": 0.0, "max": 5.0},
            "min_size": {"type": "int", "default": 50, "min": 0, "max": 10000},
        }

    def run(self, volume, params):
        import numpy as np
        from skimage.segmentation import felzenszwalb
        from skimage.morphology import remove_small_objects

        labels = np.zeros_like(volume, dtype=int)
        for z in range(volume.shape[0]):
            slice_labels = felzenszwalb(volume[z], scale=params["scale"], sigma=params["sigma"])
            labels[z] = slice_labels
        labels = remove_small_objects(labels, min_size=params["min_size"])
        return labels.astype(int)



from abc import ABC
from typing import Optional
import torch
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Assume your base class is importable like this
# ---------------------------------------------------------------------------
# from your_module import SegmentationAlgorithmSpec


class SAM2ClickAndPropagate(SegmentationAlgorithmSpec):
    """
    SAM2-based interactive segmentation for 3D volumes in napari.

    Workflow:
      1. Call `initialize(volume)` when a new volume is loaded.
         This encodes all Z-slices upfront (expensive, but amortized).
      2. User clicks → call `add_prompt(z, x, y, label, obj_id)`.
         Returns an updated mask for that slice immediately.
      3. User clicks "Propagate" → call `propagate(obj_id, direction)`.
         Fills the full label volume forward/backward from the seed slices.
      4. Call `get_label_volume()` to retrieve the current full label array.
      5. Call `reset_object(obj_id)` or `reset_all()` to start over.

    The `run(volume, params)` method is also implemented for registry
    compatibility — it runs fully automatic propagation from the center
    slice with no user prompts (useful for batch pipelines / testing).
    """

    name = "SAM2 Click & Propagate"
    mode: str = "interactive"  # "batch" | "interactive" — default keeps all existing algos unchanged


    # ------------------------------------------------------------------
    # SegmentationAlgorithmSpec interface
    # ------------------------------------------------------------------

    def tunable_params(self) -> dict:
        return {
            "model_size": {
                "type": "choice",
                "default": "tiny",
                "options": ["tiny", "small", "base_plus", "large"],
                "description": "SAM2 model variant (larger = slower but more accurate)",
            },
            "propagation_direction": {
                "type": "choice",
                "default": "both",
                "options": ["forward", "backward", "both"],
                "description": "Direction to propagate from seed slices",
            },
            "auto_seed_slice": {
                "type": "choice",
                "default": "center",
                "options": ["center", "first", "last"],
                "description": "Which slice to auto-click in batch `run()` mode",
            },
            "score_threshold": {
                "type": "float",
                "default": 0.0,
                "min": -5.0,
                "max": 5.0,
                "description": "Logit threshold for mask binarisation (0 = sigmoid 0.5)",
            },
            "multimask_output": {
                "type": "bool",
                "default": False,
                "description": "Use SAM2 multimask mode (picks highest-score mask)",
            },
            "GPU": {'type': "int", 'default':0,'min':0,'max':100},

        }

    def run(self, volume: np.ndarray, params: dict) -> np.ndarray:
        """
        Batch-compatible entry point required by the registry.

        Initialises the volume, places a single positive point at the
        geometric centre of the chosen seed slice, then propagates in
        the requested direction(s).  Returns the full label volume.

        For real interactive use, call initialize() + add_prompt() +
        propagate() directly from your napari widget.
        """
        self.initialize(volume, params)

        direction = params.get("propagation_direction", "both")
        auto_slice = params.get("auto_seed_slice", "center")

        z_max = volume.shape[0] - 1
        seed_z = {
            "center": z_max // 2,
            "first":  0,
            "last":   z_max,
        }[auto_slice]

        cy = volume.shape[1] // 2
        cx = volume.shape[2] // 2

        self.add_prompt(z=seed_z, x=cx, y=cy, label=1, obj_id=1, params=params)
        self.propagate(obj_id=1, direction=direction, params=params)

        return self.get_label_volume()

    # ------------------------------------------------------------------
    # Interactive API (called from napari widget)
    # ------------------------------------------------------------------

    def __init__(self):
        self._predictor   = None   # sam2 VideoPredictor
        self._inf_state   = None   # opaque state dict managed by sam2
        self._label_vol   = None   # np.ndarray  (Z, Y, X)  int32
        self._volume      = None   # original volume, kept for re-init
        self._frames_dir  = None   # tempdir holding per-slice PNGs
        self._n_slices    = 0
        self._initialized = False
        self._model_size  = None   # track so we can lazy-reload

    # ------------------------------------------------------------------ #
    #  Step 1 — initialize                                                 #
    # ------------------------------------------------------------------ #

    def initialize(self, volume: np.ndarray, params: Optional[dict] = None) -> None:
        """
        Encode every Z-slice with the SAM2 image encoder.

        Args:
            volume: 3-D array  (Z, Y, X),  uint8 or float (auto-normalised).
            params: dict from tunable_params(); falls back to defaults if None.
        """
        check_sam2_installed() #verify installation first
        import sam2
        
        REPO_ROOT = Path(__file__).resolve().parents[1]  # AdaptFM/AdaptFM
        ckpt_dir = REPO_ROOT / "segmentation" / "sam2" / "checkpoints"

        os.environ["SAM2_REPO_ROOT"] = os.path.join(REPO_ROOT,'segmentation/sam2/sam2')

        os.environ["SAM2_CHECKPOINT_DIR"] = str(ckpt_dir)
#note that if this is their first time using, SAM2 checkpoints are downloaded
        params = params or {}
        model_size = params.get("model_size", "tiny")
        gpu = str(params.get('GPU'))


        self._volume = volume
        self._n_slices = volume.shape[0]
        self._label_vol = np.zeros(volume.shape, dtype=np.int32)

        self._load_predictor(model_size,gpu)
        self._write_frames_to_disk(volume)

        # init_state encodes every frame; this is the expensive call
        self._inf_state = self._predictor.init_state(
            video_path=str(self._frames_dir)
        )
        self._initialized = True

    # ------------------------------------------------------------------ #
    #  Step 2 — add_prompt                                                 #
    # ------------------------------------------------------------------ #

    def add_prompt(
        self,
        z: int,
        x: int,
        y: int,
        label: int = 1,
        obj_id: int = 1,
        params: Optional[dict] = None,
        extra_points: Optional[list] = None,
    ) -> np.ndarray:
        """
        Add a point prompt on slice `z` and return the predicted mask
        for that slice immediately.

        Args:
            z:            Z-index of the slice being clicked.
            x, y:         Pixel coordinates within the slice (SAM2 convention).
            label:        1 = foreground, 0 = background.
            obj_id:       Integer object ID (increment for each new object).
            params:       tunable_params dict.
            extra_points: Optional list of (x, y, label) for multi-point prompts
                          on the same slice — passed in a single SAM2 call.

        Returns:
            Binary mask for slice `z`, shape (H, W), dtype bool.
        """
        self._assert_initialized()
        params = params or {}
        threshold = params.get("score_threshold", 0.0)
        multimask = params.get("multimask_output", False)

        # Build point arrays
        all_pts    = [(x, y)]
        all_labels = [label]
        if extra_points:
            for ex, ey, el in extra_points:
                all_pts.append((ex, ey))
                all_labels.append(el)

        points_arr = np.array(all_pts,    dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int32)

        _, out_obj_ids, out_logits = self._predictor.add_new_points_or_box(
            inference_state=self._inf_state,
            frame_idx=z,
            obj_id=obj_id,
            points=points_arr,
            labels=labels_arr,
            clear_old_points=False,  # accumulate within a session
        )

        mask = self._logits_to_mask(out_logits, obj_id, out_obj_ids, threshold, multimask)

        # Write into label volume immediately so the viewer updates
        self._label_vol[z] = np.where(mask, obj_id, self._label_vol[z])

        return mask

    def add_box_prompt(
        self,
        z: int,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        obj_id: int = 1,
        params: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Add a bounding-box prompt on slice `z`.
        box = [x0, y0, x1, y1]  (top-left, bottom-right).
        """
        self._assert_initialized()
        params    = params or {}
        threshold = params.get("score_threshold", 0.0)
        multimask = params.get("multimask_output", False)

        box = np.array([x0, y0, x1, y1], dtype=np.float32)

        _, out_obj_ids, out_logits = self._predictor.add_new_points_or_box(
            inference_state=self._inf_state,
            frame_idx=z,
            obj_id=obj_id,
            box=box,
        )

        mask = self._logits_to_mask(out_logits, obj_id, out_obj_ids, threshold, multimask)
        self._label_vol[z] = np.where(mask, obj_id, self._label_vol[z])
        return mask

    # ------------------------------------------------------------------ #
    #  Step 3 — propagate                                                  #
    # ------------------------------------------------------------------ #


    def propagate(
        self,
        obj_id: int = 1,
        direction: str = "both",
        params: Optional[dict] = None,
    ) -> np.ndarray:
        self._assert_initialized()
        params    = params or {}
        threshold = params.get("score_threshold", 0.0)
        multimask = params.get("multimask_output", False)

        # SAM2 propagates all prompted objects in one pass internally.
        # Trying to propagate one object at a time causes memory state
        # inconsistencies. Always do a full propagation and collect all objects.
        if direction in ("forward", "both"):
            self._run_propagation(reverse=False, threshold=threshold, multimask=multimask)

        if direction in ("backward", "both"):
            self._run_propagation(reverse=True, threshold=threshold, multimask=multimask)

        return self._label_vol.copy()

    def _run_propagation(self, reverse: bool, threshold: float, multimask: bool) -> None:
        """Propagate ALL prompted objects in one pass."""
        for frame_idx, out_obj_ids, out_logits in self._predictor.propagate_in_video(
            self._inf_state,
            reverse=reverse,
        ):
            # Write every object that comes back, not just the requested one
            for obj_id in out_obj_ids:
                try:
                    mask = self._logits_to_mask(
                        out_logits, obj_id, out_obj_ids, threshold, multimask
                    )
                except Exception:
                    continue

                existing = self._label_vol[frame_idx]
                self._label_vol[frame_idx] = np.where(
                    mask & (existing == 0),
                    obj_id,
                    existing,
                )

    # ------------------------------------------------------------------ #
    #  Utility / query methods                                             #
    # ------------------------------------------------------------------ #

    def get_label_volume(self) -> np.ndarray:
        """Return a copy of the current label volume (Z, Y, X)."""
        self._assert_initialized()
        return self._label_vol.copy()

    def reset_object(self, obj_id: int) -> None:
        """
        Remove all predictions for one object and clear its SAM2 prompts.
        Resets the inference state so propagation can be re-run cleanly.
        """
        self._assert_initialized()
        self._label_vol[self._label_vol == obj_id] = 0
        # SAM2 has no partial-reset; cheapest correct option is full reinit
        self._inf_state = self._predictor.init_state(
            video_path=str(self._frames_dir)
        )

    def reset_all(self) -> None:
        """Wipe every prediction and reset SAM2 state."""
        self._assert_initialized()
        self._label_vol[:] = 0
        self._inf_state = self._predictor.init_state(
            video_path=str(self._frames_dir)
        )

    def reset_inference_state(self) -> None:
        self._inf_state= self._predictor.init_state(video_path = str(self._frames_dir))

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def n_slices(self) -> int:
        return self._n_slices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_predictor(self, model_size: str,gpu:int) -> None:
        """Lazy-load the SAM2 VideoPredictor; skip if already loaded."""
        if self._predictor is not None and self._model_size == model_size:
            return

        # Inline import so the class is importable even without SAM2 installed
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise ImportError(
                "SAM2 is not installed. "
                "Run: pip install git+https://github.com/facebookresearch/sam2.git"
            ) from e

        cfg_map = {
            "tiny":      ("sam2.1_hiera_t.yaml",   "sam2.1_hiera_tiny.pt"),
            "small":     ("sam2.1_hiera_s.yaml",   "sam2.1_hiera_small.pt"),
            "base_plus": ("sam2.1_hiera_b+.yaml",  "sam2.1_hiera_base_plus.pt"),
            "large":     ("sam2.1_hiera_l.yaml",   "sam2.1_hiera_large.pt"),
        }
        if model_size not in cfg_map:
            raise ValueError(f"Unknown model_size '{model_size}'. "
                             f"Choose from {list(cfg_map)}")

        cfg_file, ckpt_file = cfg_map[model_size]

        # Resolve checkpoint — honour env var, otherwise expect it next to this file
        ckpt_dir  = Path(os.environ.get("SAM2_CHECKPOINT_DIR"))
        ckpt_path = ckpt_dir / ckpt_file

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found at {ckpt_path}.\n"
                f"Download it from https://github.com/facebookresearch/sam2#model-description\n"
                f"or set the SAM2_CHECKPOINT_DIR environment variable."
            )

        device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

        cfg_file = os.path.join("configs/sam2.1",cfg_file)

        self._predictor  = build_sam2_video_predictor(cfg_file, str(ckpt_path), device=device)
        self._model_size = model_size

    def _write_frames_to_disk(self, volume: np.ndarray) -> None:
        """
        SAM2's VideoPredictor.init_state() expects a directory of
        lexicographically sorted JPEG/PNG files, one per frame.

        We write zero-padded PNGs into a persistent tempdir.
        Converts float volumes to uint8 via min-max normalisation.
        Handles grayscale → RGB conversion (SAM2 expects 3-channel input).
        """
        # Clean up any previous temp dir
        if self._frames_dir is not None:
            import shutil
            shutil.rmtree(self._frames_dir, ignore_errors=True)

        tmp = tempfile.mkdtemp(prefix="sam2_napari_")
        self._frames_dir = Path(tmp)

        # Normalise to uint8
        vol = volume.astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin) * 255.0
        vol_u8 = vol.astype(np.uint8)

        n_digits = len(str(volume.shape[0] - 1))

        for z in range(volume.shape[0]):
            slice_2d = vol_u8[z]

            # Ensure 3-channel RGB
            if slice_2d.ndim == 2:
                rgb = np.stack([slice_2d] * 3, axis=-1)
            elif slice_2d.shape[2] == 1:
                rgb = np.concatenate([slice_2d] * 3, axis=-1)
            else:
                rgb = slice_2d[:, :, :3]

            fname = self._frames_dir / f"{str(z).zfill(n_digits)}.jpg"
            Image.fromarray(rgb).save(fname)

    @staticmethod
    def _logits_to_mask(
        out_logits,
        target_obj_id: int,
        out_obj_ids: list,
        threshold: float,
        multimask: bool,
    ) -> np.ndarray:
        """
        Convert SAM2 logit output to a boolean 2-D mask.

        out_logits shape: (n_objects, n_masks, H, W)
        When multimask=False, n_masks=1.
        When multimask=True,  n_masks=3; pick the highest-confidence one.
        """
        obj_idx = list(out_obj_ids).index(target_obj_id)
        logits  = out_logits[obj_idx]           # (n_masks, H, W)

        if multimask and logits.shape[0] > 1:
            # Pick the mask whose max logit is highest
            best = int(logits.amax(dim=(-2, -1)).argmax())
            logits = logits[best : best + 1]

        mask = (logits[0] > threshold).cpu().numpy()   # (H, W)  bool
        return mask

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "SAM2ClickAndPropagate is not initialised. "
                "Call .initialize(volume) first."
            )

    def __del__(self):
        """Clean up temp frame directory on garbage collection."""
        if self._frames_dir is not None:
            import shutil
            shutil.rmtree(self._frames_dir, ignore_errors=True)


import numpy as np
import tempfile
import os
import torch
from pathlib import Path
from typing import Optional
from PIL import Image

# from your_module import SegmentationAlgorithmSpec


class SAM3TextAndPropagate(SegmentationAlgorithmSpec):
    """
    SAM3-based interactive segmentation for 3D volumes.

    Supports two prompt modes that can be mixed freely:
      - Click prompts  (left-click = foreground, right-click = background)
      - Text prompts   ("nucleus", "cell membrane", "organoid boundary", etc.)
                       Text is fed to SAM3's concept detector on the seed slice,
                       then the resulting mask is propagated via the tracker.

    Workflow:
      1. initialize(volume)        — encode all slices, open session
      2. add_text_prompt(...)      — run concept detector on one slice
         and/or add_prompt(...)    — add point prompt on one slice
      3. propagate(...)            — propagate forward / backward / both
      4. get_label_volume()        — retrieve full Z×Y×X label array
      5. reset_object(obj_id)      — remove one object cleanly
         reset_all()               — wipe everything, keep session open
    """
    name = "SAM3 Text-Based"
    mode = "interactive"
    supports_text_prompts=True

    # ------------------------------------------------------------------
    # SegmentationAlgorithmSpec interface
    # ------------------------------------------------------------------

    def tunable_params(self) -> dict:
        return {
            "propagation_direction": {
                "type": "choice",
                "default": "both",
                "options": ["forward", "backward", "both"],
            },
            "score_threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Confidence threshold for mask binarisation",
            },
            "text_conf_threshold": {
                "type": "float",
                "default": 0.25,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05,
                "description": "Minimum detection score for text-prompted instances",
            },
            "auto_seed_slice": {
                "type": "choice",
                "default": "center",
                "options": ["center", "first", "last"],
                "description": "Seed slice used in batch run() mode",
            },
            
            "GPU": {'type': "int", 'default':0,'min':0,'max':100},

        }

    def run(self, volume: np.ndarray, params: dict) -> np.ndarray:
        """
        Batch-compatible entry point. Uses the centre-of-volume point as a
        single foreground click, then propagates. For real interactive use,
        call initialize() / add_prompt() / add_text_prompt() / propagate()
        directly from the widget.
        """
        self.initialize(volume, params)
        direction = params.get("propagation_direction", "both")
        auto_slice = params.get("auto_seed_slice", "center")
        z_max = volume.shape[0] - 1
        seed_z = {"center": z_max // 2, "first": 0, "last": z_max}[auto_slice]

        self.add_prompt(
            z=seed_z,
            x=volume.shape[2] // 2,
            y=volume.shape[1] // 2,
            label=1,
            obj_id=1,
            params=params,
        )
        self.propagate(obj_id=1, direction=direction, params=params)
        return self.get_label_volume()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self):
        self._predictor   = None   # Sam3VideoPredictor
        self._session_id  = None
        self._label_vol   = None   # (Z, Y, X) int32
        self._volume      = None
        self._frames_dir  = None
        self._n_slices    = 0
        self._initialized = False
        self._active_obj_ids = []


    def initialize(self, volume: np.ndarray, params: Optional[dict] = None) -> None:
        """
        Write slices to disk, load the SAM3 video predictor, open a session.
        All subsequent add_prompt / add_text_prompt calls share this session.
        """
        import sam3
        REPO_ROOT = Path(__file__).resolve().parents[1]
        ckpt_dir = REPO_ROOT / 'segmentation'/'sam3'/'checkpoint'

        os.environ['SAM3_REPO_ROOT']=os.path.join(REPO_ROOT,'segmentation/sam3/sam3') # path to this repo

        os.environ['PYTHONPATH']="${SAM3_REPO_ROOT}:${PYTHONPATH}"
        os.environ['SAM3_CHECKPOINT_DIR']= str(ckpt_dir)
        params = params or {}
        self.gpu = params.get('GPU')

        self._volume   = volume
        self._n_slices = volume.shape[0]
        self._img_h = volume.shape[1]
        self._img_w = volume.shape[2]
        self._label_vol = np.zeros(volume.shape, dtype=np.int32)

        self._load_predictor()
        self._write_frames_to_disk(volume)

        # Open / replace session
        if self._session_id is not None:
            try:
                self._predictor.close_session(self._session_id)
            except Exception:
                pass

        response = self._predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=str(self._frames_dir),
            )
        )
        self._session_id = response["session_id"]
        self._initialized = True

    # ------------------------------------------------------------------
    # Point prompt (click-based)
    # ------------------------------------------------------------------


    def add_prompt(
        self,
        z: int,
        x: int,
        y: int,
        label: int = 1,
        obj_id: int = None,
        params: Optional[dict] = None,
        extra_points: Optional[list] = None,
    ) -> np.ndarray:

        self._assert_initialized()
        params = params or {}

        # Determine which object we are editing
        if obj_id is None:

            clicked_label = int(self._label_vol[z, y, x])

            # If background clicked → create new object
            if clicked_label == 0:
                obj_id = int(self._label_vol.max() + 1)
            else:
                obj_id = clicked_label

        # Build point list
        pts_abs = [[x, y]]
        labels_raw = [label]

        if extra_points:
            for ex, ey, el in extra_points:
                pts_abs.append([ex, ey])
                labels_raw.append(el)

        # Normalize coordinates
        pts_rel = [
            [px / self._img_w, py / self._img_h]
            for px, py in pts_abs
        ]

        device = torch.device("cuda:2")

        points_tensor = torch.tensor(
            pts_rel,
            dtype=torch.float32
        ).to(device)

        labels_tensor = torch.tensor(
            labels_raw,
            dtype=torch.int32
        ).to(device)

        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=z,
                points=points_tensor,
                point_labels=labels_tensor,
                obj_id=obj_id,
            )
        )

        result = response.get("outputs", response)

        instance_masks = self._result_to_mask(result, params)

        union_mask = np.zeros(
            (self._img_h, self._img_w),
            dtype=bool
        )

        if instance_masks is not None:

            sam_ids = result.get("out_obj_ids", [])

            for i, inst_mask in enumerate(instance_masks):

                oid = sam_ids[i]
                oid_val = (
                    oid.item()
                    if hasattr(oid, "item")
                    else int(oid)
                )

                inst_bool = inst_mask.astype(bool)

                # Update only this instance
                self._label_vol[z][inst_bool] = oid_val

                union_mask |= inst_bool

            self._active_obj_ids = [
                int(
                    oid.item()
                    if hasattr(oid, "item")
                    else oid
                )
                for oid in sam_ids
            ]

        return union_mask

    # ------------------------------------------------------------------
    # Text prompt  ← SAM3-specific
    # ------------------------------------------------------------------

    def add_text_prompt(
        self,
        z: int,
        text: str,
        obj_id: int = 1,
        params: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Run SAM3's concept detector on slice z with a text description.

        The detector may return multiple instances; all are merged into
        obj_id in the label volume, and the union mask is returned so the
        viewer immediately shows the result.

        After calling this, propagate() will track all detected instances
        forwards/backwards through the volume.

        Args:
            z:      Slice index to run detection on.
            text:   Open-vocabulary noun phrase, e.g. "nucleus", "organoid".
            obj_id: Object ID to assign to the detected instances.
            params: tunable_params dict.

        Returns:
            Union of all detected instance masks, shape (H, W), dtype bool.
        """
        self._assert_initialized()
        params = params or {}
        conf_thresh = params.get("text_conf_threshold", 0.25)

        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=z,
                obj_id=obj_id,
                text=text,
            )
        )
        result = response.get("outputs", response)  # handles both shapes defensively

        #######################
        detected_obj_ids = result.get("out_obj_ids", [])
        self._active_obj_ids = [
            oid.item() if hasattr(oid, "item") else int(oid)
            for oid in detected_obj_ids
        ]
        #######################


        # mask = self._result_to_mask(result, params, conf_threshold=conf_thresh)
        # if mask is not None:
        #     self._label_vol[z] = np.where(mask, obj_id, self._label_vol[z])
        # return mask if mask is not None else np.zeros(self._volume.shape[1:], dtype=bool)

        instance_masks = self._result_to_mask(
            result,
            params,
            conf_threshold=conf_thresh,
        )
        # Create empty union mask
        union_mask = np.zeros(
            self._volume.shape[1:],
            dtype=bool
        )
        sam_ids = result.get("out_obj_ids", [])

        if instance_masks is not None:

            for i, inst_mask in enumerate(instance_masks):

                oid = sam_ids[i]
                oid_val = oid.item() if hasattr(oid, "item") else int(oid)
                inst_bool = inst_mask.astype(bool)

                self._label_vol[z][inst_mask.astype(bool)] = oid_val
                
                union_mask |= inst_bool

        return union_mask

    def add_text_and_point_prompt(
        self,
        z: int,
        text: str,
        x: int,
        y: int,
        label: int = 1,
        obj_id: int = 1,
        params: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Combine a text concept prompt with a refinement point on the same slice.
        Text initialises the detection; the point refines which instance to keep.
        """
        self._assert_initialized()
        params = params or {}
        conf_thresh = params.get("text_conf_threshold", 0.25)

        points_tensor = torch.tensor(
            [[x / self._img_w, y / self._img_h]], dtype=torch.float32
        )
        labels_tensor = torch.tensor([label], dtype=torch.int32)


        response = self._predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=self._session_id,
                frame_index=z,
                obj_id=obj_id,
                text=text,
                points=points_tensor,
                point_labels=labels_tensor,
            )
        )
        result = response.get("outputs", response)  # handles both shapes defensively
        mask = self._result_to_mask(result, params, conf_threshold=conf_thresh)
        if mask is not None:
            self._label_vol[z] = np.where(mask, obj_id, self._label_vol[z])
        return mask if mask is not None else np.zeros(self._volume.shape[1:], dtype=bool)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------


    def propagate(self, obj_id=1, direction="both", params=None):
        self._assert_initialized()
        params = params or {}
        conf_thresh = params.get("score_threshold", 0.5)

        # Use all detected obj_ids if available, otherwise just the requested one
        obj_ids_to_propagate = self._active_obj_ids if self._active_obj_ids else [obj_id]
        print(f"Propagating obj_ids: {obj_ids_to_propagate}")

        for frame_output in self._predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=self._session_id,
                propagation_direction=direction,
            )
        ):
            frame_idx = frame_output["frame_index"]
            outputs   = frame_output.get("outputs", {})

            binary_masks = outputs.get("out_binary_masks", [])
            probs        = outputs.get("out_probs", [])
            out_obj_ids  = outputs.get("out_obj_ids", [])

            if len(binary_masks) == 0:
                continue

            for i, oid in enumerate(out_obj_ids):
                oid_val = oid.item() if hasattr(oid, "item") else int(oid)

                # Propagate any object SAM3 is tracking
                if oid_val not in obj_ids_to_propagate:
                    continue

                prob = probs[i].item() if hasattr(probs[i], "item") else float(probs[i])
                if prob < conf_thresh:
                    continue

                mask = binary_masks[i]
                mask_np = mask.cpu().numpy() if hasattr(mask, "cpu") else np.array(mask)

                existing = self._label_vol[frame_idx]
                # All detected instances get the same label_id for display
                # (since they all came from the same text prompt)
                self._label_vol[frame_idx] = np.where(
                    mask_np.astype(bool) & (existing == 0),
                    oid_val,  # keep same label for all instances of this concept
                    existing,
                )

        return self._label_vol.copy()


    # ------------------------------------------------------------------
    # Reset / query
    # ------------------------------------------------------------------

    def reset_object(self, obj_id: int) -> None:
        """
        Remove one object from tracking. SAM3 supports clean per-object
        removal without needing to reinitialise the whole session.
        """
        self._assert_initialized()
        self._label_vol[self._label_vol == obj_id] = 0
        self._active_obj_ids = []  # clear since session state changed

        self._predictor.handle_request(
            request=dict(
                type="remove_object",
                session_id=self._session_id,
                obj_id=obj_id,
            )
        )

    def reset_all(self) -> None:
        """Wipe all predictions; keep the session and encoded frames."""
        self._assert_initialized()
        self._label_vol[:] = 0
        self._active_obj_ids = []

        self._predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=self._session_id,
            )
        )
    def get_label_volume(self) -> np.ndarray:
        self._assert_initialized()
        return self._label_vol.copy()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def n_slices(self) -> int:
        return self._n_slices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_predictor(self) -> None:

        import sys
        # print(sys.path)
        if self._predictor is not None:
            return

        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError as e:
            raise ImportError(
                "SAM3 is not installed. "
                "Clone https://github.com/facebookresearch/sam3 and run `pip install -e .`"
            ) from e

        try:
            ckpt_dir  = Path(os.environ["SAM3_CHECKPOINT_DIR"])
            assert ckpt_dir.exists(), f"{ckpt_dir} does not exist!"
            # (in future) SAM3 loads checkpoints via HuggingFace by default; pass the dir so it
            # can find a locally cached copy, or leave empty to trigger HF download.
            ckpt_path = next(ckpt_dir.glob('*.pt'))
        except Exception as e:
            print(e)

        self._predictor  = build_sam3_video_predictor(
           checkpoint_path=ckpt_path,
            gpus_to_use=[self.gpu]
        )

    def _write_frames_to_disk(self, volume: np.ndarray) -> None:
        """Write normalised RGB PNGs. SAM3's video loader expects the same
        format as SAM2: a directory of zero-padded image files."""
        if self._frames_dir is not None:
            import shutil
            shutil.rmtree(self._frames_dir, ignore_errors=True)

        tmp = tempfile.mkdtemp(prefix="sam3_napari_")
        self._frames_dir = Path(tmp)

        vol = volume.astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin) * 255.0
        vol_u8 = vol.astype(np.uint8)

        n_digits = len(str(volume.shape[0] - 1))
        for z in range(volume.shape[0]):
            s = vol_u8[z]
            rgb = np.stack([s, s, s], axis=-1) if s.ndim == 2 else s[:, :, :3]
            Image.fromarray(rgb).save(
                self._frames_dir / f"{str(z).zfill(n_digits)}.jpg"
            )

    def _result_to_mask(
        self,
        result,
        params: dict,
        conf_threshold: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Convert add_prompt() response to a single (H, W) bool mask.
        
        SAM3 output keys:
            out_binary_masks : list/array of (H, W) bool arrays, one per instance
            out_probs        : list/array of float confidence scores
            out_obj_ids      : list/array of object IDs
        """
        if result is None:
            return None

        thresh = conf_threshold if conf_threshold is not None else params.get("score_threshold", 0.5)

        binary_masks = result.get("out_binary_masks")
        probs        = result.get("out_probs")

        if binary_masks is None or len(binary_masks) == 0:
            return None

        # Normalise to numpy
        masks_np  = np.array([
            m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)
            for m in binary_masks
        ])                                              # (N, H, W)
        scores_np = np.array([
            p.item() if hasattr(p, "item") else float(p)
            for p in probs
        ])                                              # (N,)

        keep = scores_np >= thresh
        if not keep.any():
            keep[np.argmax(scores_np)] = True           # always keep best if none pass

        #return masks_np[keep].any(axis=0).astype(bool)  # (H, W)
        return masks_np[keep]

    @staticmethod
    def _tensor_to_mask(obj_output: dict, threshold: float) -> Optional[np.ndarray]:
        """
        Convert a single object's entry from propagate_in_video frame output.
        Same keys as add_prompt output.
        """
        binary_masks = obj_output.get("out_binary_masks")
        probs        = obj_output.get("out_probs")

        if binary_masks is None or len(binary_masks) == 0:
            return None

        masks_np  = np.array([
            m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)
            for m in binary_masks
        ])
        scores_np = np.array([
            p.item() if hasattr(p, "item") else float(p)
            for p in probs
        ])

        keep = scores_np >= threshold
        if not keep.any():
            keep[np.argmax(scores_np)] = True

        return masks_np[keep].any(axis=0).astype(bool)

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "SAM3ClickAndPropagate is not initialised. Call .initialize(volume) first."
            )

    def __del__(self):
        try:
            if self._session_id is not None and self._predictor is not None:
                self._predictor.handle_request(
                    request=dict(
                        type="close_session",
                        session_id=self._session_id,
                    )
                )
        except Exception:
            pass
        if self._frames_dir is not None:
            import shutil
            shutil.rmtree(self._frames_dir, ignore_errors=True)
