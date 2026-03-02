#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform
from scipy.stats import truncnorm

# ── Internal data container ───────────────────────────────────────────────────


@dataclass
class Sample:
    """Data container passed through the transform pipeline.

    Attributes:
        image:     HWC numpy array (uint8, or float32 after Normalize).
        keypoints: (N, 2) float32 array, NaN marks invisible keypoints.
        bboxes:    (M, 4) float32 array in XYXY pixel coordinates.
    """

    image: np.ndarray
    keypoints: np.ndarray
    bboxes: np.ndarray

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """numpy HWC uint8 → torch CHW uint8 tensor."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)


def _from_tensor(t: torch.Tensor) -> np.ndarray:
    """torch CHW → numpy HWC."""
    return t.permute(1, 2, 0).numpy()


# ── Compose ───────────────────────────────────────────────────────────────────


class Compose:
    """Transform pipeline with an albumentations-compatible dict interface.

    Args:
        transforms: list of transform objects (subclasses of ``_Transform``).

    Call signature (same as ``albumentations.Compose``)::

        result = compose(
            image=img,            # numpy HWC uint8
            keypoints=kps,        # list of (x, y) tuples
            class_labels=labels,  # parallel list (preserved unchanged)
            bboxes=boxes,         # list of (x, y, w, h) COCO tuples
            bbox_labels=blabels,  # parallel list (preserved unchanged)
        )

    Returns a dict with the same keys.  Keypoints are (x, y) tuples; NaN
    coordinates mark keypoints that left the image boundary.  Bboxes are
    COCO (x, y, w, h) tuples clipped to the image.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self,
        image: np.ndarray,
        keypoints: list | None = None,
        class_labels: list | None = None,
        bboxes: list | None = None,
        bbox_labels: list | None = None,
    ) -> dict:
        keypoints = list(keypoints or [])
        class_labels = list(class_labels or [])
        bboxes = list(bboxes or [])
        bbox_labels = list(bbox_labels or [])

        # (x, y) tuples → (N, 2) float32
        kp_xy = (
            np.array([[k[0], k[1]] for k in keypoints], dtype=np.float32)
            if keypoints
            else np.empty((0, 2), dtype=np.float32)
        )
        # COCO (x, y, w, h) → XYXY (x1, y1, x2, y2)
        bbox_xyxy = (
            np.array(
                [[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in bboxes],
                dtype=np.float32,
            )
            if bboxes
            else np.empty((0, 4), dtype=np.float32)
        )

        sample = Sample(image=image, keypoints=kp_xy, bboxes=bbox_xyxy)
        for t in self.transforms:
            sample = t(sample)

        out_kps = [(float(xy[0]), float(xy[1])) for xy in sample.keypoints]
        out_bboxes = [
            (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            for x1, y1, x2, y2 in sample.bboxes
        ]
        return {
            "image": sample.image,
            "keypoints": out_kps,
            "class_labels": class_labels,
            "bboxes": out_bboxes,
            "bbox_labels": bbox_labels,
        }


# ── Base ──────────────────────────────────────────────────────────────────────


class _Transform:
    """Abstract base for all transforms."""

    def __call__(self, sample: Sample) -> Sample:
        raise NotImplementedError


# ── Image-only transforms ─────────────────────────────────────────────────────


class Normalize(_Transform):
    """Normalise image to float32 using per-channel mean/std (ImageNet defaults)."""

    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, sample: Sample) -> Sample:
        t = _to_tensor(sample.image).float().div(255.0)
        t = TF.normalize(t, mean=self.mean, std=self.std)
        sample.image = _from_tensor(t)
        return sample


class ScaleToUnitRange(_Transform):
    """Scale uint8 image to [0, 1] float32."""

    def __call__(self, sample: Sample) -> Sample:
        sample.image = sample.image.astype(np.float32) / 255.0
        return sample


class Equalize(_Transform):
    """Histogram equalisation applied with probability *p*."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        t = _to_tensor(sample.image)
        sample.image = _from_tensor(TF.equalize(t))
        return sample


class GaussianNoise(_Transform):
    """Add per-channel Gaussian noise."""

    def __init__(
        self,
        var_limit: tuple[float, float] = (0.0, (0.05 * 255) ** 2),
        p: float = 0.5,
    ) -> None:
        self.var_limit = var_limit
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        var = np.random.uniform(*self.var_limit)
        sigma = math.sqrt(var)
        noise = np.random.normal(0.0, sigma, sample.image.shape).astype(np.float32)
        img = sample.image.astype(np.float32) + noise
        sample.image = np.clip(img, 0, 255).astype(sample.image.dtype)
        return sample


class MotionBlur(_Transform):
    """Apply motion blur with a random direction."""

    def __init__(
        self,
        blur_limit: int | tuple[int, int] = 7,
        p: float = 0.5,
    ) -> None:
        self.blur_limit: tuple[int, int] = (
            (3, blur_limit) if isinstance(blur_limit, int) else tuple(blur_limit)  # type: ignore[assignment]
        )
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        lo, hi = self.blur_limit
        # Kernel size must be odd
        choices = list(range(lo | 1, hi + 1, 2))
        k = int(np.random.choice(choices)) if choices else (lo | 1)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        angle = float(np.random.uniform(0, 360))
        M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        kernel /= kernel.sum() + 1e-7
        sample.image = cv2.filter2D(sample.image, -1, kernel)
        return sample


class Grayscale(_Transform):
    """Convert to grayscale with alpha blending back to the original colour image."""

    def __init__(
        self,
        alpha: float | tuple[float, float] = 1.0,
        p: float = 0.5,
    ) -> None:
        if isinstance(alpha, (float, int)):
            self._alpha: float | tuple[float, float] = float(np.clip(alpha, 0.0, 1.0))
        elif isinstance(alpha, tuple) and len(alpha) == 2:
            self._alpha = (
                float(np.clip(alpha[0], 0.0, 1.0)),
                float(np.clip(alpha[1], 0.0, 1.0)),
            )
        else:
            raise ValueError("`alpha` must be a float or a 2-tuple of floats.")
        self.p = p

    @property
    def alpha(self) -> float:
        if isinstance(self._alpha, tuple):
            return float(np.random.uniform(*self._alpha))
        return self._alpha

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        img = sample.image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_rgb = np.stack([gray, gray, gray], axis=-1)
        a = self.alpha
        sample.image = (img * (1.0 - a) + gray_rgb * a).astype(img.dtype)
        return sample


# ── Geometric transforms ──────────────────────────────────────────────────────


class HorizontalFlip(_Transform):
    """Flip image, keypoints, and bboxes horizontally."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def _flip_keypoints(self, kp: np.ndarray, width: int) -> np.ndarray:
        kp = kp.copy()
        valid = ~np.isnan(kp[:, 0])
        kp[valid, 0] = (width - 1) - kp[valid, 0]
        return kp

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        w = sample.width
        sample.image = sample.image[:, ::-1, :].copy()
        if len(sample.keypoints):
            sample.keypoints = self._flip_keypoints(sample.keypoints, w)
        if len(sample.bboxes):
            b = sample.bboxes.copy()
            x1_new = w - b[:, 2]
            x2_new = w - b[:, 0]
            b[:, 0], b[:, 2] = x1_new, x2_new
            sample.bboxes = b
        return sample


class HFlip(HorizontalFlip):
    """Horizontal flip that also swaps symmetric keypoint pairs."""

    def __init__(self, symmetries: list[tuple[int, int]], p: float = 0.5) -> None:
        super().__init__(p=p)
        self._sym: dict[int, int] = {}
        for i, j in symmetries:
            self._sym[i] = j
            self._sym[j] = i

    def _flip_keypoints(self, kp: np.ndarray, width: int) -> np.ndarray:
        kp = kp.copy()
        # Swap symmetric pairs first
        for i, j in self._sym.items():
            if i < j:
                kp[[i, j]] = kp[[j, i]]
        valid = ~np.isnan(kp[:, 0])
        kp[valid, 0] = (width - 1) - kp[valid, 0]
        return kp


class Resize(_Transform):
    """Resize image to (height, width), scaling keypoints and bboxes proportionally."""

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, sample: Sample) -> Sample:
        old_h, old_w = sample.height, sample.width
        sample.image = cv2.resize(
            sample.image, (self.width, self.height), interpolation=self.interpolation
        )
        sx = self.width / old_w
        sy = self.height / old_h
        if len(sample.keypoints):
            sample.keypoints[:, 0] *= sx
            sample.keypoints[:, 1] *= sy
        if len(sample.bboxes):
            sample.bboxes[:, [0, 2]] *= sx
            sample.bboxes[:, [1, 3]] *= sy
        return sample


class LongestMaxSize(_Transform):
    """Resize so the longest side equals *max_size*, preserving aspect ratio."""

    def __init__(self, max_size: int, interpolation: int = cv2.INTER_LINEAR) -> None:
        self.max_size = max_size
        self.interpolation = interpolation

    def __call__(self, sample: Sample) -> Sample:
        h, w = sample.height, sample.width
        scale = self.max_size / max(h, w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        sample.image = cv2.resize(
            sample.image, (new_w, new_h), interpolation=self.interpolation
        )
        if len(sample.keypoints):
            sample.keypoints *= scale
        if len(sample.bboxes):
            sample.bboxes *= scale
        return sample


class PadIfNeeded(_Transform):
    """Pad image to at least (min_height, min_width) or to a size divisible by given values.

    Args:
        min_height:         minimum output height.
        min_width:          minimum output width.
        pad_height_divisor: if set, pad height to be divisible by this value.
        pad_width_divisor:  if set, pad width to be divisible by this value.
        position:           one of ``"top_left"``, ``"center"``, ``"random"``.
        border_mode:        ``"constant"`` or ``"reflect_101"``.
        border_value:       fill value for constant border.
    """

    def __init__(
        self,
        min_height: int | None = None,
        min_width: int | None = None,
        pad_height_divisor: int | None = None,
        pad_width_divisor: int | None = None,
        position: str = "top_left",
        border_mode: str = "constant",
        border_value: float | int = 0,
    ) -> None:
        self.min_height = min_height
        self.min_width = min_width
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor
        self.position = position
        self.border_mode = border_mode
        self.border_value = border_value

    def _target_size(self, h: int, w: int) -> tuple[int, int]:
        th, tw = h, w
        if self.min_height is not None:
            th = max(th, self.min_height)
        if self.min_width is not None:
            tw = max(tw, self.min_width)
        if self.pad_height_divisor and self.pad_height_divisor > 1:
            th = math.ceil(th / self.pad_height_divisor) * self.pad_height_divisor
        if self.pad_width_divisor and self.pad_width_divisor > 1:
            tw = math.ceil(tw / self.pad_width_divisor) * self.pad_width_divisor
        return th, tw

    def _offsets(self, pad_h: int, pad_w: int) -> tuple[int, int]:
        if self.position == "center":
            return pad_h // 2, pad_w // 2
        if self.position == "random":
            top = np.random.randint(0, pad_h + 1) if pad_h > 0 else 0
            left = np.random.randint(0, pad_w + 1) if pad_w > 0 else 0
            return top, left
        return 0, 0  # top_left

    def __call__(self, sample: Sample) -> Sample:
        h, w = sample.height, sample.width
        th, tw = self._target_size(h, w)
        if th == h and tw == w:
            return sample

        pad_h = th - h
        pad_w = tw - w
        top, left = self._offsets(pad_h, pad_w)
        bottom = pad_h - top
        right = pad_w - left

        cv2_mode = (
            cv2.BORDER_REFLECT_101
            if self.border_mode == "reflect_101"
            else cv2.BORDER_CONSTANT
        )
        sample.image = cv2.copyMakeBorder(
            sample.image, top, bottom, left, right, cv2_mode, value=self.border_value
        )
        if len(sample.keypoints):
            sample.keypoints[:, 0] += left
            sample.keypoints[:, 1] += top
        if len(sample.bboxes):
            sample.bboxes[:, [0, 2]] += left
            sample.bboxes[:, [1, 3]] += top
        return sample


class Affine(_Transform):
    """Random affine transform (scale, rotation, translation) around image centre.

    Args:
        scale:        ``(min, max)`` scale range or a fixed scale value.
        rotate:       ``(min_deg, max_deg)`` rotation range or a fixed angle.
        translate_px: ``(min, max)`` pixel translation range or a fixed value.
        p:            probability of applying the transform.
        keep_ratio:   if True, the same scale factor is used for x and y.
    """

    def __init__(
        self,
        scale: tuple[float, float] | float | None = None,
        rotate: tuple[float, float] | float | None = None,
        translate_px: tuple[int, int] | int | None = None,
        p: float = 0.9,
        keep_ratio: bool = True,
    ) -> None:
        self.scale = scale
        self.rotate = rotate
        self.translate_px = translate_px
        self.p = p
        self.keep_ratio = keep_ratio

    def _sample_scale(self) -> tuple[float, float]:
        if self.scale is None:
            return 1.0, 1.0
        lo, hi = (
            (self.scale, self.scale)
            if isinstance(self.scale, (int, float))
            else self.scale
        )
        s = float(np.random.uniform(lo, hi))
        if self.keep_ratio:
            return s, s
        return float(np.random.uniform(lo, hi)), float(np.random.uniform(lo, hi))

    def _sample_rotate(self) -> float:
        if self.rotate is None:
            return 0.0
        lo, hi = (
            (self.rotate, self.rotate)
            if isinstance(self.rotate, (int, float))
            else self.rotate
        )
        return float(np.random.uniform(lo, hi))

    def _sample_translate(self) -> tuple[float, float]:
        if self.translate_px is None:
            return 0.0, 0.0
        lo, hi = (
            (self.translate_px, self.translate_px)
            if isinstance(self.translate_px, (int, float))
            else self.translate_px
        )
        return float(np.random.uniform(lo, hi)), float(np.random.uniform(lo, hi))

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample

        h, w = sample.height, sample.width
        cx, cy = w / 2.0, h / 2.0
        sx, sy = self._sample_scale()
        angle = self._sample_rotate()
        tx, ty = self._sample_translate()

        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # Affine matrix: rotate+scale around image centre, then translate
        M = np.array(
            [
                [
                    sx * cos_a,
                    -sy * sin_a,
                    tx + cx * (1 - sx * cos_a) + cy * sy * sin_a,
                ],
                [
                    sx * sin_a,
                    sy * cos_a,
                    ty + cy * (1 - sy * cos_a) - cx * sx * sin_a,
                ],
            ],
            dtype=np.float64,
        )
        sample.image = cv2.warpAffine(
            sample.image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if len(sample.keypoints):
            kp = sample.keypoints.copy()
            valid = ~np.isnan(kp[:, 0])
            if valid.any():
                pts = np.column_stack([kp[valid], np.ones(valid.sum())])
                kp[valid] = (M @ pts.T).T
                # Mark newly out-of-bounds as NaN
                oob = valid & (
                    (kp[:, 0] < 0) | (kp[:, 0] >= w) | (kp[:, 1] < 0) | (kp[:, 1] >= h)
                )
                kp[oob] = np.nan
            sample.keypoints = kp

        if len(sample.bboxes):
            new_bboxes = np.empty_like(sample.bboxes)
            for i, (x1, y1, x2, y2) in enumerate(sample.bboxes):
                corners = np.array(
                    [[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]],
                    dtype=np.float64,
                )
                c = (M @ corners.T).T
                new_bboxes[i] = [
                    c[:, 0].min(),
                    c[:, 1].min(),
                    c[:, 0].max(),
                    c[:, 1].max(),
                ]
            new_bboxes[:, [0, 2]] = np.clip(new_bboxes[:, [0, 2]], 0, w)
            new_bboxes[:, [1, 3]] = np.clip(new_bboxes[:, [1, 3]], 0, h)
            sample.bboxes = new_bboxes.astype(np.float32)

        return sample


class KeepAspectRatioResize(_Transform):
    """Resize preserving aspect ratio.

    In ``'pad'`` mode the image is scaled so it fits inside (height, width) and
    can be padded up to the target size.  In ``'crop'`` mode it is scaled so it
    covers the target size and can be cropped down.
    """

    def __init__(
        self,
        width: int,
        height: int,
        mode: str = "pad",
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.mode = mode
        self.interpolation = interpolation
        self.p = p

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample
        h, w = sample.height, sample.width
        scale = (
            min(self.height / h, self.width / w)
            if self.mode == "pad"
            else max(self.height / h, self.width / w)
        )
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        sample.image = cv2.resize(
            sample.image, (new_w, new_h), interpolation=self.interpolation
        )
        if len(sample.keypoints):
            sample.keypoints *= scale
        if len(sample.bboxes):
            sample.bboxes *= scale
        return sample


# ── Custom pose-estimation transforms ────────────────────────────────────────


class KeypointAwareCrop(_Transform):
    """Random crop centred around annotated keypoints.

    Args:
        width:         crop width in pixels.
        height:        crop height in pixels.
        max_shift:     maximum centre offset as a fraction of crop size (clamped to 0.4).
        crop_sampling: one of ``"uniform"``, ``"keypoints"``, ``"density"``, ``"hybrid"``.
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_shift: float = 0.4,
        crop_sampling: str = "hybrid",
    ) -> None:
        self.width = width
        self.height = height
        self.max_shift = max(0.0, min(max_shift, 0.4))
        if crop_sampling not in ("uniform", "keypoints", "density", "hybrid"):
            raise ValueError(
                f"Invalid sampling '{crop_sampling}'. Must be one of: "
                "'uniform', 'keypoints', 'density', 'hybrid'."
            )
        self.crop_sampling = crop_sampling

    @staticmethod
    def _calc_n_neighbors(xy: NDArray, radius: float) -> NDArray:
        d = pdist(xy, "sqeuclidean")
        mat = squareform(d <= radius * radius, checks=False)
        return np.sum(mat, axis=0)

    def _crop_origin(self, sample: Sample) -> tuple[int, int]:
        h, w = sample.height, sample.width
        kpts = sample.keypoints
        valid_kpts = kpts[~np.isnan(kpts[:, 0])] if len(kpts) else kpts

        sampling = self.crop_sampling
        if sampling == "hybrid":
            sampling = np.random.choice(["uniform", "density"])
        if len(valid_kpts) == 0:
            sampling = "uniform"

        shift = (
            self.max_shift * np.random.random(2) * np.array([self.width, self.height])
        )

        if sampling == "uniform":
            # h_start, w_start ∈ [0, 1) → top-left from RandomCrop convention
            center = np.random.random(2)
        else:
            n = len(valid_kpts)
            inds = np.arange(n)
            if sampling == "density":
                radius = 0.1 * min(h, w)
                n_neighbors = self._calc_n_neighbors(valid_kpts, radius) + 1
                p = n_neighbors / n_neighbors.sum()
            else:
                p = np.ones(n) / n
            center = (valid_kpts[np.random.choice(inds, p=p)] + shift) / [w, h]
            center = np.clip(center, 0.0, np.nextafter(1.0, 0.0))

        left = int((w - self.width) * center[0])
        top = int((h - self.height) * center[1])
        left = max(0, min(left, w - self.width))
        top = max(0, min(top, h - self.height))
        return top, left

    def __call__(self, sample: Sample) -> Sample:
        top, left = self._crop_origin(sample)
        sample.image = sample.image[top : top + self.height, left : left + self.width]

        if len(sample.keypoints):
            kp = sample.keypoints.copy()
            kp[:, 0] -= left
            kp[:, 1] -= top
            oob = (
                (kp[:, 0] < 0)
                | (kp[:, 0] >= self.width)
                | (kp[:, 1] < 0)
                | (kp[:, 1] >= self.height)
            )
            kp[oob] = np.nan
            sample.keypoints = kp

        if len(sample.bboxes):
            b = sample.bboxes.copy()
            b[:, [0, 2]] -= left
            b[:, [1, 3]] -= top
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, self.width)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, self.height)
            sample.bboxes = b

        return sample

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "width", "height", "max_shift", "crop_sampling"


class ElasticTransform(_Transform):
    """Elastic deformation of image and keypoints.

    The displacement field is generated by smoothing uniform random noise with
    a Gaussian filter (controlled by *sigma*) and scaling by *alpha*.
    Keypoints are tracked via small binary heatmaps warped by the same field.
    """

    def __init__(
        self,
        alpha: float = 20.0,
        sigma: float = 5.0,
        interpolation: int = cv2.INTER_CUBIC,
        border_mode: int = cv2.BORDER_CONSTANT,
        value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.p = p
        self._neighbor_dist_sq = 9  # 3-pixel radius²

    def _displacement_field(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        ksize = int(8 * self.sigma + 1) | 1  # nearest odd integer
        dx = np.random.rand(h, w) * 2.0 - 1.0
        dx = cv2.GaussianBlur(dx, (ksize, ksize), self.sigma) * self.alpha
        return dx, dx  # same_dxdy=True (faster, matches original behaviour)

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample

        h, w = sample.height, sample.width
        dx, dy = self._displacement_field(h, w)

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + dx).astype(np.float32)
        map_y = (grid_y + dy).astype(np.float32)

        sample.image = cv2.remap(
            sample.image,
            map_x,
            map_y,
            self.interpolation,
            borderMode=self.border_mode,
            borderValue=self.value,
        )

        if len(sample.keypoints):
            kpts = sample.keypoints
            valid = np.all(kpts > 0.0, axis=1)
            heatmaps = np.zeros((h, w, len(kpts)), dtype=np.float32)
            # Build a small disc around each valid keypoint
            grid = np.mgrid[:h, :w].transpose(1, 2, 0)  # (H, W, 2) in (row, col) order
            for i, (kp, is_valid) in enumerate(zip(kpts, valid)):
                if is_valid:
                    dist = ((grid - kp[::-1]) ** 2).sum(axis=2)
                    heatmaps[:, :, i] = (dist <= self._neighbor_dist_sq).astype(
                        np.float32
                    )

            heatmaps_aug = cv2.remap(
                heatmaps,
                map_x,
                map_y,
                cv2.INTER_NEAREST,
                borderMode=self.border_mode,
                borderValue=0,
            )

            inds = np.indices(heatmaps_aug.shape[:2])[::-1]  # (2, H, W) in (x, y)
            mask = np.transpose(heatmaps_aug == 1, (2, 0, 1))  # (N, H, W)
            div = mask.sum(axis=(1, 2))
            sum_indices = np.einsum(
                "chw,xhw->cx", mask.astype(np.float32), inds.astype(np.float32)
            ).T

            new_kpts = kpts.copy()
            for i, (d, s) in enumerate(zip(div, sum_indices)):
                if d > 0:
                    new_kpts[i] = s / d
            sample.keypoints = new_kpts

        return sample


class CoarseDropout(_Transform):
    """Randomly occlude rectangular image regions, marking hidden keypoints as NaN.

    *max_height* and *max_width* may be given as fractions of the image
    dimension (float < 1) or as absolute pixel values (int ≥ 1).
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height: int | float = 8,
        max_width: int | float = 8,
        min_holes: int | None = None,
        min_height: int | float | None = None,
        min_width: int | float | None = None,
        fill_value: int = 0,
        p: float = 0.5,
    ) -> None:
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else 1
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.p = p

    @staticmethod
    def _px(val: int | float, dim: int) -> int:
        return int(val * dim) if isinstance(val, float) and val < 1.0 else int(val)

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p:
            return sample

        h, w = sample.height, sample.width
        n_holes = np.random.randint(self.min_holes, self.max_holes + 1)
        img = sample.image.copy()
        holes: list[tuple[int, int, int, int]] = []

        for _ in range(n_holes):
            hole_h = self._px(self.max_height, h)
            hole_w = self._px(self.max_width, w)
            min_h = self._px(self.min_height, h)
            min_w = self._px(self.min_width, w)
            if min_h < hole_h:
                hole_h = np.random.randint(min_h, hole_h + 1)
            if min_w < hole_w:
                hole_w = np.random.randint(min_w, hole_w + 1)
            x1 = np.random.randint(0, max(1, w - hole_w))
            y1 = np.random.randint(0, max(1, h - hole_h))
            x2, y2 = x1 + hole_w, y1 + hole_h
            img[y1:y2, x1:x2] = self.fill_value
            holes.append((x1, y1, x2, y2))

        sample.image = img

        if len(sample.keypoints):
            kp = sample.keypoints.copy()
            for i, (x, y) in enumerate(kp):
                if np.isnan(x):
                    continue
                for x1, y1, x2, y2 in holes:
                    if x1 <= x < x2 and y1 <= y < y2:
                        kp[i] = np.nan
                        break
            sample.keypoints = kp

        return sample


class RandomBBoxTransform(_Transform):
    """Random scale and shift jitter for bounding boxes (top-down pose estimation).

    Based on the mmpose ``RandomBBoxTransform``.  Passes the image and keypoints
    through unchanged; only bboxes are modified.
    """

    def __init__(
        self,
        shift_factor: float = 0.1,
        shift_prob: float = 0.25,
        scale_factor: tuple[float, float] = (0.5, 1.5),
        scale_prob: float = 1.0,
        sampling: str = "truncnorm",
        p: float = 1.0,
    ) -> None:
        self.shift_factor = shift_factor
        self.shift_prob = shift_prob
        self.scale_factor = scale_factor
        self.scale_prob = scale_prob
        self.sampling = sampling
        self.p = p

    def _sample(self, size: tuple, low: float, high: float) -> np.ndarray:
        if self.sampling == "truncnorm":
            return truncnorm.rvs(low, high, size=size).astype(np.float32)
        if self.sampling == "uniform":
            return (low + (high - low) * np.random.random(size)).astype(np.float32)
        raise ValueError(f"Unknown sampling: {self.sampling}")

    def __call__(self, sample: Sample) -> Sample:
        if np.random.random() >= self.p or len(sample.bboxes) == 0:
            return sample

        h, w = sample.height, sample.width
        # Normalise to [0, 1] for scale-invariant jitter
        norm = np.array([w, h, w, h], dtype=np.float32)
        bboxes = sample.bboxes / norm

        n = len(bboxes)
        scale_factors = np.ones((n, 2), dtype=np.float32)
        mask = np.random.random(n) < self.scale_prob
        if mask.any():
            scale_factors[mask] = self._sample(
                (mask.sum(), 2), self.scale_factor[0], self.scale_factor[1]
            )

        shift_factors = np.zeros((n, 2), dtype=np.float32)
        mask = np.random.random(n) < self.shift_prob
        if mask.any():
            shift_factors[mask] = self._sample(
                (mask.sum(), 2), -self.shift_factor, self.shift_factor
            )

        wh = bboxes[:, 2:] - bboxes[:, :2]
        cxcy = bboxes[:, :2] + 0.5 * wh
        cxcy += shift_factors * wh
        wh *= scale_factors
        half_wh = 0.5 * wh
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, :2] = cxcy - half_wh
        new_bboxes[:, 2:] = cxcy + half_wh
        new_bboxes = np.clip(new_bboxes, 0.0, 1.0)

        sample.bboxes = new_bboxes * norm
        return sample

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "shift_factor", "shift_prob", "scale_factor", "scale_prob", "sampling"


# ── Builder helpers ───────────────────────────────────────────────────────────


def build_transforms(augmentations: dict) -> Compose:
    transforms: list[_Transform] = []

    if resize_aug := augmentations.get("resize", False):
        transforms += build_resize_transforms(resize_aug)

    if (lms_cfg := augmentations.get("longest_max_size")) is not None:
        transforms.append(LongestMaxSize(lms_cfg))

    if hflip_cfg := augmentations.get("hflip"):
        hflip_proba = 0.5
        symmetries = None
        if isinstance(hflip_cfg, float):
            hflip_proba = hflip_cfg
        elif isinstance(hflip_cfg, dict):
            if "p" in hflip_cfg:
                hflip_proba = float(hflip_cfg["p"])
            if "symmetries" in hflip_cfg:
                symmetries = [(int(a), int(b)) for a, b in hflip_cfg["symmetries"]]

        if symmetries is not None:
            transforms.append(HFlip(symmetries=symmetries, p=hflip_proba))
        else:
            warnings.warn(
                "Be careful! Do not train pose models with horizontal flips if you have"
                " symmetric keypoints!"
            )
            transforms.append(HorizontalFlip(p=hflip_proba))

    if (affine := augmentations.get("affine")) is not None:
        scaling = affine.get("scaling")
        rotation = affine.get("rotation")
        translation = affine.get("translation")
        if rotation is not None:
            rotation = (-rotation, rotation)
        if translation is not None:
            translation = (-translation, translation)
        transforms.append(
            Affine(
                scale=scaling,
                rotate=rotation,
                translate_px=translation,
                p=affine.get("p", 0.9),
                keep_ratio=True,
            )
        )

    if bbox_tfm := augmentations.get("random_bbox_transform", False):
        transforms.append(
            RandomBBoxTransform(
                shift_factor=bbox_tfm.get("shift_factor", 0.1),
                shift_prob=bbox_tfm.get("shift_prob", 0.25),
                scale_factor=bbox_tfm.get("scale_factor", (0.75, 1.25)),
                scale_prob=bbox_tfm.get("scale_prob", 1.0),
                p=bbox_tfm.get("p", 1.0),
            )
        )

    if crop_sampling := augmentations.get("crop_sampling"):
        transforms.append(
            PadIfNeeded(
                min_height=crop_sampling["height"],
                min_width=crop_sampling["width"],
                border_mode="constant",
            )
        )
        transforms.append(
            KeypointAwareCrop(
                crop_sampling["width"],
                crop_sampling["height"],
                crop_sampling["max_shift"],
                crop_sampling["method"],
            )
        )

    if augmentations.get("hist_eq", False):
        transforms.append(Equalize(p=0.5))
    if augmentations.get("motion_blur", False):
        transforms.append(MotionBlur(p=0.5))
    if augmentations.get("covering", False):
        transforms.append(
            CoarseDropout(
                max_holes=10,
                max_height=0.05,
                min_height=0.01,
                max_width=0.05,
                min_width=0.01,
                p=0.5,
            )
        )
    if augmentations.get("elastic_transform", False):
        transforms.append(ElasticTransform(sigma=5, p=0.5))
    if augmentations.get("grayscale", False):
        transforms.append(Grayscale(alpha=(0.5, 1.0)))
    if noise := augmentations.get("gaussian_noise", False):
        if not isinstance(noise, (int, float)):
            noise = 0.05 * 255
        transforms.append(
            GaussianNoise(
                var_limit=(0, noise**2),
                p=0.5,
            )
        )

    if augmentations.get("auto_padding"):
        transforms.append(build_auto_padding(**augmentations["auto_padding"]))

    if augmentations.get("normalize_images"):
        transforms.append(Normalize())

    if augmentations.get("scale_to_unit_range"):
        transforms.append(ScaleToUnitRange())

    return Compose(transforms)


def build_auto_padding(
    min_height: int | None = None,
    min_width: int | None = None,
    pad_height_divisor: int | None = 1,
    pad_width_divisor: int | None = 1,
    position: str = "random",
    border_mode: str = "reflect_101",
    border_value: float | None = None,
    border_mask_value: float | None = None,  # kept for API compatibility
) -> PadIfNeeded:
    """Create a :class:`PadIfNeeded` transform from a config dict.

    Args:
        min_height:         minimum output height.
        min_width:          minimum output width.
        pad_height_divisor: pad height to be divisible by this value.
        pad_width_divisor:  pad width to be divisible by this value.
        position:           ``"top_left"``, ``"center"``, or ``"random"``.
        border_mode:        ``"constant"`` or ``"reflect_101"``.
        border_value:       fill value for constant padding.
        border_mask_value:  ignored (kept for backwards compatibility).

    Returns:
        Configured :class:`PadIfNeeded` instance.
    """
    valid_modes = ("constant", "reflect_101")
    if border_mode not in valid_modes:
        raise ValueError(
            f"Unknown border mode for auto_padding: '{border_mode}' "
            f"(valid values: {valid_modes})"
        )
    return PadIfNeeded(
        min_height=min_height,
        min_width=min_width,
        pad_height_divisor=pad_height_divisor
        if (pad_height_divisor or 1) > 1
        else None,
        pad_width_divisor=pad_width_divisor if (pad_width_divisor or 1) > 1 else None,
        position=position,
        border_mode=border_mode,
        border_value=border_value or 0,
    )


def build_resize_transforms(resize_cfg: dict) -> list[_Transform]:
    """Build a list of resize transforms from a config dict.

    Args:
        resize_cfg: dict with keys ``height``, ``width``, and optionally
            ``keep_ratio`` (default ``True``).

    Returns:
        List of one or two :class:`_Transform` instances.
    """
    height, width = resize_cfg["height"], resize_cfg["width"]
    if resize_cfg.get("keep_ratio", True):
        return [
            KeepAspectRatioResize(width=width, height=height, mode="pad"),
            PadIfNeeded(min_height=height, min_width=width, position="top_left"),
        ]
    return [Resize(height, width)]
