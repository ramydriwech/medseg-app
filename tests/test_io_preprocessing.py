import cv2
import numpy as np
import pytest

from medseg.evaluation.similarity import mse
from medseg.io import align_mask, describe_image, load_image, load_mask, to_gray
from medseg.preprocessing import add_gaussian_noise, clahe, gaussian_filter, mean_filter, median_filter


def encode(arr):
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def test_load_color_is_rgb():
    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[..., 2] = 200  # red in BGR
    loaded = load_image(encode(bgr), name="x.png")
    assert loaded.image.shape == (4, 5, 3)
    assert loaded.image[0, 0].tolist() == [200, 0, 0]


def test_identical_channels_become_gray():
    g = np.arange(20, dtype=np.uint8).reshape(4, 5)
    loaded = load_image(encode(np.stack([g, g, g], -1)))
    assert loaded.image.ndim == 2 and np.array_equal(loaded.image, g)
    assert loaded.notes


def test_16bit_is_rescaled():
    arr = (np.arange(16, dtype=np.uint16).reshape(4, 4) * 1000)
    loaded = load_image(encode(arr))
    assert loaded.image.dtype == np.uint8 and loaded.image.max() == 255 and loaded.image.min() == 0


def test_unsupported_extension(tmp_path):
    p = tmp_path / "scan.dcm"
    p.write_bytes(b"x")
    with pytest.raises(ValueError):
        load_image(p)


def test_load_mask_and_align():
    m = np.zeros((10, 10), np.uint8)
    m[2:5, 2:5] = 255
    mask = load_mask(encode(m))
    assert mask.dtype == bool and mask.sum() == 9
    big = align_mask(mask, (20, 20))
    assert big.shape == (20, 20) and big.sum() == 36


def test_describe_image():
    loaded = load_image(encode(np.zeros((6, 8, 3), np.uint8) + [1, 2, 3]))
    info = describe_image(loaded)
    assert info["Dimensions (L x H)"] == "8 x 6" and info["Canaux"] == 3


@pytest.mark.parametrize("func,kwargs", [
    (median_filter, {"ksize": 5}), (gaussian_filter, {"sigma": 1.0}), (mean_filter, {"radius": 3}), (clahe, {}),
])
@pytest.mark.parametrize("color", [False, True])
def test_filters_keep_shape_dtype_and_purity(func, kwargs, color, dark_disk):
    rgb, _ = dark_disk
    img = rgb if color else to_gray(rgb)
    before = img.copy()
    out = func(img, **kwargs)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(img, before)  # pure: input untouched


@pytest.mark.parametrize("func", [median_filter, gaussian_filter, mean_filter])
def test_denoising_reduces_error(func, dark_disk):
    clean = to_gray(dark_disk[0])
    noisy = add_gaussian_noise(clean, std=25, seed=3)
    assert mse(func(noisy), clean) < mse(noisy, clean)


def test_noise_is_reproducible():
    img = np.full((10, 10), 100, np.uint8)
    assert np.array_equal(add_gaussian_noise(img, seed=1), add_gaussian_noise(img, seed=1))
    assert not np.array_equal(add_gaussian_noise(img, seed=1), add_gaussian_noise(img, seed=2))


def test_median_rejects_even_kernel():
    with pytest.raises(ValueError):
        median_filter(np.zeros((5, 5), np.uint8), 4)
