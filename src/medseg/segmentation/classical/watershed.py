"""Marker-controlled watershed (TP2 ex.5), with automatic markers.

TP2 built the markers in two ways, both relying on the ground truth:
- "gradient" variant: the morphological gradient was restricted to a search zone obtained
  by dilating the glottis mask (``image_glottis``, the ground truth - a variable that is
  moreover never defined in the notebook, so the cell cannot run as is);
- "ground truth" variant: sure foreground = GT eroded 3 times, sure background = complement
  of the GT dilated 10 times (5 x 5 kernel).
A segmentation method cannot use the reference it is evaluated against, so the GT is
replaced here by an *automatic* initial mask, everything else being kept from TP2:

1. initial mask = Otsu threshold (TP2 ex.2) restricted to the useful field, + shared
   component filtering (TP3 ex.3), largest component - no ground truth involved;
2. sure foreground = initial mask eroded ``fg_erosions`` times (TP2: 3);
3. sure background = complement of the mask dilated ``bg_dilations`` times (TP2: 10);
4. ``cv2.watershed`` on the colour image (TP2) floods the unknown band between them.

The watershed therefore *refines the contour* of the Otsu mask along the image edges;
its interest in the comparison is precisely to show what that refinement brings.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...io import ensure_rgb, to_gray
from ..postprocessing import Target, field_of_view, filter_components, morphological_opening


def watershed_segmentation(
    image: np.ndarray,
    fg_erosions: int = 3,
    bg_dilations: int = 10,
    kernel_size: int = 5,
    min_area: int = 20,
    target: Target = "darkest",
    fov_threshold: int = 28,
    fov_margin: int = 5,
    exclude_border: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Watershed with automatic markers. Returns labels 1 = region of interest, 0 = rest.

    ``target``, ``fov_*``, ``min_area`` and ``exclude_border`` are the shared post-processing
    parameters, injected by the pipeline so that the initial mask is built with exactly the
    same rules as the masks of the other methods.
    """
    gray = to_gray(image)
    h, w = gray.shape
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    initial = binary == 0 if target == "darkest" else binary > 0
    field = field_of_view(gray, fov_threshold, fov_margin)
    initial = morphological_opening(initial & field, kernel_size)
    initial = filter_components(
        initial, exclude_border=exclude_border, min_area=min_area, keep="largest", field=field
    )
    if not initial.any():
        return np.zeros((h, w), dtype=np.int32), {"marqueur_fg_px": 0}

    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    m = initial.astype(np.uint8) * 255
    sure_fg = cv2.erode(m, kernel, iterations=int(fg_erosions))
    erosions = int(fg_erosions)
    while not sure_fg.any() and erosions > 0:  # small objects: erode less
        erosions -= 1
        sure_fg = cv2.erode(m, kernel, iterations=erosions)
    sure_bg = cv2.bitwise_not(cv2.dilate(m, kernel, iterations=int(bg_dilations)))

    markers = np.zeros((h, w), dtype=np.int32)
    markers[sure_bg == 255] = 1
    markers[sure_fg == 255] = 2
    bgr = cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2BGR)
    result = cv2.watershed(bgr, markers)
    labels = (result == 2).astype(np.int32)
    return labels, {"marqueur_fg_px": int((sure_fg > 0).sum()), "érosions_utilisées": erosions}
