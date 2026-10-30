"""medseg - medical image preprocessing / segmentation toolkit built from the SUP'COM workshop TPs.

Conventions used everywhere in the package:
- images are ``numpy.uint8`` arrays, either grayscale ``(H, W)`` or RGB ``(H, W, 3)``
  (RGB order, never BGR: conversion happens once, in :mod:`medseg.io`);
- binary masks are ``numpy.bool_`` arrays ``(H, W)``;
- label maps are integer arrays ``(H, W)``;
- every processing function is pure: inputs are never modified in place.
"""

__version__ = "1.0.0"
