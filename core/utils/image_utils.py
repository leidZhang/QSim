from typing import Tuple

import cv2
import numpy as np


def region_of_interest(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon

    Parameters:
    - img (np.ndarray): Input image for line following.
    - vertices (list): Vertices of the region of interest.

    Returns:
    - np.ndarray: Image with the region of interest.
    """
    mask: np.ndarray = np.zeros_like(img)
    channel_count: int = img.shape[2]
    match_mask_color: tuple = (255,) * channel_count
    cv2.fillPoly(mask, vertices, match_mask_color)
    return cv2.bitwise_and(img, mask)


def find_slope_intercept_from_points(points: np.ndarray) -> Tuple[float, float]:
    """
    This function will return the linear polinomial coefficients to the lane found
    in the point set

    Parameters:
    - points: np.ndarray: The points of the lane

    Returns:
    - Tuple[float, float]: The slope and intercept of the lane
    """
    x: np.ndarray = points[:, 0]
    y: np.ndarray = points[:, 1]
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def find_slope_intercept_from_binary(binary: np.ndarray) -> Tuple[float, float]:
    """
    This function will return the linear polinomial coefficients to the lane found
    in the binary image

    Parameters:
    - binary: np.ndarray: The binary image

    Returns:
    - Tuple[float, float]: The slope and intercept of the lane
    """
    # convert the binary image to row/col format
    line_pixels: np.ndarray = np.argwhere(binary > 0)
    rows: np.ndarray = line_pixels[0:-1, 0]
    cols: np.ndarray = line_pixels[0:-1, 1]

    num_of_pixels: int = len(rows)
    num_of_choices: int = int(num_of_pixels * 0.1)
    indices: np.ndarray = np.random.choice(num_of_pixels, num_of_choices)
    x, y = cols, binary.shape[0] - rows
    if len(x) > 0 and len(y) > 0:
        slope, intercept = np.polyfit(x[indices], y[indices], 1)
    else:
        slope, intercept = 0.0, 0.0

    return slope, intercept