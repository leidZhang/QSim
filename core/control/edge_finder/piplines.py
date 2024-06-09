from typing import Tuple, Union

import cv2
import numpy as np

from hal.utilities.image_processing import ImageProcessing

from .exceptions import NoImageException, NoContourException
from .constants import HOUGH_ANGLE_LOWER_BOUND, HOUGH_ANGLE_UPPER_BOUND
from .constants import EDGES_LOWER_BOUND, EDGES_UPPER_BOUND, HOUGH_CONFIDENT_THRESHOLD
from .constants import THRESH_LOWER_BOUND, THRESH_UPPER_BOUND


class SobelPipeLine:
    """
    The SobelPipeLine class is responsible for finding the edges of the road using Sobel
    edge detection. It can be used in serial or parallel with other edge finders.
    """

    def __init__(self, device: str = 'cpu') -> None:
        """
        Initialize the SobelPipeLine instance

        Parameters:
        - device: str: the device to use for edge detection
        """
        if device == 'gpu':
            self.get_edge_image = self._apply_sobel_gpu
            self.gpu_mask: cv2.cuda_GpuMat = cv2.cuda_GpuMat()
            self.sobel_filter: cv2.cuda_SobelFilter = cv2.cuda.createSobelFilter(
                cv2.CV_64F, cv2.CV_64F, 1, 1, ksize=15
            )
        else:
            self.get_edge_image = self._apply_sobel_cpu


    def __call__(self, largest_contour: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        The main method for the SobelPipeLine class.

        Parameters:
        - largest_contour (np.ndarray): Contours found in the input image.
        - image (np.ndarray): The input image for edge detection.

        Returns:
        - np.ndarray: The edge image of the input image.
        """
        return self.get_edge_image(largest_contour, image)

    def _apply_sobel_gpu(self, largest_contour: np.ndarray, gpu_image: cv2.cuda_GpuMat) -> np.ndarray:
        """
        Gets the edge image from the input image and largest_contour.

        Prameters:
        - largest_contour (np.ndarray): Contours found in the input image.
        - image (np.ndarray): The input image for edge detection.

        Returns:
        - np.ndarray: The edge image of the input image.
        """
        image: np.ndarray = gpu_image.download()
        hull: np.ndarray = cv2.convexHull(largest_contour)

        mask: np.ndarray = np.zeros_like(image)
        cv2.fillPoly(mask, [hull], (255, 255, 255))
        self.gpu_mask.upload(mask)

        # Apply the Sobel edge detector in the GPU
        gpu_diff: cv2.cuda_GpuMat = self.sobel_filter.apply(self.gpu_mask)
        # Apply the threshold in the GPU
        _, gpu_edge = cv2.cuda.threshold(gpu_diff, 0, 255, cv2.THRESH_BINARY)

        return gpu_edge.download()

    def _apply_sobel_cpu(self, largest_contour: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Gets the edge image from the input image and contours.

        Prameters:
        - contours (np.ndarray): Contours found in the input image.

        Returns:
        - np.ndarray: The edge image of the input image.
        """
        # cv2.drawContours(self.reference_image, [largest_contour], -1, (0, 255, 0), 3)
        # draw the largest contour on the image
        hull: np.ndarray = cv2.convexHull(largest_contour)
        # draw the hull on the image
        mask: np.ndarray = np.zeros_like(image)
        cv2.fillPoly(mask, [hull], (255, 255, 255))
        # mask = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2))
        # cv2.imshow("Mask", mask)
        # Calculate the difference between adjacent pixels
        diff: np.ndarray = cv2.Sobel(mask, cv2.CV_64F, 1, 1, ksize=15)
        edge: np.ndarray = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)[1] # fine tune the threshold

        # cv2.imshow("Edge", edge)
        # cv2.imshow("Largest Contour", image)
        return edge


class ContourPipeLine:
    """
    The ContourPipeLine class is responsible for finding the contours in the input image.
    It can be used in serial or parallel with other edge finders.
    """

    def __init__(
        self,
        thresh_bounds: Tuple[int, int] = (THRESH_LOWER_BOUND, THRESH_UPPER_BOUND),
        ksize: Tuple[int, int] = (9, 9),
        sigma_x: int = 0,
        device: str = 'cpu'
    ) -> None:
        """
        Initialize the ContourPipeLine instance

        Parameters:
        - thresh_bounds: Tuple[int, int]: the threshold bounds of the contour
        - ksize: Tuple[int, int]: the kernel size of the gaussian blur
        - sigma_x: int: the sigma x of the gaussian blur
        """
        self.thresh_bounds: Tuple[int, int] = thresh_bounds
        self.ksize: Tuple[int, int] = ksize
        self.sigma_x: int = sigma_x
        if device == 'gpu':
            self.draw_contour_method = self._draw_contour_on_gpu
        else:
            self.draw_contour_method = self._draw_contour_on_cpu

    def get_largest_contour(self, image: Union[np.ndarray, cv2.cuda_GpuMat]) -> np.ndarray:
        """
        Finds the largest contour in the input image.

        Parameters:
        - image: Union[np.ndarray, cv2.cuda_GpuMat]: The input image for finding the largest contour.

        Returns:
        - np.ndarray: The largest contour found in the input image.

        Raises:
        - NoImageException: If the input image is None.
        """
        contours: np.ndarray = self._find_contours(image)
        if contours is None or len(contours) == 0:
            raise NoContourException()
        largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
        self.draw_contour_method(largest_contour, image)
        return largest_contour

    __call__ = get_largest_contour # alias

    def _draw_contour_on_gpu(self, contour: np.ndarray, gpu_image: cv2.cuda_GpuMat) -> None:
        """
        Draws the contour on the input image in the gpu.

        Parameters:
        - contour: np.ndarray: The contour to draw on the image.
        - gpu_image: cv2.cuda_GpuMat: The image to draw the contour on.

        Raises:
        - NoImageException: If the input image is None.

        Returns:
        - None
        """
        # Download the image to CPU memory
        image_cpu = gpu_image.download()
        # Draw the contours on the CPU image
        cv2.drawContours(image_cpu, [contour], -1, (0, 255, 0), 3)
        # Upload the image back to GPU memory
        gpu_image.upload(image_cpu)

    def _draw_contour_on_cpu(self, contour: np.ndarray, image: np.ndarray) -> None:
        """
        Draws the contour on the input image in the cpu.

        Parameters:
        - contour: np.ndarray: The contour to draw on the image.
        - image: np.ndarray: The image to draw the contour on.

        Raises:
        - NoImageException: If the input image is None.

        Returns:
        - None
        """
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)

    def _find_contours(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the contours in the input image.

        Parameters:
        - image: np.ndarray: The input image for finding contours.

        Returns:
        - np.ndarray: Contours found in the input image.

        Raises:
        - NoImageException: If the input image is None.
        """
        if image is None:
            raise NoImageException()
        # gaussian blur the image
        blurred_image: np.ndarray = cv2.GaussianBlur(image, self.ksize, self.sigma_x)
        blurred_image = ImageProcessing.image_filtering_open(blurred_image)
        # threshold the image
        thresh: np.ndarray = cv2.threshold(
            blurred_image,
            self.thresh_bounds[0],
            self.thresh_bounds[1],
            cv2.THRESH_BINARY
        )[1] # fine tune the threshold IM
        # find the contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class HoughPipeLine:
    """
    The HoughPipeLine class is responsible for hough line transformation.
    It can be used in serial or parallel with other edge finders.
    """

    def __init__(
        self,
        angle_bounds: Tuple[int, int] = (HOUGH_ANGLE_LOWER_BOUND, HOUGH_ANGLE_UPPER_BOUND),
        edges_bounds: Tuple[int, int] = (EDGES_LOWER_BOUND, EDGES_UPPER_BOUND),
        hough_confident_threshold: int = HOUGH_CONFIDENT_THRESHOLD,
        device: str = "cpu"
    ) -> None:
        """
        Initialize the HoughPipeLine instance

        Parameters:
        - angle_bounds: Tuple[int, int]: the angle bounds of the hough line
        - edges_bounds: Tuple[int, int]: the edges bounds of the hough line
        - hough_confident_threshold: int: the confident threshold of the hough line
        - device: str: the device to use for hough line detection
        """
        self.device: str = device
        self.angle_bounds: Tuple[int, int] = angle_bounds
        self.edges_bounds: Tuple[int, int] = edges_bounds
        self.hough_confident_threshold: int = hough_confident_threshold
        self.prev_x1 = 0 # the x1 of the found edge on the previous frame
        if device == "gpu":
            self.get_hough_lines = self._get_hough_lines_on_gpu
            # create a canny edge detector and hough lines detector in the gpu
            self.hough_lines_detector: cv2.cuda_HoughLinesDetector = \
            cv2.cuda.createHoughLinesDetector(
                1, np.pi/180, self.hough_confident_threshold
            )
            self.canny_edge_detector: cv2.cuda_CannyEdgeDetector = \
            cv2.cuda.createCannyEdgeDetector(
                self.edges_bounds[0],
                self.edges_bounds[1],
                apertureSize=3
            )
        else:
            self.get_hough_lines = self._get_hough_lines_on_cpu

    def _handle_draw_hough_lines(
            self,
            image: Union[np.ndarray, cv2.cuda_GpuMat],
            point_1: tuple,
            point_2: tuple
        ) -> None:
        """
        Draws the hough lines on the input image.

        Parameters:
        - image: Union[np.ndarray, cv2.cuda_GpuMat]: The input image for hough line detection.
        - point_1: tuple: The first point of the line.
        - point_2: tuple: The second point of the line.
        """
        if self.device == "cpu":
            cv2.line(image, point_1, point_2, (0, 0, 255), 3)
        else:
            image_cpu: np.ndarray = image.download()
            cv2.line(image_cpu, point_1, point_2, (0, 0, 255), 3)
            image.upload(image_cpu)

    def get_hough_lines_image(self, image: Union[np.ndarray, cv2.cuda_GpuMat]) -> None:
        """
        Gets the Hough line image from the input image.

        Parameters:
        - image: Union[np.ndarray, cv2.cuda_GpuMat]: The input image for hough line detection.

        Returns:
        - np.ndarray: Hough line image for line following.

        Raises:
        - NoImageException: If the input image is None.
        """
        lines: np.ndarray = self.get_hough_lines(image)
        if lines is None:
            return

        # calculate the line parameters
        params: list[list[int]] = [[], [], [], []] # for valid lines
        for line in lines:
            result: tuple = self._get_line_params(line)
            if result is None or (self.prev_x1 and abs(result[0] - self.prev_x1) > 15):
                continue
            # draw the line on the image
            for i, value in enumerate(result):
                params[i].append(int(value))

        if len(params[0]) > 0:
            np_params = np.array(params, dtype=int)
            x1, y1, x2, y2 = map(int, np.mean(np_params, axis=1))
            self.prev_x1 = x1 # update prev x1
            self._handle_draw_hough_lines(image, (x1, y1), (x2, y2))
            # cv2.line(self.reference_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        else:
            self.prev_x1 = 0 # there is a sudden edge change
        # cv2.imshow("HoughLine", image)

    __call__ = get_hough_lines_image # alias

    def _get_hough_lines_on_gpu(self, gpu_image: cv2.cuda_GpuMat) -> np.ndarray:
        """
        Get the hough lines from the input image in the gpu

        Parameters:
        - gpu_image: cv2.cuda_GpuMat: The input image for hough line detection

        Returns:
        - np.ndarray: The hough lines detected in the image
        """
        if gpu_image is None:
            raise NoImageException()

        # apply the canny edge detector and hough lines detector
        edges = self.canny_edge_detector.detect(gpu_image)
        lines = self.hough_lines_detector.detect(edges)
        return lines.download()

    def _get_hough_lines_on_cpu(self, image: np.ndarray) -> np.ndarray:
        """
        Get the hough lines from the input image in the cpu

        Parameters:
        - image: np.ndarray: The input image for hough line detection

        Returns:
        - np.ndarray: The hough lines detected in the image
        """
        if image is None:
            raise NoImageException()

        edges: np.ndarray = cv2.Canny(
            image,
            self.edges_bounds[0],
            self.edges_bounds[1],
            apertureSize=3
        ) # fine tune the threshold
        return cv2.HoughLines(edges, 1, np.pi/180, self.hough_confident_threshold)

    def _get_line_params(self, line: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate the intersect point for each line

        Parameters:
        - line: np.ndarray: line generated by hough transformation

        Returns:
        - Tuple[int, int, int, int]: the coordinates of the line
        """
        rho: float = line[0][0]
        theta: float = line[0][1]
        angle: float = theta * 180 / np.pi
        if self.angle_bounds[0] <= angle <= self.angle_bounds[1]:
            a: float = np.cos(theta)
            b: float = np.sin(theta)
            x0: float = a * rho
            y0: float = b * rho
            x1: int = int(x0 + 1000 * (-b))
            y1: int = int(y0 + 1000 * (a))
            x2: int = int(x0 - 1000 * (-b))
            y2: int = int(y0 - 1000 * (a))
            # cv2.line(self.reference_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            return x1, y1, x2, y2
        return None
