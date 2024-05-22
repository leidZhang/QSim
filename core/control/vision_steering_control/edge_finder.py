import cv2
import numpy as np

from hal.utilities.image_processing import ImageProcessing

from .exceptions import NoImageException, NoContourException
from .constants import HOUGH_ANGLE_LOWER_BOUND, HOUGH_ANGLE_UPPER_BOUND
from .constants import EDGES_LOWER_BOUND, EDGES_UPPER_BOUND, HOUGH_CONFIDENT_THRESHOLD
from .constants import THRESH_LOWER_BOUND, THRESH_UPPER_BOUND

class TraditionalEdgeFinder: 
    """
    The TraditionEdgeFinder class is responsible for finding the edges of the road using traditional computer vision techniques.

    Attributes:
    - image_width: int: The width of the image
    - image_height: int: The height of the image

    Methods:
    - preprocess_image: Preprocesses the input image for line following.
    - get_houghline_image: Gets the Hough line image from the input image.
    - find_contours: Finds the contours in the input image.
    - get_edge_image: Gets the edge image from the input image and contours.
    - execute: Executes the traditional edge finder.

    Raises:
    - NoImageException: If the input image is None.
    - NoContourException: If the contours are None or empty.
    """

    def __init__(self, image_width: int = 820, image_height: int = 410) -> None:
        """
        Initializes the TraditionEdgeFinder object.

        Parameters:
        - image_width: int: The width of the image
        - image_height: int: The height of the image

        Returns:
        - None
        """
        self.image_width: int = image_width
        self.image_hight: int = image_height

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input image for line following.

        Parameters:
            image (np.ndarray): Input image for line following.

        Returns:
            np.ndarray: Preprocessed image for line following.
        """
        # check if the image is None
        if image is None: 
            raise NoImageException()
        # crop the image
        self.image: np.ndarray = image.copy() 
        cropped_image: np.ndarray = image[220:360, 100:]
        # convert the image to grayscale
        gray_image: np.ndarray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        return gray_image
    
    def get_houghline_image(self, grey_image: np.ndarray) -> np.ndarray:
        """
        Gets the Hough line image from the input image.

        Parameters:
        - grey_image (np.ndarray): Greyscale image for line following.

        Returns:
        - np.ndarray: Hough line image for line following.
        """
        if grey_image is None: 
            raise NoImageException()
        # thresholds of the line angles
        min_angle: float = HOUGH_ANGLE_LOWER_BOUND
        max_angle: float = HOUGH_ANGLE_UPPER_BOUND
        edges: np.ndarray = cv2.Canny(
            grey_image, 
            EDGES_LOWER_BOUND, 
            EDGES_UPPER_BOUND, 
            apertureSize=3
        ) #fine tune the threshold
        lines: np.ndarray = cv2.HoughLines(edges, 1, np.pi/180, HOUGH_CONFIDENT_THRESHOLD)
        if lines is None: 
            # cv2.imshow("HoughLine", grey_image)
            return grey_image 
        # calculate the line parameters
        for line in lines: 
            rho: float = line[0][0]
            theta: float = line[0][1]
            angle: float = theta * 180 / np.pi
            if min_angle <= angle <= max_angle: 
                a: float = np.cos(theta)
                b: float = np.sin(theta)
                x0: float = a * rho
                y0: float = b * rho
                x1: int = int(x0 + 1000 * (-b))
                y1: int = int(y0 + 1000 * (a))
                x2: int = int(x0 - 1000 * (-b))
                y2: int = int(y0 - 1000 * (a))
                # draw the line on the image
                cv2.line(grey_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imshow("HoughLine", self.image)
        return grey_image
    
    def find_conturs(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the contours in the input image.

        Parameters:
        - image (np.ndarray): Input image for line following.

        Returns:
        - np.ndarray: Contours found in the input image.
        """
        if image is None: 
            raise NoImageException()
        # gaussian blur the image
        blurred_image: np.ndarray = cv2.GaussianBlur(image, (9, 9), 0)
        blurred_image = ImageProcessing.image_filtering_open(blurred_image)
        # threshold the image
        thresh: np.ndarray = cv2.threshold(
            blurred_image, 
            THRESH_LOWER_BOUND, 
            THRESH_UPPER_BOUND, 
            cv2.THRESH_BINARY
        )[1] # fine tune the threshold IM
        # find the contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        return contours
    
    def get_edge_image(self, image: np.ndarray, contours: np.ndarray) -> np.ndarray:
        """
        Gets the edge image from the input image and contours.

        Prameters:
        - image (np.ndarray): Input image for line following.
        - contours (np.ndarray): Contours found in the input image.

        Returns:
        - np.ndarray: Edge image for line following.
        """
        if image is None or contours is None or len(contours) == 0: 
            raise NoContourException()
        # find the largest contour
        largest_contour: np.ndarray = max(contours, key=cv2.contourArea)
        # cv2.drawContours(self.image, [largest_contour], -1, (0, 255, 0), 3)
        # draw the largest contour on the image
        hull: np.ndarray = cv2.convexHull(largest_contour)
        # draw the hull on the image
        mask: np.ndarray = np.zeros_like(image)
        cv2.fillPoly(mask, [hull], (255, 255, 255))
        # cv2.imshow("Mask", mask)
        # Calculate the difference between adjacent pixels
        diff: np.ndarray = cv2.Sobel(mask, cv2.CV_64F, 1, 1, ksize=15)
        edge: np.ndarray = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)[1] # fine tune the threshold
        # cv2.imshow("Edge", edge)
        # cv2.imshow("Largest Contour", self.image)
        return edge
    
    def execute(self, original_image: np.ndarray) -> tuple: 
        """
        Executes the traditional edge finder.

        Parameters:
        - original_image (np.ndarray): Original image for line following.

        Returns:
        - tuple: Edge image for line following.
        """
        grey_image: np.ndarray = self.preprocess_image(original_image)
        houghline_image: np.ndarray = self.get_houghline_image(grey_image)
        contours: np.ndarray = self.find_conturs(houghline_image)
        edge_image: np.ndarray = self.get_edge_image(original_image, contours)
        result: tuple = ImageProcessing.find_slope_intercept_from_binary(binary=edge_image)

        return result