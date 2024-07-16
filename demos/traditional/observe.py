# python imports
import cv2
import time
import math
from queue import Queue
from typing import Any
import numpy as np
# custom imports


class SignalFilter:
    """
    A decaying sliding window filter for the traffic light signal detection.

    Attributes:
        buffer_size (int): Size of the buffer for the filter.
        buffer (Queue): Buffer for the filter.
        accumulator (float): Accumulator for the filter.
        decay_factor (float): Decay factor for the filter.
        threshold (float): Threshold for the filter.

    Methods:
        __call__(signal: float) -> str:
            Filters the signal and returns the traffic light status.
        clear() -> None:
            Clears the buffer and the accumulator for the filter.
    """

    def __init__(self, threshold: float, buffer_size: int) -> None:
        """
        Initializes the SignalFilter class with the specified threshold and buffer size.

        Parameters:
            threshold (float): Threshold for the filter.
            buffer_size (int): Size of the buffer for the filter.

        Returns:
            None
        """
        self.buffer_size: int = buffer_size
        self.buffer: Queue = Queue(self.buffer_size)
        self.accumulator: float = 0
        self.decay_factor: float = 0.95
        self.threshold: float = threshold

    def __call__(self, signal: float) -> str:
        """
        The call method for the SignalFilter class.

        Parameters:
            signal (float): Signal value for the filter.

        Returns:
            str: Traffic light status for the signal.
        """
        if self.buffer.full():
            self.accumulator -= round(self.buffer.get() * math.pow(self.decay_factor, self.buffer_size - 1), 10)
        self.buffer.put(signal)
        self.accumulator = round(self.accumulator * self.decay_factor + signal, 10)
        result: float = abs(self.accumulator) / self.buffer.qsize()
        # print(round(result, 10))
        if round(result, 10) > self.threshold:
            return "red light"
        return "green light"
    
    def clear(self) -> None:
        """
        Clears the buffer and the accumulator for the filter.

        Returns:
            None
        """
        self.buffer = Queue(self.buffer_size)
        self.accumulator = 0


class RawImagePipeline(object):
    def __init__(self, crop_size : tuple =(100,100), yellow_low_thresh_frame :tuple =(20, 100, 100),
                 yellow_high_thresh_frame : tuple = (30, 255, 255),
                 red_low_thresh_sign1 :tuple= (0,50,50),red_high_thresh_sign1:tuple = (10,255,255)
                ,red_low_thresh_sign2 :tuple= (170,50,50) ,red_high_thresh_sign2:tuple = (180,255,255), ratio_thresh:float = np.pi) -> None:
        '''
        Parameters
        ----------
        crop_size : tuple
            `(h,w)` the size of the image that will be returned. The stop sign or traffic light will
            be immersed in an empty image of this size. The size is in row, column order a.k.a (y, x) order in image plane
        yellow_low_thresh_frame : tuple
            `(x1, x2, x3)` the lower threshold values for yellow on the traffic light
        yellow_high_thresh_frame : tuple
            `(x1, x2, x3)` the higher threshold values for yellow on the traffic light
        red_low_thresh_sign1: tuple
            `(x1, x2, x3)` the threshold ranges for red, there are 4 values
        red_low_thresh_sign2: tuple
            `(x1, x2, x3)` the threshold ranges for red, there are 4 values
        red_high_thresh_sign1: tuple
            `(x1, x2, x3)` the threshold ranges for red, there are 4 values
        red_high_thresh_sign2: tuple
            `(x1, x2, x3)` the threshold ranges for red, there are 4 values
        ratio_thresh: float
            A tolerance value for the ratio , which is used to dtermine how circly an shape is
        Returns
        -------
        None'''
        self.crop_size = crop_size
        self.yellow_low_thresh_frame = yellow_low_thresh_frame
        self.yellow_high_thresh_frame = yellow_high_thresh_frame
        self.red_low_thresh_sign1 = red_low_thresh_sign1
        self.red_high_thresh_sign1 = red_high_thresh_sign1
        self.red_low_thresh_sign2 = red_low_thresh_sign2
        self.red_high_thresh_sign2 = red_high_thresh_sign2
        self.ratio_thresh = ratio_thresh
        self.area_arc_constant = 4*np.pi
        self.set_constants()
        self.found_flag = False
        self.bound_area = 0.0
        self.detected_area = 0.0

    def set_constants(self):
        '''Sets some constants. This architetcure allows to reinitialize constants after manually modifying a
        parameter
        Paramaters
        ----------
        Returns
        -------
        '''
        self.window = np.zeros((*self.crop_size, 3), dtype = np.uint8)
        self.center = (self.crop_size[0]//2, self.crop_size[1]//2)
        self.windowshape = self.window.shape

    def __call__(self,img:np.ndarray, slice_image:bool = True) -> np.ndarray:
        '''__call__ will perform the main logic of the pipeline

        ----------
        img: np.ndarray
            input image
        slice_image: bool
            slice the input image, so that only relevant portions of the image isused.
            This reduces the false positives and improves run time
        Returns
        img: np.ndarray
            cropped image immersed in the window with the size self.crop_size (default =(100,100))
        -------
        '''
        if slice_image:
            img_ = self.merge_with_window(self.segment_stop_sign(img[:img.shape[0]//2:, img.shape[1]//2:,:]))
            if self.found_flag:
                self.found_flag = False
                return img_ ,'stop'

            img_ = self.merge_with_window(self.segment_traffic_light(img[:img.shape[0]//2:, :,:]))
            if self.found_flag:
                retval = 'traffic'
            else:
                retval = 'unknown'
            self.found_flag = False

            return img_, retval
        else:
            img_ = self.merge_with_window(self.segment_stop_sign(img))
            if self.found_flag:
                self.found_flag = False
                return img_ ,'stop'
            img_ = self.merge_with_window(self.segment_traffic_light(img))
            if self.found_flag:
                retval = 'traffic'
            else:
                retval = 'unknown'

            self.found_flag = False
            return img_, retval

    def __setattr__(self, name: str, value: Any) -> None:
        '''Overlaoded setattr to reflect parameter modifications
        Paramaters
        ----------
        name: str
            name of the attribute
        value: Any
            The desired value for this attribute

        Returns
        -------
        '''
        if name == 'crop_size':
            super().__setattr__(name, value)
            self.set_constants()
        else:
            super().__setattr__(name, value)

    def segment_traffic_light(self, img:np.ndarray)->np.ndarray:
        '''Segments and crops the traffic light
        Paramaters
        ----------
        img: np.ndarray
            Image with the traffic light
        Returns
        -------
        img: np.ndarray
            Cropped image with only traffic light'''
        img_original = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img, self.yellow_low_thresh_frame, self.yellow_high_thresh_frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), dtype= np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contours = max(contours, key = cv2.contourArea)
            self.detected_area = cv2.contourArea(contours)
            # return only those pass teh area thershold.
            # Some lights aere not relevant (that is they are really far away and theier features are hard to distinguish)
            if 300 < self.detected_area:
                # print(f"Frame area: {self.detected_area}")
                x, y, w, h = cv2.boundingRect(contours)
                img =  img_original[y:(y+h),x:(x+w),:]
                self.found_flag = True
                return img
        # return default
        self.found_flag = False
        return self.window

    def segment_stop_sign(self, img:np.ndarray) ->np.ndarray:
        '''Segments and crops the stop sign
        Paramaters
        ----------
        img: np.ndarray
            Image with the stop sign
        Returns
        -------
        img: np.ndarray
            Cropped image with only sign'''
        img_original = img.copy()
        img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask0 = cv2.inRange(img_hsv, self.red_low_thresh_sign1, self.red_high_thresh_sign1)
        mask1 = cv2.inRange(img_hsv,self.red_low_thresh_sign2, self.red_high_thresh_sign2)
        mask = cv2.bitwise_or(mask0,mask1)
        mask =  cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10,10),np.uint8))
        cnt,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnt) > 0:
            # if we have a contour, it means we have some object with a red color, and since the image is
            # only of the right half, this will be 99.99% the stop sign
            for cnt_ in cnt:
                a_l = cv2.arcLength(cnt_, True)
                a_l_2 = a_l**2
                a_ = cv2.contourArea(cnt_)
                a_ = max(a_,0)
                if a_ < 1200: # csi: 950: # rgbd: 1300:
                    continue
                # ratio of area to arclength of a circle is : (4*pi^2*r^2)/pi*r^2 = 4*pi
                # area based filtering, we dont want contours with high area or less area
                ratio = a_l_2 / a_
                if ratio> self.area_arc_constant-self.ratio_thresh and ratio<self.area_arc_constant+self.ratio_thresh and a_> 200:
                    x, y, width, height = cv2.boundingRect(cnt_)
                    self.found_flag = True
                    return img_original[y:y+height, x:x+width]
        self.found_flag = False
        return self.window

    def determine_if_true_stop(self, img:np.ndarray, x:float, y:float, width:float, height:float)->bool:
        '''Verifies the detected stop sign is a true positive, by utilizing a thresholding
        Paramaters
        ----------
        img: np.ndarray
            Image with the stop sign
        x: float
            the x coordinate of the stop sign rectangle
        y: float
            the y coordinate of the stop sign rectangle
        width: float
            the width of the rectangle
        height: float
            the height of the rectangle
        Returns
        -------
        dec: bool
            bool decision wether or not this detection was a true p'''
        # img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # mask0 = cv2.inRange(img_hsv, self.red_low_thresh_sign1, self.red_high_thresh_sign1)
        # mask1 = cv2.inRange(img_hsv,self.red_low_thresh_sign2, self.red_high_thresh_sign2)
        # mask = cv2.bitwise_or(mask0,mask1)
        # detected_mask = np.zeros_like(mask)
        # cv2.rectangle(detected_mask, (x, y), (x+width, y+height), (255,255,255), -1)
        # final_mask = cv2.bitwise_and(mask, detected_mask)
        # cnt,_ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # x_s = 0
        # y_s = 0
        # w_s = 0
        # h_s = 0
        # if len(cnt)>0:
        #     x_s, y_s, w_s, h_s = cv2.boundingRect(cnt[0])

        # return  np.sum(final_mask) > 20 and (w_s > 20) and (h_s> 20)
        ...

    def merge_with_window(self, img :np.ndarray)->np.ndarray:
        '''Merges a segmented image with the window. Specifically, this will immerse the segment into the window.
        The segment will be ceneterd at the window,
        and if the segment exceeds the window size, then the overflowing pixels will be ignored
        Paramaters
        ----------
        img: np.ndarray
            Segemented image
        Returns
        -------
        img: np.ndarray
            Merged image
        '''
        imgshape = img.shape
        #case 1, window is larger than the segment
        if all(np.array(self.window.shape)-np.array(imgshape)>=0):
            imgdx = imgshape[1]//2
            imgdy = imgshape[0]//2
            result = self.window.copy()
            topleftcorner = np.array(self.center) - np.array((imgdy, imgdx))
            result[topleftcorner[0]:topleftcorner[0]+imgshape[0], topleftcorner[1]:topleftcorner[1]+imgshape[1], :] = img
            return result

        #case2, img is larger than window
        else:
            imgdx = imgshape[1]//2
            imgdy = imgshape[0]//2
            result = self.window.copy()
            topleftcorner = np.array((imgdy, imgdx)) - np.array(self.center)
            result = img[topleftcorner[0]:topleftcorner[0]+self.windowshape[0], topleftcorner[1]:topleftcorner[1]+self.windowshape[1], :]
            return result

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#class for performing traffic identification
#The raw image should be fed into this class.
from typing import Union, Tuple, Callable
from skimage.transform import pyramid_expand
import matplotlib.pyplot as plt


class Identity:
    '''Identity function'''
    def __call__(self, sample:Any) -> Any:
        return sample


class WrapCallableDevice:
    def __init__(self, callable_method:Callable, device:str) -> None:
        '''Wraps callable such that their output tensors returns the Tensor in the device
        Parameters
        ----------
        callable_method: Callable
            Callable function that will return a Tensor
        device: Union[str, torch.device]
            device to wrap to'''
        self.callable_method = callable_method
        self.device = device

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.callable_method(*args, **kwds).to(self.device)


class Compose:
    def __init__(self, *args) -> None:
        '''Composer for compunding multiple callables
        Parameters
        ----------
        args: Variadic number of callables
        '''
        self.callables = args

    def __call__(self, input: Any) -> Any:
        '''Returns the input after passing through the pipeline
        Parameters
        ----------
        input : Any
            Anything
        Returns
        -------
        output : Any
            Output after passed through the pipeline'''
        for calli in self.callables:
            input = calli(input)
        return input


class DecisionMaker:
    def __init__(self, stop_sign_ignore_interval:int = 5, classic_traffic_pipeline:bool = False,
                  network_class:Any = None, input_preprocess:Any = Identity(), output_postprocess:Any = Identity(),
                  weights_file:str=None, device:str = 'cpu') -> None:
        '''Parameters
           ----------
           stop_sign_ignore_interval: int
                The inetrval within which any newly detected stop sign is ignore. This enables
                us to prevent detecting the same stop sign again and again
           classic_traffic_pipeline: bool
                If true, then the traffic light detection is based on thresholding in rgb space
                If False, then neural network based detector is used. The weights should be in a file named
                traffic_weights.qcar, and the Network class should be specified
           network_class: Any
                The class of the network model
           input_preprocess: Any
                The preprocessing that needs to be performed on the segmented traffic image.
                If None, segmented image will be directly passed to
                network
           output_postprocess: Any
                The postprocessing that needs to be done with the output
           weights_file : str
                The file with the weights to load
           device : Union[str, torch.device]
                Device in which the model will run
                '''
        self.pipeline = RawImagePipeline()
        self.filter = SignalFilter(0, 10)
        self.stop_sign_ignore_interval = stop_sign_ignore_interval
        self.pointer = -1
        self.dist_coeffs = [1200, 1200]# [1.25, 2.5]
        if classic_traffic_pipeline:
            # Final vars. dont touch these. Touch them at your own risk
            print("Using classical method")
            self.lower = (225/255, 40/255,10/255)#(214, 19,0)#np.array([7, 218, 223])
            self.upper =  (255/255, 78/255, 35/255)#(255, 112, 30)#np.array([15, 255, 255])
            # self.output_postprocess = 0.0
        self.classic_traffic_pipeline = classic_traffic_pipeline
        self.has_horizontal_line = False
        self.has_horizontal_line_decay_time = time.time()
        self.detection_flags = {
            'stop_sign': False,
            'horizontal_line': False,
            'red_light': False, # True if traffic light is red
            'unknown_error': False
        }
        self.last_stopped_time = time.time()

    def thesh_horizontal_line(self, low_bottom_image:np.ndarray) ->bool:
        '''
        Thresholds and detects horiontal line. This pipeline only uses the bottom half of the image
        This pipeline performs canny edge detection, followed by Hough line detection with theta = 180 degrees
        Parameters
        ---------
        low_bottom_image: np.ndarray
            Lower bottom of an Image
        Returns
        -------
        line_exist : bool
            Wether or not a line was detected
        '''
        gray_image = cv2.cvtColor(low_bottom_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 90, None, 250, 5)
        lines: np.ndarray = cv2.HoughLines(edges, 1, np.pi/180, 98)
        # print(lines)
        if lines is not None:
            self.has_horizontal_line = False
            self.has_horizontal_line_decay_time = time.time()
            for line in lines:
                rho: float = line[0][0]
                theta: float = line[0][1]
                angle: float = theta * 180 / np.pi
                if 75 <= angle <= 115:
                    a: float = np.cos(theta)
                    b: float = np.sin(theta)
                    x0: float = a * rho
                    y0: float = b * rho
                    x1: int = int(x0 + 1000 * (-b))
                    y1: int = int(y0 + 1000 * (a))
                    x2: int = int(x0 - 1000 * (-b))
                    y2: int = int(y0 - 1000 * (a))
                    # draw the line on the image
                    cv2.line(low_bottom_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # print(f'horizontal detected!')
                    self.has_horizontal_line = True
                    # print(rho)
                # else:
                    # print(f'horizontal not detected!')
                # print("=====================")
        else:
            # detetced horizontal line, but time has passed
            if time.time() - self.has_horizontal_line_decay_time > 0.25:
                self.has_horizontal_line = False
        # cv2.imshow('Images', low_bottom_image)

    def __call__(self, img:np.ndarray) -> Tuple[bool, int]:
        '''
        Returns True if need to stop else False, Also returns the time to sleep
        Parameters
        ----------
        img : np.ndarray
            Input image. Should be raw, unprocessed.
        Returns
        -------
        decision : bool
            Stop or not
        sleeptime : int
            Time to sleep in seconds
        '''
        img_back = img[img.shape[0]*7//10:,:,:].copy()
        # cv2.imshow('Images2_Raw', img)
        img, flag = self.pipeline(img)
        # cv2.imshow('Images2_Contour', img)
        # print(
        #     f"Bound: {self.dist_coeffs[self.pointer]}, \
        #     Current: {self.pipeline.detected_area}, \
        #     Pointer: {self.pointer}, \
        #     Pass: {self.dist_coeffs[self.pointer] <= self.pipeline.detected_area}"
        # )
        if flag =='unknown':
            self.filter.clear()
            self.pipeline.bound_area = 0.0
            self.pipeline.detected_area = 0.0
            self.detection_flags["horizontal_line"] = False
            # print(self.detection_flags)
            return 0
            # return False, 0
        elif flag == 'stop':
            self.detection_flags["horizontal_line"] = False
            if time.time() - self.last_stopped_time >= self.stop_sign_ignore_interval:
                self.last_stopped_time = time.time()
                self.detection_flags["stop_sign"] = True
                # print("Ignore stop sign")
                return 
            # print("Found stop sign")
            self.detection_flags["stop_sign"] = False
            return 1
        elif flag=='traffic':
            self.thesh_horizontal_line(img_back)
            if self.has_horizontal_line:
                self.detection_flags["horizontal_line"] = self.has_horizontal_line
                if self.pipeline.bound_area == 0.0:
                    self.pipeline.bound_area = self.pipeline.detected_area
                    self.pointer = (self.pointer + 1) % 2
                # print("Enter control zone")
            # self.detection_flags["horizontal_line"] = False
            # print(f"detected: {self.pipeline.detected_area}, bound: {self.pipeline.bound_area}")
            current_area = self.pipeline.detected_area
            bound_area = self.pipeline.bound_area
            if current_area >= self.dist_coeffs[self.pointer] or self.pipeline.bound_area == 0: # self.dist_coeffs[self.pointer]:
                self.detection_flags["horizontal_line"] = False
                self.detection_flags["red_light"] = False
                # print("Pass control zone")
                # print(self.detection_flags)
                return 0

            # print(self.classic_traffic_pipeline)
            if self.classic_traffic_pipeline: # and self.detection_flags["horizontal_line"]:
                mask = cv2.inRange(pyramid_expand(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channel_axis= 2), self.lower, self.upper)
                signal: float = mask.sum() / 255
                # push signal to the filter
                if signal > 0: # original is > 5
                    result: str = self.filter(1)
                else:
                    result: str = self.filter(0)
                # print(result)
                # get the result from the filter
                if result == "green light": # and signal < 1:
                    self.detection_flags["red_light"] = False
                    # print(self.detection_flags)
                    return 0
                else:
                    self.detection_flags["red_light"] = True
                    # print(self.detection_flags)
                    return 2
            else:
                # cv2.imshow('traffic', img)
                # cv2.waitKey(1)
                # with torch.no_grad():
                #     dec = self.output_postprocess(self.net(self.input_preprocess(img)))
                #     dec = dec==1
                # self.detection_flags["unknown_error"] = dec
                # print("unknown error")
                return 0
