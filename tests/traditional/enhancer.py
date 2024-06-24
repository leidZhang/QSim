import cv2
import numpy as np

class CSIImageEnhancer:
    def __init__(self):
       
        self.img = None
        self.enhanced_img = None

    def noise_reduction(self):
        self.img = cv2.GaussianBlur(self.img, (5, 5), 0)

    def histogram_equalization(self):
        ycrcb = cv2.cvtColor(self.img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        self.img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def sharpening(self):
        sharpening_kernel = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        self.img = cv2.filter2D(self.img, -1, sharpening_kernel)

    def white_balance(self):
        result = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        self.img = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    def enhance(self, img):
        self.img = img
        self.noise_reduction()
        self.histogram_equalization()
        #self.sharpening()
        self.white_balance()
        self.enhanced_img = self.img
        return self.enhanced_img