import cv2

from core.utils.io_utils import ImageReader

if __name__ == "__main__":
    reader: ImageReader = ImageReader('images/')
    reader.read_images()