import cv2 as cv
import glob
import os
import re

png_file_paths = glob.glob(r"/workspace/AF-SfMLearner/*.png")
for i, png_file_path in enumerate(png_file_paths):
    jpg_file_path = png_file_path[:-3] + "jpg"
   
    # Load .png image
    image = cv.imread(png_file_path)

    # Save .jpg image
    cv.imwrite(jpg_file_path, image, [int(cv.IMWRITE_JPEG_QUALITY), 100])

    pass