import sys
import cv2
import numpy as np
from plantcv import plantcv as pcv

class Transformation:
    def __init__(self, filepath):
        """
        Initializes the Transformation class by loading the image.
        """
        self.filepath = filepath
        # load the image
        self.img, self.path, self.filename = pcv.readimage(filename=self.filepath)

    def apply_gaussian_mask(self):
        """
        Applies the Gaussian mask logic based on saturation.
        """
        print("\n--- Running Gaussian Mask Process ---")
        # grayscale the image based on saturation
        gray_img_s = pcv.rgb2gray_hsv(rgb_img=self.img, channel='s')

        # cut out the grayscaled image and turn the cutout parts black
        mask_img_s = pcv.threshold.binary(gray_img=gray_img_s, threshold=100, object_type='light')
        
        # add a gaussian blur on the image
        gaussian_mask_s = pcv.gaussian_blur(img=mask_img_s, ksize=(5, 5))

        # show the images
        print("Plotting Original Image and Saturation-based Gaussian Mask...")
        pcv.plot_image(self.img)
        pcv.plot_image(gaussian_mask_s)
        
        return gaussian_mask_s

    def apply_rotten_mask(self):
        """
        Applies the mask for showing the rotten parts based on hue and LAB space.
        """
        print("\n--- Running Rotten Parts Mask Process ---")
        # grayscale the image based on hue
        gray_img_h = pcv.rgb2gray_hsv(rgb_img=self.img, channel='h')

        # cut out the grayscaled image and turn the cutout parts black
        mask_img_h = pcv.threshold.binary(gray_img=gray_img_h, threshold=50, object_type='dark')
        
        # add a gaussian blur on the image
        gaussian_mask_h = pcv.gaussian_blur(img=mask_img_h, ksize=(11, 11))

        # 1. Get the 'b' channel from L*a*b* color space
        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_img)

        # 2. Threshold the 'b' channel: pixels > 200 become white
        _, b_thresh = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)

        # 3. Combine gaussian_blur mask with the b_thresh mask
        bs_mask = cv2.bitwise_or(gaussian_mask_h, b_thresh)

        # 4. Apply this final mask to the original image
        # Create a white background
        background = np.full(self.img.shape, 255, dtype=self.img.dtype)

        # Get the inverse of the mask
        mask_inv = cv2.bitwise_not(bs_mask)

        # Get foreground (leaf) and background parts
        foreground = cv2.bitwise_and(self.img, self.img, mask=bs_mask)
        background_part = cv2.bitwise_and(background, background, mask=mask_inv)

        # Combine them to get the final image
        final_image = cv2.add(foreground, background_part)

        print("Plotting Final Image with Rotten Parts Masked...")
        pcv.plot_image(final_image)
        
        return final_image

    def process(self):
        """
        Runs both the Gaussian mask and the rotten parts mask sequentially.
        """
        if self.img is None:
            print(f"Error: Could not load image from {self.filepath}")
            return
        
        # Execute the split processes
        self.apply_gaussian_mask()
        self.apply_rotten_mask()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Transformation.py <path_to_image.JPG>")
    else:
        # Instantiate the class and run the full process
        transformer = Transformation(sys.argv[1])
        transformer.process()
