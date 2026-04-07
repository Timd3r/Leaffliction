from plantcv import plantcv as pcv
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Transformation:
    """Class to perform image transformations on plant images."""

    def __init__(self):
        self.img = None
        self.gaussian = None
        self.mask = None
        self.bs_mask = None
        self.gaussian_mask = None
        self.roi = None
        self.analyze = None
        self.pseudo = None

    def load_image(self, filename):
        """Load an image from the specified filename."""
        self.img, self.path, self.filename = pcv.readimage(filename=filename)

    def gaussian_blur(self):
        """Apply Gaussian blur to the image."""
        gray_img = pcv.rgb2gray_hsv(rgb_img=self.img, channel='s')

        mask_img = pcv.threshold.binary(
            gray_img=gray_img, threshold=100, object_type='light')

        gaussian_mask = pcv.gaussian_blur(img=mask_img, ksize=(7, 7))
        self.gaussian = gaussian_mask
        pcv.print_image(self.gaussian, 'gaussian_mask.jpg')

    def create_mask(self):
        """Create a mask for the leaf using a combination of HSV and L*a*b* color spaces."""
        gray_img = pcv.rgb2gray_hsv(rgb_img=self.img, channel='h')
        mask_img = pcv.threshold.binary(
            gray_img=gray_img, threshold=50, object_type='dark')
        gaussian_mask = pcv.gaussian_blur(img=mask_img, ksize=(11, 11))
        self.gaussian_mask = gaussian_mask
        # 1. Get the 'b' channel from L*a*b* color space
        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Lab)
        l, a, b = cv2.split(lab_img)

        # 2. Threshold the 'b' channel: pixels > 200 become white
        _, b_thresh = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)

        # 3. Combine  gaussian_blur mask with the b_thresh mask
        bs_mask = cv2.bitwise_or(gaussian_mask, b_thresh)

        # 4. Apply this final mask to the original image
        # Create a white background
        background = np.full(self.img.shape, 255,
                             dtype=self.img.dtype)

        # Get the inverse of the mask
        mask_inv = cv2.bitwise_not(bs_mask)

        # Get foreground (leaf) and background parts
        foreground = cv2.bitwise_and(self.img, self.img,
                                     mask=bs_mask)
        background_part = cv2.bitwise_and(background, background,
                                          mask=mask_inv)

        # Combine them to get the final image
        final_image = cv2.add(foreground, background_part)
        self.mask = final_image
        self.bs_mask = bs_mask
        pcv.print_image(self.mask, 'final_mask.jpg')

    def create_ROI(self):
        """Create a Region of Interest (ROI) by adding a blue border around the leaf."""
        x, y, w, h = cv2.boundingRect(self.bs_mask)

        # 2. Create the final_image starting with your original image
        roi = self.img.copy()

        # 3. Make the leaf green within the original image
        # We only change pixels that are part of the mask (bs_mask > 200)
        # Green in BGR is [0, 255, 0]
        roi[self.bs_mask > 200] = [0, 255, 0]

        # 4. Add a blue border square (rectangle) around the leaf
        # We use the coordinates from the bounding box
        # Blue in BGR is [255, 0, 0]. 'thickness' controls border width.
        cv2.rectangle(roi, (x, y), (x + w, y + h), (200, 0, 0), thickness=5)
        self.roi = roi
        pcv.print_image(self.roi, 'roi.jpg')

    def analyze_image(self):
        """Analyze the image by finding contours and performing color analysis."""
        obj_contours, _ = cv2.findContours(
            self.gaussian_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        largest_contour = max(obj_contours, key=cv2.contourArea)

        analysis_image = self.img.copy()

        shape_img = pcv.analyze.size(
            img=analysis_image, labeled_mask=self.gaussian_mask, n_labels=1)

        # cv2.drawContours(shape_img, [largest_contour], -1, (255, 0, 255), 5)

        analysis_image = shape_img

        gray = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)

        # Create a binary mask of just the diseased parts
        _, binary_disease = cv2.threshold(gray, 240, 255, 0)

        kernel = np.ones((5, 5), np.uint8)
        # binary_disease = cv2.bitwise_and(binary_disease, gaussian_mask)
        # binary_disease = cv2.morphologyEx(binary_disease, cv2.MORPH_OPEN, kernel)

        # Find the contours of the diseased spots
        # disease_contours, _ = cv2.findContours(binary_disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_contours = sorted(
            obj_contours, key=cv2.contourArea, reverse=True)

        if len(sorted_contours) >= 2:
            # The second one is likely your inner layer
            inner_contour = sorted_contours[1]
            # cv2.drawContours(analysis_image, [inner_contour], -1, (255, 0, 0), 1)

        # Shrink the white areas to break thin connections
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask = cv2.erode(self.gaussian_mask, kernel, iterations=1)

        obj_contours, hierarchy = cv2.findContours(
            eroded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Draw the blue disease contours onto the analysis image
        # cv2.drawContours(analysis_image, disease_contours, -1, (255, 0, 0), 1)
        self.analyze = analysis_image
        pcv.print_image(self.analyze, 'analyze.jpg')

    def create_pseudo_color(self):
        """Create a pseudo-color image by plotting the top, bottom, and center landmarks."""
        # 1. Run the function
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=self.img, mask=self.gaussian_mask)

        # 2. Add with the .tolist() conversion
        pcv.outputs.add_observation(
            sample='plant',
            variable='bottom_lmk',
            trait='bottom landmarks',
            method='x_axis_pseudolandmarks',
            value=bottom.tolist(),   # <--- Convert NumPy array to Python list here
            label='plant',
            scale='pixels',
            datatype=list
        )

        bottom_landmarks = pcv.outputs.observations['plant']['bottom_lmk']['value']
        # 1. Create canvas
        vis_img = np.copy(self.img)

        # --- CONFIGURATION ---
        dot_size = 5  # <--- INCREASE THIS for larger dots (e.g., 20, 30)
        # ---------------------

        def draw_points(image, points, color, radius):
            """Helper function to draw points on the image."""
            pts = points.reshape(-1, 2)
            for pt in pts:
                x, y = int(pt[0]), int(pt[1])
                # The 3rd argument is the radius (size)
                # The 5th argument (-1) means the circle is filled
                cv2.circle(image, (x, y), radius, color, -1)

        # 2. Draw Top (Blue)
        draw_points(vis_img, top, (255, 0, 0), dot_size)

        # 3. Draw Bottom (Red)
        draw_points(vis_img, bottom, (204, 63, 255), dot_size)

        # 4. Draw Center (Green)
        draw_points(vis_img, center_v, (0, 154, 255), dot_size)

        # 5. Display
        self.pseudo = vis_img
        pcv.print_image(vis_img, 'pseudo_color.jpg')

    def color_analysis(self):
        """Perform color analysis by plotting histograms for different color channels."""
        # 1. Convert your image to the other colorspaces
        # (Assuming 'img' is your BGR/RGB image and 'gaussian_mask' is your mask)
        hsv = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(self.img, cv2.COLOR_RGB2LAB)

        # 2. Define the exact channels and colors from your "Expected" image
        # Format: (Source_Image, Channel_Index, Label, Hex_Color)
        plot_configs = [
            (self.img, 2, 'blue', '#0000FF'),
            (lab, 2, 'blue-yellow', '#FFFF00'),
            (self.img, 1, 'green', '#008000'),
            (lab, 1, 'green-magenta', '#FF00FF'),
            (hsv, 0, 'hue', '#8A2BE2'),
            (lab, 0, 'lightness', '#696969'),
            (self.img, 0, 'red', '#FF0000'),
            (hsv, 1, 'saturation', '#00FFFF'),
            (hsv, 2, 'value', '#FFA500')
        ]

        # 4. Loop through and plot each "Level"
        for group, i, name, color in plot_configs:
            # Calculate histogram for the masked area
            hist = cv2.calcHist(
                [group], [i], self.gaussian_mask, [256], [0, 256])

            # Convert to Percentage (Proportion of pixels %)
            hist_perc = (hist / np.sum(hist)) * 100

            plt.plot(hist_perc, label=name, color=color, linewidth=2)

        # 5. Final Formatting
        plt.title("PlantCV-style Color Analysis (All Levels)",
                  loc='left', fontsize=14)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Proportion of pixels (%)')
        plt.legend(title="color Channel", loc='center left',
                   bbox_to_anchor=(1, 0.5))
        plt.xlim(0, 255)
        plt.grid(True)

        plt.savefig('color_analysis.jpg', bbox_inches='tight')


def main():
    transformation = Transformation()
    transformation.load_image(
        'leaves/images/Apple/Apple_Black_rot/image (1).JPG')
    transformation.gaussian_blur()
    transformation.create_mask()
    transformation.create_ROI()
    transformation.analyze_image()
    transformation.create_pseudo_color()
    transformation.color_analysis()


if __name__ == "__main__":
    main()
