from plantcv import plantcv as pcv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse


class Transformation:
    """Leaffliction Part 3: Image Transformation Pipeline."""

    def __init__(self, dest_dir=None, file_prefix=""):
        self.dest_dir = dest_dir
        self.file_prefix = file_prefix
        self.img_rgb = None
        self.img_bgr = None
        self.gaussian_mask = None
        self.background_removed = None
        self.disease_mask = None
        self.roi = None
        self.analyze = None
        self.pseudo = None

    def _save(self, img, suffix):
        """Save image with standardized naming."""
        filename = f"{self.file_prefix}_{suffix}.jpg"
        path = os.path.join(
            self.dest_dir, filename) if self.dest_dir else filename
        if isinstance(img, plt.Figure):
            img.savefig(path, bbox_inches='tight')
            plt.close(img)
        else:
            pcv.print_image(img, path)

    def _display(self, img, title):
        """Display image (single-image mode)."""
        cv2.imshow(title, img)
        cv2.waitKey(0)

    def load_image(self, filename):
        """Load and convert images."""
        self.img_rgb, self.path, self.filename = pcv.readimage(
            filename=filename)
        self.img_bgr = cv2.cvtColor(self.img_rgb, cv2.COLOR_RGB2BGR)
        if not self.file_prefix:
            self.file_prefix = os.path.splitext(os.path.basename(filename))[0]

    def gaussian_blur(self):
        """Standard Gaussian blur (Figure IV.2)."""
        gray = pcv.rgb2gray_hsv(rgb_img=self.img_rgb, channel='s')
        mask = pcv.threshold.otsu(gray_img=gray, object_type='light')
        self.gaussian_mask = pcv.gaussian_blur(img=mask, ksize=(7, 7))
        self._save(self.gaussian_mask, 'Gaussian_blur')
        if not self.dest_dir:
            self._display(self.gaussian_mask, 'Gaussian Blur')

    def remove_background(self):
        """
        Robust leaf segmentation for light grey backgrounds.
        Targets green tissue, bridges pale edges, and forces a solid leaf silhouette.
        """
        # 1. Smooth image to reduce noise on pale leaf edges
        blurred = cv2.GaussianBlur(self.img_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 2. HSV Range targeting GREEN tissue
        # H: 15-85 covers yellow-green to blue-green (most leaves)
        # S: 8-255 catches VERY pale edges without grabbing pure grey
        # V: 25-255 excludes dark noise/shadows
        lower_green = np.array([15, 8, 25])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 3. Morphological bridging: Connects pale edges & bridges over rot spots
        # 11x11 kernel ensures thin leaf margins connect to the main body
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4. Keep ONLY the largest object and FORCE IT SOLID
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.gaussian_mask = np.zeros_like(mask)
            return

        largest = max(contours, key=cv2.contourArea)
        # Safety: Ensure we found a leaf-sized object
        if cv2.contourArea(largest) > 5000:
            # --- FIX: Use Convex Hull to fill "bites" where rot was cut out ---
            # This wraps a convex shape around the leaf, ensuring dark spots are included.
            hull = cv2.convexHull(largest)
            final_mask = np.zeros_like(mask)
            # Draw filled hull = 100% solid white leaf shape
            cv2.drawContours(final_mask, [hull], -1, 255, thickness=cv2.FILLED)
            self.gaussian_mask = final_mask
        else:
            self.gaussian_mask = mask

        # 5. Apply mask to original image (Leaf on White Background)
        bg = np.full(self.img_bgr.shape, 255, dtype=np.uint8)
        mask_inv = cv2.bitwise_not(self.gaussian_mask)
        fg = cv2.bitwise_and(self.img_bgr, self.img_bgr,
                             mask=self.gaussian_mask)
        bg_part = cv2.bitwise_and(bg, bg, mask=mask_inv)
        self.background_removed = cv2.add(fg, bg_part)

        self._save(self.background_removed, 'removed_background')
        if not self.dest_dir:
            self._display(self.background_removed, 'Removed Background')

    def create_mask(self):
        """
        Adaptive Disease Detection (Figure IV.3: Mask).
        Uses Excess Green Index + Otsu's threshold. Cuts out ONLY rotten parts.
        """
        leaf_only = self.background_removed.copy()
        b, g, r = cv2.split(leaf_only)
        hsv = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        leaf_mask_bool = self.gaussian_mask > 128
        if not np.any(leaf_mask_bool):
            self.disease_mask = np.zeros_like(self.gaussian_mask)
            return

        # 1. Excess Green Index (ExG): Healthy = High, Disease = Low
        exg = (2.0 * g.astype(np.float32)) - \
            r.astype(np.float32) - b.astype(np.float32)
        exg_leaf = exg[leaf_mask_bool]

        # 2. Otsu's Automatic Thresholding
        otsu_val, _ = cv2.threshold(exg_leaf.astype(np.uint8).reshape(-1, 1), 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3. Shadow Filter: Shadows are dark. Disease is usually brighter.
        v_thresh = np.percentile(v[leaf_mask_bool], 12)

        # 4. Create masks (Strict: 0.35 multiplier reduces false positives)
        disease_mask = (exg < (otsu_val * 0.35)).astype(np.uint8) * 255
        bright_mask = (v > v_thresh).astype(np.uint8) * 255

        # Combine: Must be diseased (low ExG) AND not a shadow (bright enough)
        disease_mask = cv2.bitwise_and(disease_mask, bright_mask)
        disease_mask = cv2.bitwise_and(disease_mask, self.gaussian_mask)

        # 5. Morphological cleanup (Tight boundaries)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3))
        disease_mask = cv2.morphologyEx(
            disease_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        disease_mask = cv2.morphologyEx(
            disease_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        self.disease_mask = disease_mask

        # Debug output
        px_leaf = cv2.countNonZero(self.gaussian_mask)
        px_disease = cv2.countNonZero(disease_mask)
        print(
            f"🔍 ExG Threshold: {otsu_val:.1f} | Shadow Cutoff: {v_thresh:.0f}")
        print(
            f"🦠 Detected rot: {px_disease} / {px_leaf} pixels ({(px_disease/max(px_leaf, 1))*100:.2f}%)")

        # Cut out & save
        rot_only = cv2.bitwise_and(
            self.img_bgr, self.img_bgr, mask=disease_mask)
        x, y, w, h = cv2.boundingRect(self.gaussian_mask)
        pad = 15
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(self.img_bgr.shape[1], x + w +
                     pad), min(self.img_bgr.shape[0], y + h + pad)

        self._save(rot_only[y1:y2, x1:x2], 'Mask')
        if not self.dest_dir:
            self._display(rot_only, 'Mask (Rot Cut-Out)')

    def create_ROI(self):
        """ROI Overlay: Leaf + Red highlight on Rot (Figure IV.4)."""
        roi_img = self.background_removed.copy()
        contours, _ = cv2.findContours(
            self.disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi_img, contours, -1, (0, 0, 255), 2)

        if np.any(self.disease_mask):
            overlay = roi_img.copy()
            overlay[self.disease_mask > 0] = (0, 0, 255)
            roi_img = cv2.addWeighted(overlay, 0.4, roi_img, 0.6, 0)

        x, y, w, h = cv2.boundingRect(self.gaussian_mask)
        cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        self.roi = roi_img
        self._save(roi_img, 'ROI_objects')
        if not self.dest_dir:
            self._display(roi_img, 'ROI Objects (Overlay)')

    def analyze_image(self):
        """Shape/Size Analysis (Figure IV.5)."""
        analysis_img = self.img_rgb.copy()
        try:
            analysis_img = pcv.analyze.shape(img=analysis_img, objects=[
                                             self.gaussian_mask], label="leaf")
        except AttributeError:
            try:
                analysis_img = pcv.analyze.size(
                    img=analysis_img, labeled_mask=self.gaussian_mask, n_labels=1)
            except AttributeError:
                cv2.drawContours(
                    analysis_img, [self.gaussian_mask], -1, (255, 0, 255), 3)
                cv2.putText(analysis_img, f"Area: {cv2.contourArea(self.gaussian_mask):.0f} px",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.analyze = analysis_img
        self._save(self.analyze, 'Analyze_object')
        if not self.dest_dir:
            self._display(self.analyze, 'Analyze Object')

    def create_pseudo_color(self):
        """Pseudolandmarks (Figure IV.6)."""
        top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
            img=self.img_rgb, mask=self.gaussian_mask)
        pcv.outputs.add_observation(sample='plant', variable='bottom_lmk', trait='bottom landmarks',
                                    method='x_axis_pseudolandmarks', value=bottom.tolist(),
                                    label='plant', scale='pixels', datatype=list)
        vis_img = np.copy(self.img_bgr)

        # Robust drawing function that skips 'NA' values to prevent crashes
        def draw_pts(img, pts, color, r):
            for pt in pts.reshape(-1, 2):
                try:
                    # Skip PlantCV's 'NA' placeholders
                    if 'NA' in (str(pt[0]), str(pt[1])):
                        continue
                    cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, -1)
                except:
                    pass

        draw_pts(vis_img, top, (255, 0, 0), 5)
        draw_pts(vis_img, bottom, (204, 63, 255), 5)
        draw_pts(vis_img, center_v, (0, 154, 255), 5)
        self.pseudo = vis_img
        self._save(vis_img, 'Pseudolandmarks')
        if not self.dest_dir:
            self._display(vis_img, 'Pseudolandmarks')

    def color_analysis(self):
        """Color Histogram (Figure IV.7)."""
        hsv = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2LAB)
        plots = [
            (self.img_bgr, 2, 'blue', '#0000FF'),
            (lab, 2, 'blue-yellow', '#FFFF00'),
            (self.img_bgr, 1, 'green', '#008000'),
            (lab, 1, 'green-magenta', '#FF00FF'),
            (hsv, 0, 'hue', '#8A2BE2'),
            (lab, 0, 'lightness', '#696969'),
            (self.img_bgr, 0, 'red', '#FF0000'),
            (hsv, 1, 'saturation', '#00FFFF'),
            (hsv, 2, 'value', '#FFA500')
        ]
        plt.figure(figsize=(10, 6))
        for group, i, name, color in plots:
            hist = cv2.calcHist(
                [group], [i], self.gaussian_mask, [256], [0, 256])
            plt.plot((hist/hist.sum())*100, label=name,
                     color=color, linewidth=2)
        plt.title("Color Analysis", loc='left', fontsize=14)
        plt.xlabel('Pixel intensity')
        plt.ylabel('Proportion of pixels (%)')
        plt.legend(title="Channel", loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlim(0, 255)
        plt.grid(True)
        file_name = os.path.splitext(self.filename)[0]
        os.makedirs("./color_histograms", exist_ok=True)
        # save image at the path ./color_histograms/{file_name}_Color_histogram.jpg
        # plt.savefig(f"./color_histograms/{file_name}_Color_histogram.jpg")
        plt.close()  # Close the plot to free memory
        return plots


def main():
    parser = argparse.ArgumentParser(
        description='Leaffliction Transformation Part 3')
    parser.add_argument('-src', type=str, required=True,
                        help='Source image or directory')
    parser.add_argument('-dst', type=str, default=None,
                        help='Destination directory')
    parser.add_argument('-mask', action='store_true',
                        help='Enable mask output')
    args = parser.parse_args()

    if args.dst and not os.path.exists(args.dst):
        os.makedirs(args.dst)

    if os.path.isdir(args.src):
        files = glob.glob(os.path.join(args.src, '*.JPG')) + glob.glob(os.path.join(args.src, '*.jpg')) + \
            glob.glob(os.path.join(args.src, '*.png'))
        if not files:
            print(f"No images in {args.src}")
            return
        for f in sorted(files):
            print(f"\n--- Processing {f} ---")
            t = Transformation(dest_dir=args.dst)
            t.load_image(f)
            t.gaussian_blur()
            t.remove_background()
            t.create_mask()
            t.create_ROI()
            t.analyze_image()
            t.create_pseudo_color()
            t.color_analysis()
    else:
        print(f"\n--- Processing {args.src} ---")
        t = Transformation(dest_dir=None)
        t.load_image(args.src)
        t.gaussian_blur()
        t.remove_background()
        t.create_mask()
        t.create_ROI()
        t.analyze_image()
        t.create_pseudo_color()
        t.color_analysis()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
