from PIL import Image, ImageEnhance, ImageFilter
import os
import sys


def augment_image(image_path, save_location=None):
    """ Applies various augmentations to the input image and saves them."""
    try:
        # Convert to handle JPG consistently
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Perspective Matrix
        zoom = 0.7
        matrix = (1/zoom, 0, -w*(1-zoom)/2, 0, 1 /
                  zoom, -h*(1-zoom)/2, 0.0015, -0.001)

        variations = [
            (img.rotate(45), "Rotation"),
            (img.filter(ImageFilter.GaussianBlur(radius=1)), "Blur"),
            (ImageEnhance.Contrast(img).enhance(2), "Contrast"),
            (img.crop((0, 0, w // 1.2, h // 1.2)).resize((w, h)), "Scaling"),
            (ImageEnhance.Brightness(img).enhance(1.5), "Illumination"),
            (img.transform((w, h), Image.PERSPECTIVE,
             matrix, resample=Image.BICUBIC), "Projective")
        ]

        base_name, ext = os.path.splitext(image_path)
        file_name = os.path.basename(base_name)
        for i, (aug_img, aug_name) in enumerate(variations):
            if save_location:
                aug_img.save(f"{save_location}/{file_name}_{aug_name}{ext}")
            else:
                aug_img.save(f"{base_name}_{aug_name}{ext}")

    except Exception as e:
        print(f"Error: {e}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Augmentation.py <input_image_path>")
        exit(1)
    input_image_path = sys.argv[1]
    augment_image(input_image_path)


if __name__ == "__main__":
    main()
