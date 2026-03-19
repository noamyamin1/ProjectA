import os
from PIL import Image, ImageDraw

INPUT_IMAGE_PATH = "/users/epnyrk/Project/design/work/ProjectA/pyton/pics_to_test/slippery_road_redcar.jpg"
OUTPUT_IMAGE_PATH = "debug_bbox.jpg"

def draw_hardware_bbox():
    if not os.path.exists(INPUT_IMAGE_PATH):
        return

    img = Image.open(INPUT_IMAGE_PATH).convert('RGB')
    
    if img.size != (1920, 1080):
        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.ANTIALIAS
        img = img.resize((1920, 1080), resample_method)

    draw = ImageDraw.Draw(img)
    
    xmin, xmax = 1554, 1688
    ymin, ymax = 568, 677

    draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=5)
    
    img.save(OUTPUT_IMAGE_PATH)
    print(f"Saved debug image to {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    draw_hardware_bbox()