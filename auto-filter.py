import PIL.Image as Image
import PIL as pil
import json
import os
from tqdm import tqdm
import sys


DATA_FOLDER = '../DATASET_SOURCE/'
OUTPUT_FOLDER = '../RESULT_AUTO_FILTER/'


def main() :
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for entry in os.scandir(DATA_FOLDER):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            try :
                img = Image.open(entry.path)
            except pil.UnidentifiedImageError:
                continue
            img = img.convert('RGB')
            pixels = img.load()
            width, height = img.size

            all_pixels = []
            for x in range(width):
                for y in range(height):
                    cpixel = pixels[x, y]
                    all_pixels.append(cpixel)
            
            white_pixel_count = 0
            for pix in all_pixels:
                if pix == (255,255,255):
                    white_pixel_count += 1
            white_pixel_perc = white_pixel_count / (width * height)

            # Save file
            if(white_pixel_perc >= 0.2 and white_pixel_perc <= 0.5):
                r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
                img.save(f'{OUTPUT_FOLDER}{r_path}', "PNG")

if __name__ == '__main__':
   main()
