import PIL.Image as Image
import PIL as pil
import json
import os
from tqdm import tqdm
import sys

DATA_FOLDER = '../SMALLx2/'
OUTPUT_FOLDER_BIG = '../BIG/'
OUTPUT_FOLDER_SMALL = '../SMALL/'

def main() :
    os.makedirs(OUTPUT_FOLDER_BIG, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER_SMALL, exist_ok=True)
    for entry in os.scandir(DATA_FOLDER):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            try :
                img = Image.open(entry.path)
            except pil.UnidentifiedImageError:
                print('error')
                continue
            img = img.convert('RGB')
            width, height = img.size

            output_folder = OUTPUT_FOLDER_BIG
            if(width < 512 or height < 512):
                output_folder = OUTPUT_FOLDER_SMALL

            # Save file
            r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
            print(f'{output_folder}{r_path}')
            img.save(f'{output_folder}{r_path}', "PNG")

if __name__ == '__main__':
   main()
