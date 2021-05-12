import PIL.Image as Image
import PIL as pil
import json
import os
from tqdm import tqdm
import sys

DATA_FOLDER = '../BIG/'
OUTPUT_FOLDER = '../RESULT_AUTO_FILTER_512/'

def main() :
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for entry in os.scandir(DATA_FOLDER):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            try :
                img = Image.open(entry.path)
            except pil.UnidentifiedImageError:
                print('error')
                continue
            img = img.convert('RGB')
            width, height = img.size
            if(width != 512 and height != 512):
                img = img.resize((512,512), Image.LANCZOS)

            # Save file
            r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
            print(f'{OUTPUT_FOLDER}{r_path}')
            img.save(f'{OUTPUT_FOLDER}{r_path}', "PNG")

if __name__ == '__main__':
   main()
