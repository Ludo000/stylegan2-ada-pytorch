from PIL import Image, ImageFilter, ImageDraw
import PIL as pil
import json
import os
from tqdm import tqdm
import sys
import numpy as np
from subprocess import call
import cv2;


DATA_FOLDER = '../RESULT_AUTO_FILTER_512/'
OUTPUT_FOLDER = '../RESULT_AUTO_FILTER/'
TRASH_FOLDER = '../TRASH_AUTO_FILTER/'

def del_file(path):
    # delete files
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def find_matching_pixel(img, x0,x1, y, rep_value):
    found = False
    for i in range(x0, x1):
        for j in range (0, y):
            p = img.getpixel((i,j))
            if(p != rep_value):
                found = True
                break
        if(found): break
    return found

def main() :
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TRASH_FOLDER, exist_ok=True)
    del_file(OUTPUT_FOLDER)
    del_file(TRASH_FOLDER)
    for entry in os.scandir(DATA_FOLDER):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            try :
                img = Image.open(entry.path)
            except pil.UnidentifiedImageError:
                continue
            img = img.convert('RGB')
            img_greyscale = img.convert('LA')
            pixels = img_greyscale.load()
            width, height = img.size
            # Pixel Value which would
            # be used for replacement
            rep_value = (0, 0, 0, 0)

            all_pixels = []
            for x in range(width):
                for y in range(height):
                    cpixel = pixels[x, y]
                    all_pixels.append(cpixel)
            
            


            img_floodfilled = img_greyscale.convert('RGBA')
            
            # Location of seed
            seed = (0, 0)
            seed2= (width-1, 0)

            #we take the first white pixels from the bottom right as seed3_y
            seed3_y = height - 1
            for j in range(height - 1, 0, -1):
                if(img_floodfilled.getpixel((width - 1, j)) == (255,255,255,255)):
                    seed3_y = j
                    continue
            seed3= (width-1, seed3_y)

            #we take the first white pixels from the bottom left as seed4_y
            seed4_y = height - 1
            for j in range(seed4_y, 0, -1):
                if(img_floodfilled.getpixel((0, j)) == (255,255,255,255)):
                    seed4_y = j
                    continue
            seed4= (0, seed4_y)
            
            
            
            # Calling the floodfill() function and
            # passing it image, seed, value and
            # thresh as arguments
            ImageDraw.floodfill(img_floodfilled, seed, rep_value, thresh = 20)
            ImageDraw.floodfill(img_floodfilled, seed2, rep_value, thresh = 20)
            ImageDraw.floodfill(img_floodfilled, seed3, rep_value, thresh = 20)
            ImageDraw.floodfill(img_floodfilled, seed4, rep_value, thresh = 20)

            rep_pixel_count = 0
            for i in range(0, width):
                for j in range (0, height):
                    pix = img_floodfilled.getpixel((i,j))
                    if pix == rep_value:
                        rep_pixel_count += 1
            lumi = rep_pixel_count / (width * height)

            total = 0
            for i in range(0, width):
                for j in range(0, height):
                    total += img_floodfilled.getpixel((i,j))[0]

            mean = (total / (width * height))/255

            

            #borders
            waifu_border_left = round(width * 0.1)
            waifu_border_right = width - round(width * 0.1)
            waifu_border_bottom = height - round(height * 0.50)
            draw = ImageDraw.Draw(img_floodfilled)

            #detect no-white pixels in borders area
            found_left = find_matching_pixel(img_floodfilled, 0, waifu_border_left, waifu_border_bottom, rep_value)
            found_right = find_matching_pixel(img_floodfilled, waifu_border_right, width, waifu_border_bottom, rep_value)

            if found_left or found_right:
                mean = 1

            draw.line((waifu_border_left, 0,waifu_border_left, waifu_border_bottom ), fill=(255,0,0))
            draw.line((waifu_border_right, 0,waifu_border_right, waifu_border_bottom ), fill=(255,0,0))
            draw.line((0, waifu_border_bottom, waifu_border_left, waifu_border_bottom ), fill=(255,0,0))
            draw.line((waifu_border_right, waifu_border_bottom, width -1, waifu_border_bottom ), fill=(255,0,0))

            print(mean)
            


            # Save file
            if( lumi >= 0.3 and mean < 0.9):
                output_f = OUTPUT_FOLDER
            else :
                output_f = TRASH_FOLDER

            r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
            r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
            img_floodfilled.save(f'{output_f}{r_path}', "PNG")




if __name__ == '__main__':
   main()

   
