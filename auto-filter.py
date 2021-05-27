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

def count_pixel_perc(img, rep_value):
    rep_pixel_count = 0
    width, height = img.size
    for i in range(0, width):
        for j in range (0, height):
            pix = img.getpixel((i,j))
            if pix == rep_value:
                rep_pixel_count += 1
    return rep_pixel_count / (width * height)

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

            #calculate the proportion of the background/waifu
            #will remove too zoomed waifus
            back_perc = count_pixel_perc(img_floodfilled, rep_value)

            #calculate the luminosity score of the image
            #will remove sketched waifu, or ones that have halo effects
            total = 0
            for i in range(0, width):
                for j in range(0, height):
                    total += img_floodfilled.getpixel((i,j))[0]

            lumi = (total / (width * height))/255

            #get the image mask with OpenCV

            # Read image
            im_in = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE);

            # Threshold.
            # Set values equal to or above 220 to 0.
            # Set values below 220 to 255.

            th, im_th = cv2.threshold(im_in, 250, 255, cv2.THRESH_BINARY_INV);

            # Copy the thresholded image.
            im_floodfill = im_th.copy()

            # Mask used to flood filling.
            # Notice the size needs to be 2 pixels than the image.
            h, w = im_th.shape[:2]
            mask = np.zeros((h+2, w+2), np.uint8)

            # Floodfill from point (0, 0)
            cv2.floodFill(im_floodfill, mask, (0,0), 255)
            cv2.floodFill(im_floodfill, mask, (width-1,0), 255)
            cv2.floodFill(im_floodfill, mask, (width-1,seed3_y), 255)
            cv2.floodFill(im_floodfill, mask, (0,seed4_y), 255)

            # Invert floodfilled image
            im_floodfill_inv = cv2.bitwise_not(im_floodfill)

            # Combine the two images to get the foreground.
            im_out = im_th | im_floodfill_inv

            # Display images.
            # cv2.imshow("Thresholded Image", im_th)
            # cv2.imshow("Floodfilled Image", im_floodfill)
            # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
            # cv2.imshow("Foreground", im_out)
            # cv2.waitKey(0)
            img_mask_cv = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
            img_mask = Image.fromarray(img_mask_cv)

            #coloring the waifu in red (the waifu is centered)
            center = (width//2, height//2)
            ImageDraw.floodfill(img_mask, center, (255,0,0), thresh = 20)

            #applying the mask on the original image
            #red = keeping, red is the waifu
            masked_img = Image.new("RGB",(width, height), color = "white")
            for i in range(0, width):
                for j in range (0, height):
                    p_mask = img_mask.getpixel((i,j))
                    if p_mask == (255,0,0):
                        masked_img.putpixel((i,j), img.getpixel((i,j)))

            #now we are testing the emptyness some area (top left and top right part of the image)
            #this way it will remove horned waifus, hated waifus, merged back element like blackboard, pillow etc
            #borders
            waifu_border_left = round(width * 0.1)
            waifu_border_right = width - round(width * 0.1)
            waifu_border_bottom = round(height * 0.2)
            draw = ImageDraw.Draw(masked_img)

            #detect no-white pixels in borders area
            found_left = find_matching_pixel(masked_img, 0, waifu_border_left, waifu_border_bottom, (255,255,255))
            found_right = find_matching_pixel(masked_img, waifu_border_right, width, waifu_border_bottom, (255,255,255))

            if found_left or found_right:
                lumi = 1

            # draw tested borders for debugging purpose only
            # draw.line((waifu_border_left, 0,waifu_border_left, waifu_border_bottom ), fill=(255,0,0))
            # draw.line((waifu_border_right, 0,waifu_border_right, waifu_border_bottom ), fill=(255,0,0))
            # draw.line((0, waifu_border_bottom, waifu_border_left, waifu_border_bottom ), fill=(255,0,0))
            # draw.line((waifu_border_right, waifu_border_bottom, width -1, waifu_border_bottom ), fill=(255,0,0))
            

            # Save file
            if( back_perc >= 0.2 and lumi < 0.9):
                output_f = OUTPUT_FOLDER
            else :
                output_f = TRASH_FOLDER

            r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
            r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
            masked_img.save(f'{output_f}{r_path}', "PNG")




if __name__ == '__main__':
   main()

   
