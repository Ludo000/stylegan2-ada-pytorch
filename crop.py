import PIL.Image as pil
import json
import os

DATA_FOLDER = 'DataImages/'
COORD_FOLDER = 'DataImages/faceCoordinate/'
TOP_FACTOR = 2
BOTTOM_FACTOR = 2.2
LEFT_FACTOR = 1.5
RIGHT_FACTOR = 1.5
FACE_AREA_THRESHOLD = 60000
RESULT_SIZE = 512


def crop() :
    for entry in os.scandir(DATA_FOLDER):
        if (entry.path.endswith(".jpg") or entry.path.endswith(".png")) and entry.is_file():
            img = pil.open(entry.path)
            img = img.convert('RGBA')
            json_path = entry.path.replace(DATA_FOLDER,COORD_FOLDER).replace('jpg', 'json').replace('png', 'json')
            with open(json_path) as json_file:
                data = json.load(json_file)
                width, height = img.size
                if( not data['detection_bbox_ymin'] 
                    or not data['detection_bbox_ymax'] 
                    or not data['detection_bbox_xmin'] 
                    or not data['detection_bbox_xmax'] ): continue
                y_min = data['detection_bbox_ymin'][0] * height
                y_max = data['detection_bbox_ymax'][0] * height
                x_min = data['detection_bbox_xmin'][0] * width
                x_max = data['detection_bbox_xmax'][0] * width
                face_width = x_max - x_min
                face_height = y_max - y_min

                # Setting the points for cropped image 
                top = max(y_min - TOP_FACTOR * face_height, 0)
                left = max(x_min - LEFT_FACTOR * face_width, 0)
                right = min(x_max + RIGHT_FACTOR * face_width, width)
                bottom = min(y_max + BOTTOM_FACTOR * face_height, height)
                face_square = face_width * face_height

                # Make it a square
                crop_width = right - left
                crop_height = bottom - top
                margin = abs(crop_width - crop_height) / 2
                if(face_square > FACE_AREA_THRESHOLD) :
                    # by adding margin
                    if(crop_width > crop_height):
                        top -= margin
                        bottom += margin
                    elif(crop_width < crop_height):
                        left -= margin
                        right += margin
                else :
                    # by cropping
                    if(crop_width > crop_height):
                        left += margin
                        right -= margin
                    elif(crop_width < crop_height):
                        top += margin
                        bottom -= margin

                # Cropping
                img = img.crop((left, top, right, bottom)) 

                # Save file
                os.makedirs(f'{DATA_FOLDER}results', exist_ok=True)
                r_path = entry.path.replace(DATA_FOLDER, '').replace('jpg', 'png')
                img = img.resize((RESULT_SIZE,RESULT_SIZE))
                img.save(f'{DATA_FOLDER}results/{r_path}', "PNG")

if __name__ == '__main__':
   crop()
