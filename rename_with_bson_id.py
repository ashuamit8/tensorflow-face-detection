###rename all files with unique ObjectId

from bson import ObjectId


###discard below 1024x1024

from PIL import Image
import fnmatch
import os

FOLDER_PATH = "./media/images/"
OUTPUT_HIGH = "./media/images_high_resolution/"

if not os.path.exists(OUTPUT_HIGH):
    os.makedirs(OUTPUT_HIGH)

save_count = 0;
total_count = 0;
for root, dirnames, filenames in os.walk(FOLDER_PATH):
    # print("folder_name : ",root,"xxxxx", dirnames,"======" ,filenames)

    for filename in fnmatch.filter(filenames, '*.jpg'):
        total_count = total_count + 1;
        # print(total_count)
        print("filename >>>>>>>: ",filename)
        im = Image.open(FOLDER_PATH + filename)
        width, height = im.size
        if width >= 1024 and height >= 1024:
            save_count = save_count+1;
            objId = ObjectId()
            filename_new = str(objId)
            im.save(OUTPUT_HIGH + filename_new +".jpg")

print("total_count-step1",total_count)
print("save_count-step1",save_count)
