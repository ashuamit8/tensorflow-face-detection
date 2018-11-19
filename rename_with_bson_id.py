###rename all files with unique ObjectId

from bson import ObjectId


###discard below 1024x1024

from PIL import Image
import fnmatch
import os

# python rename_with_bson_id.py

# FOLDER_PATH = "./media/images/"
# OUTPUT_HIGH = "./media/images_high_resolution/"

# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/1/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/2/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/3/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/4/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/5/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/6/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/7/'
# FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/8/'
FOLDER_PATH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/9/'

OUTPUT_HIGH = '/samba/anonymous2/rohit/new_to_merged/Insta10Nov18/all_hr/'


if not os.path.exists(OUTPUT_HIGH):
    os.makedirs(OUTPUT_HIGH)

save_count = 0;
total_count = 0;
for root, dirnames, filenames in os.walk(FOLDER_PATH):
    # print("folder_name : ",root,"xxxxx", dirnames,"======" ,filenames)

    for filename in fnmatch.filter(filenames, '*.jpg'):
        total_count = total_count + 1;
        # print(total_count)
        print("filename >>>>>>>: ",total_count,"---",filename)
        if not os.path.exists(FOLDER_PATH + filename):
            print("continue -------------------------------")
            continue

        filesize=os.path.getsize(FOLDER_PATH + filename)
        if filesize == 0 :
            print("deleting --------------------------------",filesize)
            os.remove(FOLDER_PATH + filename)
            continue

        im = Image.open(FOLDER_PATH + filename)
        width, height = im.size
        if width >= 1024 and height >= 1024:
            save_count = save_count+1;
            objId = ObjectId()
            filename_new = str(objId)
            # im.save(OUTPUT_HIGH + filename_new +".jpg")
            # #deleting original file
            # os.remove(FOLDER_PATH + filename)

            #moved file instead of im.save and removed
            os.rename(FOLDER_PATH + filename,OUTPUT_HIGH + filename_new +".jpg")

print("total_count-step1",total_count)
print("save_count-step1",save_count)
