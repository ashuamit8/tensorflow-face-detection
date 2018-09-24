#for inferencing folder images and save them to other folder with filter fields
#according to bounding box percentage>3% and single face detected

python rename_with_bson_id.py
python custom_inference_folder.py

## setting personal config for github purposes.
# git config --global user.name "ashuamit"
# git config --global user.email "ashuamit786@gmail.com"
# git config user.email "ashuamit786@gmail.com"


#scripts purposes.
rename_with_bson_id.py : '''used to rename raw images name to bson id's and
 copy high resolution(1024x1024) images to other folder.'''

custom_inference_folder.py :'''moving all images having exactly 1-face and
having percentage bounding box with full images more than 3%'''

opencv_fd.py: '''detecting face in VideoCapture using opencv'''
