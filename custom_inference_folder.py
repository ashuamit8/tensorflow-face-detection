#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2,os
from PIL import Image

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'

images_path = './media/x/'
output_path = './media/x_output/'

NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.windowNotSet = True


    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(elapsed_time))

        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    import sys
    print("========>>>",sys.argv)
 #    if len(sys.argv) != 2:
 #        print ("usage:%s (cameraID | filename) Detect faces\
 # in the video example:%s 0"%(sys.argv[0], sys.argv[0]))
 #        exit(1)

    # try:
    # 	camID = int(sys.argv[1])
    # except:
    # 	camID = sys.argv[1]

    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    # cap = cv2.VideoCapture(camID)

    windowNotSet = True
    # while True:
    for filename in os.listdir(images_path):
        print(filename)
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".JPG") or filename.endswith(".JPEG"):

            # continue
            # ret, image = cap.read()
            # if ret == 0:
            #     break
            # image = Image.open(images_path+filename)
            image = cv2.imread(images_path+filename)
            # print("image.shape====",image.shape)
            [h, w] = image.shape[:2]
            # print (h, w)
            image = cv2.flip(image, 1)

            (boxes, scores, classes, num_detections) = tDetector.run(image)
            boxes=np.squeeze(boxes)
            scores=np.squeeze(scores)
            j=0
            for i in range(min(2, boxes.shape[0])):

                if scores is None or scores[i] > 0.7:
                    if i==0:
                        box = tuple(boxes[i].tolist())
                    j=j+1;
                    # print("boxes =====",len(boxes[i]),i,box,j)

            if j==1:
                # draw = ImageDraw.Draw(image)
                im_width, im_height = [w, h]
                use_normalized_coordinates=True

                ymin=box[0],
                xmin=box[1],
                ymax=box[2],
                xmax=box[3],


                # print("ymin================",ymin[0],ymin[0]*1080)

                if use_normalized_coordinates:
                    # print("use_normalized_coordinates:True===")
                    # print(xmin,im_width, xmax,im_width,ymin, im_height, ymax,im_height)
                    (left, right, top, bottom) = (xmin[0] * im_width, xmax[0] * im_width,
                    ymin[0] * im_height, ymax[0] * im_height)
                    # print("points left, right, top, bottom ====>>> : ",left, right, top, bottom)

                    box_width=right-left
                    box_height=bottom-top
                    # print("box_width,box_height====>>> : ",box_width,box_height)

                else:
                    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

                area_full=im_height*im_width,
                area_bb=(box_height*box_width)
                # print("*****************--->>",area_bb,area_full[0])

                percentage_bb=(area_bb/(area_full[0]*1.0))*100
                print("percentage_bb------->>",percentage_bb,box_height,box_width,area_bb,area_full[0])
                if percentage_bb>3:
                    cv2.imwrite(output_path+filename,image)

            # # len(boxes[i]),i
            # vis_util.visualize_boxes_and_labels_on_image_array(
            # image,
            # np.squeeze(boxes),
            # np.squeeze(classes).astype(np.int32),
            # np.squeeze(scores),
            # category_index,
            # use_normalized_coordinates=True,
            # line_thickness=4)

            # if windowNotSet is True:
            #     cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            #     windowNotSet = False

            # cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)

            #filter if bb equal exactly one.
            # if len(boxes)>=1:


            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break
        else:
            continue
        # print(os.path.join(images_path, filename))

    print("game over ==== >>>")


    # cap.release()
