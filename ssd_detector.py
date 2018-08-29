#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import cv2
# import matplotlib.pyplot as plt
import glob
import time

# if you want to use Hydrogen in Atom, uncomment the following line
# %matplotlib inline

# GRAPH_FILE = './traffic_light_inference_graph/frozen_inference_graph.pb'
GRAPH_FILE = './frozen_inference_graph.pb'
BOXES_SCORE_MIN = 0.5


def load_file_names(folders, format='png'):
    img_fns = []

    for f in folders:
        file_pattern = './' + f + '/*.' + format
        img_fns.extend(glob.glob(file_pattern))

    return img_fns


def load_images(img_fns):
    imgs = []

    for fn in img_fns:
        print(fn)
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    return imgs


def load_graph(graph_file):
    # Loads a frozen inference graph
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph


def TLDetection(image, sess):

    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={image_tensor: image_np})

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    print(scores[0:3])
    print(classes[0:3])

    return boxes, scores, classes


def TLBoxes(prob, boxes, scores, classes):
    # filter boxes under minimum probability 'prob'
    # COCO class index for TrafficLight is '10'
    n = len(boxes)
    idxs = []
    # target = {1, 2, 3}
    for i in range(n):
        if scores[i] >= prob:
            # if scores[i] >= prob and classes[i] in target:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    # print(filtered_classes)
    # print()
    return filtered_boxes, filtered_scores, filtered_classes


def TLResizeBoxes(boxes, image_height, image_width):
    # Resize boxes from original values (0:1) to image size
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * image_height
    box_coords[:, 1] = boxes[:, 1] * image_width
    box_coords[:, 2] = boxes[:, 2] * image_height
    box_coords[:, 3] = boxes[:, 3] * image_width

    return box_coords


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=3):
    # Draws bounding boxes
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def paste_light_state(image, light_state):
    """
    Return image with curvature and car offset information embedded

    Input
    -----
    image : image to be modified

    light_state : string indicates the state of traffic light, "RED", "YELLOW" or "GREEN"

    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, str(light_state), (20, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image


def process_images(images, threshold=BOXES_SCORE_MIN):
    # images need to be in RGB
    for image in images:
        gbeq_image = cv2.GaussianBlur(image, (5, 5), 0)
        boxes, scores, classes = TLDetection(gbeq_image, sess)
        boxes, scores, classes = TLBoxes(threshold, boxes, scores, classes)

        image_height = image.shape[0]
        image_width = image.shape[1]
        box_coordinates = TLResizeBoxes(boxes, image_height, image_width)

        if len(boxes) != 0:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            processed_image = draw_boxes(image, box_coordinates, color=(0, 0, 255), thick=3)
            # processed_image = paste_light_state(processed_image, )
            cv2.imwrite('./processed_sample_data/file_{}.png'.format(time.time()), processed_image)


def TLImage_Pro(image):
    # Image processing using openCV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #cv2.imwrite('/home/gabymoynahan/CarND-Capstone/data/processed_images/HSV_{}.png'.format(time.time()),image_hsv)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    red1 = cv2.inRange(image_hsv, lower_red , upper_red)

    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    red2 = cv2.inRange(image_hsv, lower_red , upper_red)
    converted_img = cv2.addWeighted(red1, 1.0, red2, 1.0, 0.0)
    #cv2.imwrite('/home/gabymoynahan/CarND-Capstone/data/processed_images/converted_{}.png'.format(time.time()),converted_img)
    blur_img = cv2.GaussianBlur(converted_img,(15,15),0)

    circles = cv2.HoughCircles(blur_img,cv2.HOUGH_GRADIENT,0.5,41, param1=70,param2=30,minRadius=5,maxRadius=120)
    #cv2.imwrite('/home/gabymoynahan/CarND-Capstone/data/processed_images/circles_{}.png'.format(time.time()),circles)
    return circles


if __name__ == '__main__':
    detection_graph = load_graph(GRAPH_FILE)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    sess = tf.Session(graph=detection_graph)

    png_img_fns = load_file_names(['camera_images'])
    # jpg_img_fns = load_file_names(['site_images'], format='jpg')
    png_imgs = load_images(png_img_fns)
    # jpg_imgs = load_images(jpg_img_fns)

    process_images(png_imgs)
    # process_images(jpg_imgs, threshold=0.5)
