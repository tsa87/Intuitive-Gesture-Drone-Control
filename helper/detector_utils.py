from helper.config import hand_config as config
from helper import label_map_util
import tensorflow as tf
import numpy as np
import cv2

""" load the pre-trained model according to TF's instruction
"""
def initalize_model():
    # initalize model, label and categories
    label_map = label_map_util.load_labelmap(config.PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=config.NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    print("\n[INFO] starting to load the frozen model.\n")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(config.PATH_TO_MODEL, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print("\n[INFO] frozen model finished loading.\n")
    return detection_graph, sess

def detect_objects(image_np, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

def return_final_predictions(num_hands, score_thresh, scores, boxes, width, height):
    rects = []
    for i in range(num_hands):
        if scores[i] > score_thresh:
            # rearrange the order from (start_y, start_x, end_y, end_x)
            # also convert the porportion to pixels
            (start_x, start_y, end_x, end_y) = (
                boxes[i][1]*width,
                boxes[i][0]*height,
                boxes[i][3]*width,
                boxes[i][2]*height)

            rects.append((start_x, start_y, end_x, end_y))

    return rects

def track_objects(frame, rects, tracker):
    hands = tracker.update(rects)
    for (objectID, hand_instance) in hands.items():
        text = "ID {}".format(objectID)
        centroid = hand_instance.cordinate
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX , 1, (255, 255, 255), 1)
        cv2.circle(frame, (centroid[0], centroid[1]), 5, (255, 255, 255), -1)
"""
rects (list of tuple): bounding box cordinate (start_x, start_y, end_x, end_y)
"""
def draw_bounding_boxes(img, rects):
    for rect in rects:
        p1 = (int(rect[0]), int(rect[1]))
        p2 = (int(rect[2]), int(rect[3]))
        cv2.rectangle(img, p1, p2, (255, 255, 255), 3)

"""calculate the movement of each object
left_right_tracker (CentroidTracker): maintains a record of detected object(s).
"""
def display_movement(frame, tracker):
    for (i, (objectID, hand_instance)) in enumerate(tracker.object_holder.items()):
        if (len(hand_instance.cordinate_history)) == config.PREV_FRAMES_COMPARE:
            (x, y) = ("", "")

            #compute the distance
            dx = hand_instance.cordinate_history[0][0] - hand_instance.cordinate_history[-1][0]
            dy = hand_instance.cordinate_history[0][1] - hand_instance.cordinate_history[-1][1]

            if np.abs(dx) >= config.CHANGEX_THRESH:
                if dx > 0:
                    x = "Right"
                else:
                    x = "Left"

            if np.abs(dy) >= config.CHANGEY_THRESH:
                if dy > 0:
                    y = "Down"
                else:
                    y = "Up"

            if (x, y) != ("", ""):
                direction = "Direction: {}-{}".format(x, y)
                cv2.putText(frame,
                    direction,
                    (400,50 + i*50), #origin
                    cv2.FONT_HERSHEY_DUPLEX,
                    1, #scale
                    (255,255,255),
                    1,
                    )

            dx_dy = "dX: {} dY: {}".format(dx, dy)

            # object ID:
            cv2.putText(frame,
                "#{} {}".format(objectID,dx_dy),
                (10,50 + i*50), #origin
                cv2.FONT_HERSHEY_DUPLEX,
                1, #scale
                (255,255,255),
                1,
                )
