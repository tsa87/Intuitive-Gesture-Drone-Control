from helper.config import hand_config as config
from scipy.spatial import distance
from collections import OrderedDict
from collections import deque
import numpy as np
import math

class HandObject:

    """ contains all infomation of an instance of hand
    attributes:
        cordinates (tuple of int): (x, y)
        forget (int): # of consequtive frames not containing this object
        maxsize (int): compare position with this number of frames before
    """
    def __init__(self, cordinate = (0,0), maxsize = config.PREV_FRAMES_COMPARE):
        self.cordinate = cordinate
        self.forget  = 0
        self.cordinate_history = deque(maxlen = maxsize)

class CentroidTracker:

    """ maintains a record of detected object(s).
    attributes:
        object_holder (OrderedDict): list of HandObject
        next_index (int): index of latest addition to object_holder
        forget_thresh (int): after this # of frames without the object, delete its record
        deleted_holder (set of int): set of indices deleted from object_holder
        max_movement (int): max # of pixels in movement to register as same object
        max_object (int): max # of objects that can exist at one time
        min_distance (int): min # of pixels between unique objects
    """
    def __init__(self,
        forget_thresh= config.FORGET_THRESH,
        max_movement= config.MAX_MOVEMENT,
        max_object= config.NUM_HANDS_TO_DETECT,
        min_distance= config.MIN_DISTANCE,):

        self.object_holder = OrderedDict()
        self.deleted_holder = set()
        self.next_index = 0
        self.forget_thresh = forget_thresh
        self.max_movement = max_movement
        self.max_object = max_object
        self.min_distance = min_distance

    """ enqueue new cordinate history
    delete old one if needed
    args:
        dict_index (int) = index in the CentroidTracker.object_holder
    """
    def append_coordinate_toqueue(self, dict_index):
        hand_instance = self.object_holder[dict_index]
        hand_instance.cordinate_history.appendleft(hand_instance.cordinate)

    """ returns a matrix representing the cordinates of the current tracked objects.
    return:
        cord_matrix (numpy_matrix):
            row: object entries' cordinates
            column: x cordinate and y cordinate
    """
    def dict_to_matrix(self):

        assert(len(self.object_holder) >= 1)

        cord_matrix = np.zeros((len(self.object_holder), 2), dtype=int)

        dict_index = 0
        for i in range(len(self.object_holder)):
            dict_index = list(self.object_holder.keys())[i]
            cord_matrix[i] = self.object_holder[dict_index].cordinate

        return cord_matrix

    """ adds new instance of hand to the record
    do not add if the new instance is too close to a existing object
    args:
        hand (HandObject): contains all infomation of an instance of hand
    """
    def register(self, hand):

        far_enough = True

        # determine if the new object is far enough from existing ones
        if len(self.object_holder) >= 1:
            old_cords = self.dict_to_matrix()

            for i in range(old_cords.shape[0]):
                difference = old_cords[i] - hand.cordinate
                distance = math.sqrt(difference[0] ** 2 + difference[1] ** 2)

                if distance <= self.min_distance:
                    far_enough = False

        if (len(self.object_holder) < self.max_object) and far_enough:
            self.object_holder[self.next_index] = hand
            self.append_coordinate_toqueue(self.next_index)
            self.next_index += 1

    """ delete instance of hand from the record
    args:
        id (int): the key to the hand instance from the object_holder
    """
    def deregister(self, id):
        #print("deleted {}".format(id))
        if len(self.object_holder) > 1:
            self.deleted_holder.add(id)
            del self.object_holder[id]

    """ corrects the self.object_holder key mismatch
    ex. if key "0" in self.object_holder is deleted,
    this function corrects the 1st item of the list
    to the key after key "0".
    args:
        dict_index (int): the #th item in the object_holder
    return:
        dict_index (int): the correct key to the hand instance from the object_holder
    """
    def correct_key(self, dict_index):
        while dict_index in self.deleted_holder:
            dict_index += 1
        return dict_index

    """ maintain existing object instance to not be too close to each other
    deletes one of the instances that is too close
    """
    def clean(self):
        # if object holder is empty, nothing needs to be done
        if len(self.object_holder) < 1:
            return

        old_cords = self.dict_to_matrix()
        clean_index = set()

        for i in range(old_cords.shape[0]):
            if i in clean_index:
                continue
            for j in range(i+1, old_cords.shape[0]):

                difference = old_cords[i] - old_cords[j]
                distance = math.sqrt(difference[0] ** 2 + difference[1] ** 2)

                if distance < self.min_distance:
                    clean_index.add(j)

        dict_index = 0
        for i in range(old_cords.shape[0]):
            dict_index = list(self.object_holder.keys())[i]
            if i in clean_index:
                self.deregister(dict_index)


    """ print method
    used for debugging
    """
    def print_object_holder(self):
        dict_index = 0
        for i in range(len(self.object_holder)):
            dict_index = dict_index = list(self.object_holder.keys())[i]
            print("ID: {}".format(dict_index))
            print("Cordinate: {}".format(self.object_holder[dict_index].cordinate))
            print("forget: {}".format(self.object_holder[dict_index].forget))

    """ update method
    match objects from last frame to existing frame
    update cordinates accordingly
    remove the forgotten
    args:
        rects (list of tuple): detected bounding boxes (startX, startY, endX, endY)
    """
    def update(self, rects):
        # corner case: no bounding box detected
        if (len(rects) == 0):
            for id, hand_instance in self.object_holder.items():
                hand_instance.forget += 1
                self.append_coordinate_toqueue(id)
                if (hand_instance.forget > self.forget_thresh):
                    self.deregister(id)
            return self.object_holder

        input_cords = np.zeros((len(rects), 2), dtype=int)
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            x = int((startX + endX) / 2.0)
            y = int((startY + endY) / 2.0)
            input_cords[i] = (x, y)

        # corner case: if we are initalizing
        if (len(self.object_holder) == 0):
            for (x, y) in input_cords:
                self.register(HandObject((x, y)))
            return self.object_holder

        # main program:
        old_cords = self.dict_to_matrix()

        # compute the distance of bounding boxes from last frame to this frame
        dist = (distance.cdist(old_cords, input_cords)).flatten()
        sorted_indice = dist.argsort()

        used_old_cords = set()
        used_input_cords = set()

        for i in range(len(sorted_indice)):
            # unravel the original row, column index
            old_index = sorted_indice[i] // input_cords.shape[0]
            new_index = sorted_indice[i] - old_index * input_cords.shape[0]

            if old_index in used_old_cords or new_index in used_input_cords:
                continue

            dict_index = list(self.object_holder.keys())[old_index]

            if dist[sorted_indice[i]] < self.max_movement:
                # update cordinate
                self.object_holder[dict_index].cordinate = input_cords[new_index]
                self.object_holder[dict_index].forget = 0

                self.append_coordinate_toqueue(dict_index)

                used_old_cords.add(old_index)
                used_input_cords.add(new_index)

                dict_index += 1

            else:
                break

        self.clean()

        # we need to delete some of the existing objects
        dict_index = 0

        if old_cords.shape[0] > input_cords.shape[0]:
            for i in range(old_cords.shape[0]):
                if i not in used_old_cords:
                    dict_index = self.correct_key(dict_index)
                    self.object_holder[dict_index].forget += 1
                    self.append_coordinate_toqueue(dict_index)
                    if self.object_holder[dict_index].forget > self.forget_thresh:
                        self.deregister(dict_index)

                dict_index += 1
        # we need to add some new objects
        else:
            for i in range(input_cords.shape[0]):
                if i not in used_input_cords:
                    self.register(HandObject(input_cords[i]))

        return self.object_holder
