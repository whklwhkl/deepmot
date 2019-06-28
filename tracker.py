import numpy as np
import torch.nn as nn


class Track:

    CURRENT_ID = 0
    HEALTH = 5

    def __init__(self, init_box:np.ndarray):
        self.box = init_box
        self.age = 0
        self.health = Track.HEALTH
        self.is_visible = True
        self.velocity_2d = np.zeros(4)
        self.velocity_3d = np.zeros(3)  # todo: estimate this using 2d info
        self.id = Track.CURRENT_ID
        Track.CURRENT_ID += 1

    def predict_box(self):
        pred_box = self.box + self.velocity_2d
        return pred_box

    def update(self, new_box:np.ndarray):
        self.age += 1
        self.health = Track.HEALTH
        self.velocity_2d = new_box - self.box
        self.box = new_box

    def update_null(self):
        self.age += 1
        self.health -= 1
        self.box = self.predict_box()

    def is_dead(self):
        return self.health <= 0


class Tracker:

    CUT_OFF_IOU = .5
    BIRTH_IOU = .5
    MINIMUM_CONFIDENCE = .6

    def __init__(self, model:nn.Module):
        self.tracks = {}    # type: {str:Track}
        self.model = model

    def update(self, boxes:[np.ndarray]):
        ids, anchors = self._gather()
        iou_matrix = iou(anchors, boxes)    # type: np.ndarray
        distance_matrix = 1 - iou_matrix + 12
        trackers2boxes = self._assign(distance_matrix)
        unassigned_box_idx = set(range(iou_matrix.shape[1]))

        for track_id, box_index in zip(ids, trackers2boxes):
            try:
                self.tracks[track_id].update(boxes[box_index])
                unassigned_box_idx -= {box_index}
            except:     # don't have an assigned box
                self.tracks[track_id].update_null()
                if self.tracks[track_id].is_dying():
                    self._kill(track_id)

        max_iou_per_box = iou_matrix.max(0)

        for idx in unassigned_box_idx:
            if max_iou_per_box[idx] < Tracker.BIRTH_IOU:
                self._birth(boxes[idx])

    def _gather(self):
        ids, anchors = [], []
        for id, track in self.tracks.items():
            ids += [id]
            anchors += [track.box]
        return ids, np.stack(anchors)

    def _assign(self, distance_matrix:np.ndarray):
        trackers2boxes = self.model(distance_matrix)
        trackers2boxes_ = np.concatenate([trackers2boxes,
                                          Tracker.MINIMUM_CONFIDENCE * np.ones([trackers2boxes.shape[0], 1])],
                                         -1)
        trackers2boxes_ /= np.linalg.norm(trackers2boxes_, 1)
        return np.argmax(trackers2boxes_, -1)

    def _birth(self, box):
        new = Track(box)
        self.tracks[new.id] = new

    def _kill(self, track_id:str):
        del self.tracks[track_id]


class Object:

    def __init__(self, id:int, box:np.ndarray, feature:np.ndarray):
        self.id = id
        self.box = box
        self.feature = feature


class Detector:

    def __init__(self, model:nn.Module):
        self.feature_map = None
        self.model = model

        def get(module, input, output):
            self.feature_map = output

        self.model.register_forward_hook(get)

    def detection(self, img):
        box = img
        return box, roi_pooling(self.feature_map, box)


class Curator:

    def __init__(self):
        self.objects = []   # type: [Object]

    def show(self):
        pass

    def update(self, new_objects:[Object]):
        self.objects = new_objects


def iou(box1:np.ndarray, box2:np.ndarray):
    area = lambda box: (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    upper_left = np.maximum(box1[:, :2], box2[:, :2])
    bottom_right = np.minimum(box1[:, 2:4], box2[:, 2:4])
    intersection = np.maximum(0, area(np.concatenate([upper_left, bottom_right], -1)))
    union = area(box1) + area(box2) - intersection
    return intersection / union


def roi_pooling(feature_map, box):
    return feature_map[box]