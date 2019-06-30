import cv2
import torch
import torch.nn as nn
from torchvision.ops import box_iou, roi_align


def distance(box1, box2):
    center = lambda box:(box[:,:2] + box[:,2:4])/2
    center1 = center(box1)
    center2 = center(box2)
    delta = center1[:,None] - center2[None]
    return 1 - torch.exp(-5 * delta.norm(2, dim=-1))


class Object:

    def __init__(self, id:int, box:torch.Tensor, feature:torch.Tensor):
        self.id = id
        self.box = box
        self.feature = feature


class Detector:

    def __init__(self, model:nn.Module):
        self.feature_map = None
        self.model = model

        def get(module, input, output):
            self.feature_map = output

        self.model.register_forward_hook(get)   # locale some layer

    def detection(self, img):
        img = torch.FloatTensor(img)
        img = img.permute(2, 0, 1)[None]
        box = self.model(img)
        return box, roi_align(self.feature_map, [box], [3,3], 1.)


class Track:

    CURRENT_ID = 0
    HEALTH = 5
    PROBATION = 3

    def __init__(self, init_box:torch.Tensor):
        self.box = init_box
        self.age = 0
        self.health = Track.HEALTH
        self.is_visible = True
        self.is_candidate = True
        self.velocity_2d = torch.zeros(5)
        self.velocity_3d = torch.zeros(3)  # todo: estimate this using 2d info
        self.id = Track.CURRENT_ID
        Track.CURRENT_ID += 1

    def predict_box(self):
        pred_box = self.box + self.velocity_2d
        return pred_box

    def update(self, new_box:torch.Tensor):
        self.age += 1
        self.health = Track.HEALTH
        self.velocity_2d = new_box - self.box
        self.velocity_2d[4] = 0
        self.box = new_box
        self.is_candidate = self.age < Track.PROBATION

    def update_null(self):
        self.age += 1
        if self.is_candidate:
            self.health = -9999
        else:
            self.health -= 1
        self.is_visible = False
        self.box = self.predict_box()

    def is_dying(self):
        return self.health <= 0


class Tracker:

    BIRTH_IOU = .5
    CANDIDATE_IOU = .55
    MINIMUM_CONFIDENCE = .6

    def __init__(self, model:nn.Module):
        self.tracks = {}    # type: {str:Track}
        self.model = model

    def update(self, boxes:[torch.Tensor]):
        ids, anchors = self._gather()
        if len(ids):
            anchors = torch.stack(anchors)
            iou_matrix = box_iou(anchors[:,:4], boxes[:,:4])    # type: torch.Tensor
            # iou_matrix[iou_matrix < Tracker.CANDIDATE_IOU] += 1e5
            distance_matrix = 1 - iou_matrix + distance(anchors, boxes)
            # todo: add appearance distance
            trackers2boxes = self._assign(distance_matrix/2)
            unassigned_box_idx = set(range(iou_matrix.shape[1]))

            for track_id, box_index in zip(ids, trackers2boxes):
                try:
                    self.tracks[track_id].update(boxes[box_index])
                    unassigned_box_idx -= {box_index}
                except:     # don't have an assigned box
                    self.tracks[track_id].update_null()
                    if self.tracks[track_id].is_dying():
                        self._kill(track_id)

            max_iou_per_box = iou_matrix.max(0)[0]

            for idx in unassigned_box_idx:
                # breakpoint()
                if max_iou_per_box[idx] < Tracker.BIRTH_IOU:
                    self._birth(boxes[idx])
        else:
            for box in boxes:
                self._birth(box)

    def _gather(self):
        ids, anchors = [], []
        for id, track in self.tracks.items():
            ids += [id]
            anchors += [track.box]
        return ids, anchors

    def _assign(self, distance_matrix:torch.Tensor):
        # breakpoint()
        self.model.hidden_row = self.model.init_hidden(1)
        self.model.hidden_col = self.model.init_hidden(1)
        with torch.no_grad():
            # breakpoint()
            trackers2boxes = self.model(distance_matrix[None])[0]        # scores
        trackers2boxes_ = torch.cat([trackers2boxes,
                                     Tracker.MINIMUM_CONFIDENCE * torch.ones([trackers2boxes.shape[0], 1])],
                                    -1)
        return torch.argmax(trackers2boxes_, -1)

    def _birth(self, box):
        new = Track(box)
        self.tracks[new.id] = new

    def _kill(self, track_id:str):
        del self.tracks[track_id]


class Curator:

    def __init__(self, video_path=None):
        self.video_path = video_path
        self.objects = []   # type: [Object]

    def show(self, callback=lambda x:x):
        cap = cv2.VideoCapture(0 or self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = callback(frame)
            cv2.imshow('video', frame)
            if cv2.waitKey(1) == 27:
                break

    def update(self, new_objects:[Object]):
        self.objects = new_objects


if __name__ == '__main__':
    import pickle
    import numpy as np
    import os.path as osp, os

    curator = Curator('MOT17-04-SDP.mp4')

    class MockDetector(nn.Module):
        def __init__(self, det_path):
            super().__init__()
            self.counter = 0
            self.frame2box = np.load(det_path, allow_pickle=True).tolist()  # start with '1', all lists

        def forward(x):
            self.counter += 1
            return torch.FloatTensor(self.frame2box[str(object=self.counter)])

    det_net = MockDetector('det.npy')
    det = Detector(det_net)
    curator.show()
