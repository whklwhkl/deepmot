import unittest
import torchvision
import torch
import sys
sys.path.append('.')
import tracker


class T_torch(unittest.TestCase):
    def setUp(self):
        self.box1 = torch.rand(3, 4) * 240
        self.box1[:, 2:] += self.box1[:, :2]
        self.box2 = torch.rand(6, 4) * 240
        self.box2[:, 2:] += self.box2[:, :2]
        self.feature_map = torch.rand(1, 32, 480, 720)

    def test_iou(self):
        iou = torchvision.ops.box_iou(self.box1, self.box2)
        print(iou)

    def test_roi(self):
        roi = torchvision.ops.roi_align(self.feature_map, [self.box1], [3,3])
        print(roi.shape)

    def test_box_distance(self):
        dist = tracker.distance(self.box1, self.box2)
        print('box dist')
        print(dist)

class T_detector(unittest.TestCase):
    def setUp(self):
        return

    def test_box(self):
        return

    def test_feature(self):
        return

class T_curator(unittest.TestCase):
    video_path ='‪C:\\Users\\phiii\\Downloads\\MOT17-04-SDP.mp4'
    curator = tracker.Curator(video_path)
    curator.show()

if __name__ == '__main__':
    unittest.main()
