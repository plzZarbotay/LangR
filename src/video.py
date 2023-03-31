import cv2
import os
import numpy as np
from tabulate import tabulate
from tqdm.autonotebook import tqdm

class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        if os.path.isfile(video_path) and video_path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            self.num_frames = 1
            self.fps = 1
            self.height, self.width, _ = cv2.imread(video_path).shape
        else:
            self.video_cap = cv2.VideoCapture(video_path)
            self.num_frames = self._get_num_frames()
            self.fps = self._get_fps()
            self.height, self.width = self._get_frames_dimension()

    def __str__(self):
        video_details = [
            ['Video path', self.video_path],
            ['Number of frames', self.num_frames],
            ['FPS', self.fps],
            ['(height, width)', f'({self.height}, {self.width})']
        ]
        return tabulate(video_details)

    def _get_num_frames(self):
        num_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert num_frames > 0, 'The video contains 0 frames.'
        return num_frames

    def _get_fps(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FPS))

    def _get_frames_dimension(self):
        _, frame = cv2.VideoCapture(self.video_path).read()
        height, width, channels = frame.shape
        return height, width

    def _get_image_details(self):
        height, width, channels = cv2.imread(self.video_path).shape
        return height, width, channels

    def get_frames_tensor(self):
        if self.num_frames == 1:
            return np.array([cv2.imread(self.video_path)], dtype=np.uint8)
        else:
            frames = []
            for frame in self:
                frames.append(frame)
            return np.array(frames, dtype=np.uint8)
