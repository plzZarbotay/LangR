import cv2
import os
import numpy as np
from tabulate import tabulate
from tqdm.autonotebook import tqdm



class Video:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)

        self.num_frames = 1
        self.fps = 0
        self.height, self.width = self.image.shape[:2]

    def __str__(self):
        video_details = [
            ['Image path', self.image_path],
            ['Number of frames', self.num_frames],
            ['FPS', self.fps],
            ['(height, width)', f'({self.height}, {self.width})']
        ]
        return tabulate(video_details)

    def __iter__(self):
        yield self.image

    def export_frames(self, frames_path):
        os.makedirs(frames_path, exist_ok=True)
        cv2.imwrite(f'{frames_path}/0.jpg', self.image)

    def get_frames_tensor(self):
        return np.array([self.image])
