import cv2
import os
import numpy as np
from tabulate import tabulate
from tqdm.autonotebook import tqdm

class Video:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
    def __iter__(self):
        self.current_frame = 0
        while self.current_frame < self.num_frames:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame += 1
            yield frame
            
    def get_frame(self, frame_num):
        if frame_num < 0 or frame_num >= self.num_frames:
            raise ValueError(f"Invalid frame number {frame_num}")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Error reading frame {frame_num}")
        return frame
    
    def get_video_writer(self, output_path, codec='mp4v', fps=None, frame_size=None):
        if fps is None:
            fps = self.fps
        if frame_size is None:
            frame_size = self.frame_size
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        return writer
