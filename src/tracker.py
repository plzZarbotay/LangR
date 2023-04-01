import torch
from PIL import Image
from tqdm import tqdm

from .deep_sort_pytorch.deep_sort import DeepSort
from .deep_sort_pytorch.utils.parser import get_config


class MultiObjectTracker:
    def __init__(self, image_path, detected_image_path, detector, max_lost):
        self.image_path = image_path
        self.detected_image_path = detected_image_path
        self.detector = detector
        self.max_lost = max_lost
        self.trackers = []
        self.frames_since_last_detection = 0
        self.iou_threshold = 0.5 # default value

    def update(self, frame_idx, boxes, confidences=None, classes=None, embeddings=None):
        if len(self.trackers) == 0:
            self.frames_since_last_detection += 1
            return []

        # вызов трекера deep_sort с передачей iou_threshold в качестве аргумента
        outputs = self.deepsort.update(torch.Tensor(boxes), confidences, classes, embeddings, self.iou_threshold)

        # обновление трекеров
        self._update_trackers_state(outputs)

        # удаление потерянных трекеров
        self._remove_lost_trackers()

        # сохранение изображения с рамками вокруг отслеживаемых объектов
        self._save_image_with_boxes(frame_idx)

        # возвращение рамок вокруг отслеживаемых объектов
        return self.get_boxes(frame_idx)
    @staticmethod
    def _initialize_deepsort():
        cfg = get_config()
        cfg.merge_from_file('src/deep_sort_pytorch/configs/deep_sort.yaml')

        deepsort = DeepSort(
            cfg.DEEPSORT.REID_CKPT,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=torch.cuda.is_available()
        )
        return deepsort
