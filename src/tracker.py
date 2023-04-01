import torch
from PIL import Image
from tqdm import tqdm

from .deep_sort_pytorch.deep_sort import DeepSort
from .deep_sort_pytorch.utils.parser import get_config



class MultiObjectTracker:
    def __init__(self, image_path, detected_image_path, detector, iou_threshold=0.5, max_lost=5):  # добавлены параметры
        self.image = Image.open(image_path)
        self.detected_image_path = detected_image_path

        self.detector = detector
        self.deepsort = self._initialize_deepsort(iou_threshold, max_lost)  # передаем аргументы

    def __iter__(self):
        tracking = None

        for frame in [self.image]:
            bbox_xywh, bbox_conf, bbox_classes = self._detect_image_for_deepsort(frame)

            if bbox_conf is not None and len(bbox_conf):
                tracking = self.deepsort.update(bbox_xywh, bbox_conf, bbox_classes, frame)
            else:
                self.deepsort.increment_ages()
            yield frame, tracking

    def _detect_image_for_deepsort(self, image):
        """
        Convert detections from YOLOv5 output to the needed format in DeepSORT.
        """
        bbox_df = self.detector(image).pandas().xywh[0]
        bbox_xywh = torch.Tensor(bbox_df[['xcenter', 'ycenter', 'width', 'height']].to_numpy())
        bbox_conf = torch.Tensor(bbox_df[['confidence']].to_numpy())
        bbox_classes = torch.Tensor(bbox_df[['class']].to_numpy())
        return bbox_xywh, bbox_conf, bbox_classes

    @staticmethod
    def _initialize_deepsort(iou_threshold, max_lost):
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
            use_cuda=torch.cuda.is_available(),
            iou_threshold=iou_threshold,  # добавлен аргумент
            max_lost=max_lost  # добавлен аргумент
        )
        return deepsort

