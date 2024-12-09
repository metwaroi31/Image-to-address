import math
from torchvision.ops import box_convert
import torch


class DistanceCalculator:
    FOCAL_LENGTH_MM = 12  # fov : Wide
    FOV_DEG = 170.0
    IMAGE_WIDTH_PX = 4028
    SENSOR_WIDTH_MM = 6.17
    
    def __init__(self) -> None:
        self.focal_length_px = (self.FOCAL_LENGTH_MM * self.IMAGE_WIDTH_PX) / self.SENSOR_WIDTH_MM
    
    def calculate_distance(self, boxes):
        h, w, _ = [4028, 4028, 3]
        return_distance = []
        boxes_ = boxes * torch.Tensor([w, h, w, h])
        xyxys = box_convert(boxes=boxes_, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()
        for xyxy in xyxys:
            x1, y1, x2, y2 = [int(_) for _ in xyxy]
            width = x2 - x1
            height = y2 - y1
            ratio = width/height
            real_heigt = 80
            # area = width * height
            # For billboards that are not 
            # if ratio > 4:
            #     continue
            # if ratio < 0.5:
            #     continue
            # if ratio > 0.8 and ratio < 1.2:
            #     continue
            
            # if height < 200:
            #     continue
            # if width < 200:
            #     continue
            if ratio > 1.8:
                real_heigt = 30
            calculated_distance = (self.focal_length_px * real_heigt) / height
            print (calculated_distance)
            # Adjust using scale factor
            return_distance.append(calculated_distance)
        return return_distance
