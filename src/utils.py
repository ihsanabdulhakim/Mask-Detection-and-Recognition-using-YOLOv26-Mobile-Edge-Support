import cv2
import numpy as np

def stretch(img, size=320):
    h, w = img.shape[:2]

    # hitung scale sesuai ukuran asli → nanti dipakai untuk bbox
    scale_x = size / w
    scale_y = size / h

    # full resize tanpa padding
    canvas = cv2.resize(img, (size, size))


    return canvas, scale_x, scale_y



def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0

def nms(boxes, scores, iou_thresh=0.7):
    idxs = np.argsort(scores)[::-1]  # sort by confidence desc
    keep = []

    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)

        rest = []
        for i in idxs[1:]:
            if iou(boxes[current], boxes[i]) < iou_thresh:
                rest.append(i)

        idxs = np.array(rest)

    return keep
