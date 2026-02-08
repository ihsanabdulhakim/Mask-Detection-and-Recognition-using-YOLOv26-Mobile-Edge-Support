from src.utils import stretch, nms
from src.detector import load_models, tflite_objectdetection
from src.similarity import generate_mask, embedding_converter
import numpy as np


def detection_pipeline(
    img,
    face_path="model/face_dynamic_range_quant.tflite",
    mask_path="model/maskdetection_dynamic_range_quant.tflite",
):
    conf_face=0.1
    conf_mask=0.7

    face_model = load_models(face_path)
    mask_model = load_models(mask_path)


    stretched, scale_x, scale_y = stretch(img, 320)
    boxes, scores = tflite_objectdetection(face_model, stretched, conf_face)

    results = []

    if len(scores) == 0:
        return results

    keep = nms(boxes, scores)

    for i in keep:
        x1, y1, x2, y2 = boxes[i]

        x1 = int(x1 / scale_x)
        y1 = int(y1 / scale_y)
        x2 = int(x2 / scale_x)
        y2 = int(y2 / scale_y)


        facecrop = img[y1:y2, x1:x2]
        if facecrop.size == 0:
            continue

        facecrop_stretch, _, _, = stretch(facecrop, 320)
        m_boxes, m_scores = tflite_objectdetection(mask_model, facecrop_stretch, conf_mask)

        mask_score = ""
        if len(m_scores) == 0:
            label = "nomask"
        else:
            mask_score = round(max(m_scores), 2)
            label ="mask"
        results.append((x1, y1, x2, y2, label, mask_score))

    return results

def generate_mask_pipeline(
    img,
    nomask_path="model/nomaskarea_dynamic_range_quant.tflite",
    mask_png="recognizer/mask.png"       
):
    
    nomask_model = load_models(nomask_path)
    mask_x1 = mask_y1 = mask_x2 = mask_y2 = None
    conf_nomask = 0.1
    stretch_320, scale_x, scale_y = stretch(img, 320)
    nomask_boxes, nomask_scores = tflite_objectdetection(nomask_model, stretch_320, conf_nomask)
    if len(nomask_scores) > 0:
        best_idx = np.argmax(nomask_scores)
        generate_x1, generate_y1, generate_x2, generate_y2 = nomask_boxes[best_idx]
        mask_x1 = int(generate_x1 / scale_x)
        mask_y1 = int(generate_y1 / scale_y)
        mask_x2 = int(generate_x2 / scale_x)
        mask_y2 = int(generate_y2 / scale_y)
        bbox = (mask_x1, mask_y1, mask_x2, mask_y2)

    imgcopy = img.copy()
    if mask_x1 is not None:
        face_masked = generate_mask(imgcopy,mask_png,bbox)
        return face_masked


def embedding_pipeline(
    img,
    embed_model="model/modelfacenet/20180402-114759.pb",
):
    stretch_160,_,_ = stretch(img,160)
    facemasked_embed = embedding_converter(stretch_160,embed_model)
    return facemasked_embed
