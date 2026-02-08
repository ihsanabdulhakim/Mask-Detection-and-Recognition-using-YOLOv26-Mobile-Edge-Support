# About
A comprehensive pipeline for real-time face mask detection and recognition using YOLOv26, optimized for mobile and edge devices. This project includes model training, ONNX and TFLite export, and deployment-ready pipelines for CPU/GPU inference with optional quantization for low-resource environments.

![mask2_detected](https://github.com/user-attachments/assets/eeccb634-ba4d-4d61-8bff-c35c838b73f6)
![rdj_recognition](https://github.com/user-attachments/assets/87e29622-8ff7-48b9-831b-e48ca6f7b8d1)
![girl_recognition](https://github.com/user-attachments/assets/93f92913-2e0b-4b53-a4cf-f10d7bb87a90)




# Workflow
## Training
### Preprocessing
<img width="1428" height="401" alt="training" src="https://github.com/user-attachments/assets/0225468d-ac11-4082-91c8-a587b45bbf86" />

I used two datasets:

[Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
) for training face and mask detection.

[Celebrity Face Image Dataset](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
) for training mask-free areas (covering the lower nose, mouth, chin, left cheek, and right cheek).

You may click [this link](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/tree/main/train) to access the training notebooks. I did the annotations or labelers using [Roboflow](https://app.roboflow.com/ihsanroboflownew/face-mask-detection-13ygd/)

<img width="1905" height="879" alt="image" src="https://github.com/user-attachments/assets/96176c27-da43-4069-bd0b-20ae8a547638" />

The model was trained using YOLOv26 since it is the latest YOLO-pytorch model that newestly launch on mid January 2026. YOLO is commonly use for object detection task and already been trained using [80 label COCO datasets](https://pypi.org/project/ultralytics/).

The models were trained for maximum 100 epochs with image augmentation resize into 320x320 px. The results of metrics evaluation mention as below:

- Face Detection (yolov26m): accuracy mAP50 = 92%, accuracy mAP50-95 = 64.1%, precision = 94.4%, recall = 84.7%
- Mask Detection (yolov26n): accuracy mAP50 = 98.8%, accuracy mAP50-95 = 69.5%, precision = 97.6%, recall = 96.2%
- Mask-Free Detection (yolov26s): accuracy mAP50 = 99.5%, accuracy mAP50-95 = 77.6%, precision ~ 99%, recall = 90.9%

This results shows that different distribution and quantity of images is affected the model quality

<img width="1408" height="1408" alt="image" src="https://github.com/user-attachments/assets/08806c42-58a4-4923-af7d-6862b0a6394d" />
<img width="1408" height="1408" alt="image" src="https://github.com/user-attachments/assets/4b5dec2a-54d7-404d-aa68-29cb992e88c9" />

### Convert To TFLite
In order to support mobile-edge development, I do compressing the model into TFLite support using onnx format first (https://pinto0309.github.io/onnx2tf/). Acknowledged that this program won't be work on recent library, I suggest to create different environment that support this dependencies below:

Version **Python 3.10**

Libraries:
```
onnx2tf
onnx-graphsurgeon
ai_edge_litert 
sng4onnx
onnx==1.15.0 
ml-dtypes==0.3.2
tensorflow
tf_keras
tf_keras
ultralytics
numpy==1.26.4
```
The convertor script can be access [here](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/train/convert_to_tflite/convert_to_tflite.py) 


## Inference
### Face Detection

