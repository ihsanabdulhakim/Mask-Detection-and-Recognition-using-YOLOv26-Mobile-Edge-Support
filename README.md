# About
A comprehensive pipeline for real-time face mask detection and recognition using YOLOv26, optimized for mobile and edge devices. This project includes model training, ONNX and TFLite export, and deployment-ready pipelines for CPU/GPU inference with optional quantization for low-resource environments.

![mask2_detected](https://github.com/user-attachments/assets/eeccb634-ba4d-4d61-8bff-c35c838b73f6)
![rdj_recognition](https://github.com/user-attachments/assets/87e29622-8ff7-48b9-831b-e48ca6f7b8d1)
![girl_recognition](https://github.com/user-attachments/assets/93f92913-2e0b-4b53-a4cf-f10d7bb87a90)


# Prerequisites
Make sure you already secure by following this particular environment:

run ```git clone https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support.git```

then run this script ```cd .\Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support```

already installed environment **python 3.10**

run ```pip install -r requirements.txt```
make sure the libraries follow successfully installed:
```
tensorflow
ultralytics
opencv-python
matplotlib
pillow
```

for running training montage, you may use CLI or notebook for better experiences.

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
### Mask-face Detection
<img width="1243" height="723" alt="maskfacedetection" src="https://github.com/user-attachments/assets/74206db2-43e8-4d7f-9024-faefb1101038" />

The workflow of the mask-face detection is consist by usage of two object detections; face detection and mask detection. The steps conclude of raw image reading, preprocessing, detecting, cropping, and making decision based on the score value of the models. The scripts need an input of image path that be address to running the scripts as this below:
```
python run.py --input {input_path)
```
for example, the location of the image is [this image](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/data/mask_detection/input/mask1.jpg), you may run this scripts to inference the program:
```
python run.py --input .\data\mask_detection\input\mask1.jpg
```
The results gonna be located into [".\data\mask_detection\output\"](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/tree/main/data/mask_detection/output)

We can also insert video here by this example:
```
python run.py --input .\data\mask_detection\input\shibuyacrossing.mp4
```
Once the program succes, you may check the detection results in [".\data\mask_detection\output\shibuyacrossing_detected.mp4"](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/data/mask_detection/output/shibuyacrossing_detected.mp4)

### Mask Generator and Face-Mask Embedding Registration
<img width="1609" height="416" alt="maskfacegenerator" src="https://github.com/user-attachments/assets/6a561ee0-b9d0-4032-8ff8-6dd4068249e5" />

This program focus on generating mask for mask recognition later. The strategy is by finding the mask-free area of no-masked face then integrating by [mask vector](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/recognizer/mask.png). 

<img width="936" height="459" alt="image" src="https://github.com/user-attachments/assets/326332bc-a79f-4d82-ae71-780d13b39766" />

After the mask-free area is detected, I inserted the mask into the remain mask-free bounding box that follows the original raw picture's scale. Now the image is ready to convert as face-embedding. Here, I used pretrained FaceNet [20180402-114759] (https://github.com/davidsandberg/facenet/tree/master). I use this model as the face templates for matching similarity purpose only. This model had been built through combination of [Inception and Resnet layers](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py). Converted embedding value received by FaceNet will be sent to remain database CSV format act as the reference images ["./data/mask_recognition/csv/withoutmask_generated_embed.csv"](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/data/mask_recognition/csv/withoutmask_generated_embed.csv).

To run this program, we can use this scripts below for this [image example](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/data/mask_recognition/csv/unregistered_image/nicolekidman.jpg)
```
python .\generatemask_reference.py --input .\data\mask_recognition\csv\unregistered_image\nicolekidman.jpg
```
The image will be generated in directory [".\data\mask_recognition\csv\registered_image\"](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/tree/main/data/mask_recognition/csv/registered_image)
![nicolekidman](https://github.com/user-attachments/assets/049975a7-9cf2-4377-af39-9e2f1503bc08) ![nicolekidman_maskgenerated](https://github.com/user-attachments/assets/5200d2ba-2f43-4680-8290-f3701f97302c)

Then, the face embedding will be conceived to [this dataframe](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/blob/main/data/mask_recognition/csv/withoutmask_generated_embed.csv)



### Mask-face Recognition
<img width="1503" height="723" alt="maskfacesimilarity" src="https://github.com/user-attachments/assets/1fecd44e-7803-447a-bb03-363c68071b0a" />
The mask-face recognition follows the similarity rules. Here, I used cosine similarity as the approach to calculate best match feature between two embedding generated by FaceNet model.


<img width="455" height="164" alt="image" src="https://github.com/user-attachments/assets/8370e253-6b00-4642-9caf-140aefc39095" />


The score of cosine similarity indicates how closely every feature between two images. Because of that, we need to calculate every picture from the dataframe and compare to the test dataset. The idea of this is the mask already covering mouth-nose-cheeks area so we need to compare the image with same condition. Let's take a look at this scheme below:
<img width="1038" height="514" alt="image" src="https://github.com/user-attachments/assets/91d15797-7c88-4caa-9869-3c50df38e7eb" />

The scenarios follow the rules of high value becomes the winner. The more that scores is going to achieved, it means the more that similarity getting approach as the same person. If the similarity score already passing the threshold and become the maximum value, then the program will be conclude this person is similar with the best_max person from dataframe. Therefore, the picture that contain blur feature or not having any score that surpassing the threshold indicates that face-masked have not yet registered. By this case, I labelled them as **"unregister"**.
![leonardo_recognition](https://github.com/user-attachments/assets/b0bf00a0-a67f-4fc9-8d21-9dbda53fffb4)

To run this program, you may use this scripts below:
```
python .\mask_recognition.py --input .\data\mask_recognition\input\leonardo.jpg
```
The results is located into [".\data\mask_recognition\output\"](https://github.com/ihsanabdulhakim/Mask-Detection-and-Recognition-using-YOLOv26-Mobile-Edge-Support/tree/main/data/mask_recognition/output)

# Evaluation and Future Works
Besides the logics that already being created, there are still few things that may be noticed. Since the accuracy is not yet optimized, I suggest to increasing and exploring more spectrum of datasets in order to anticipated overfitting and misclassified labels. The numbers of datasets is one of the key to improving the quality of models accuracy since they may have recognized any situation of the datasets. The uses of image classification can be establish as the alternative of current mask detection that probably giving more accurate results for classifying mask - no mask samples. 

For the future works that probably able to be realized, system such as counting mask-no mask attendance, mask-face tracking id, or even better detection for mask-face in different term situation such as in night sky, low light, incorrect or improper use of mask on face. 













