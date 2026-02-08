import tensorflow as tf
import numpy as np

def load_models(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def tflite_objectdetection(interpreter, image, conf_thresh):
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]

    inp = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)
    interpreter.set_tensor(input_det["index"], inp)
    interpreter.invoke()

    preds = np.squeeze(interpreter.get_tensor(output_det["index"]))
    # preds: [N, 6] → x1,y1,x2,y2,conf,cls

    boxes, scores = [], []

    for p in preds:
        x1, y1, x2, y2, conf, _ = p
        if conf < conf_thresh:
            continue

        boxes.append([
            int(x1), int(y1),
            int(x2), int(y2)
        ])
        scores.append(float(conf))

    return boxes, scores