import tensorflow as tf
import numpy as np
import cv2

def generate_mask(bg, png, bbox):
    mask_img = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    mask_img = cv2.cvtColor(mask_img[:, :, :3], cv2.COLOR_BGR2RGB)
    mask_img = np.dstack([mask_img, cv2.imread(png, cv2.IMREAD_UNCHANGED)[:, :, 3]])
    x1, y1, x2, y2 = bbox
    h, w = bg.shape[:2]

    # clamp biar aman
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return bg

    png_resized = cv2.resize(mask_img, (x2 - x1, y2 - y1))

    alpha = png_resized[:, :, 3] / 255.0
    for c in range(3):
        bg[y1:y2, x1:x2, c] = (
            alpha * png_resized[:, :, c] +
            (1 - alpha) * bg[y1:y2, x1:x2, c]
        )

    return bg


def load_models_facenet(modelpath):
    tf.compat.v1.disable_eager_execution()

    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(modelpath, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

    sess = tf.compat.v1.Session(graph=graph)
    images_ph = graph.get_tensor_by_name("input:0")
    embeddings_ph = graph.get_tensor_by_name("embeddings:0")
    phase_ph = graph.get_tensor_by_name("phase_train:0")

    return sess, images_ph, embeddings_ph, phase_ph

def embedding_converter(face_160, modelpath):    
    face = face_160.astype(np.float32) / 255.0
    sess, images_ph, embeddings_ph, phase_ph = load_models_facenet(modelpath)
    emb = sess.run(
        embeddings_ph,
        feed_dict={images_ph: [face], phase_ph: False}
    )
    return emb[0]  # (512,)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0  # handle zero-vector
    return np.abs(dot / (norm_a * norm_b))
