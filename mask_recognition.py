import argparse
import os
import csv
import cv2
import numpy as np

from src.pipeline import detection_pipeline, embedding_pipeline
from src.similarity import cosine_similarity



output_dir = "data/mask_recognition/output"
csv_reference = "data/mask_recognition/csv/withoutmask_generated_embed.csv"
threshold_similarity = 0.4

os.makedirs(output_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Mask Recognition Inference")
    parser.add_argument("--input",required=True,help="Input image path"
    )
    return parser.parse_args()



def load_reference_embeddings(csv_path):
    reference_data = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            name = row[0]
            embedding = np.array(row[1:], dtype=np.float32)
            reference_data.append((name, embedding))

    return reference_data



def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb



def recognizer(face_crop, reference_data):
    best_score = 0
    display_name = "unregister"

    face_embed = embedding_pipeline(face_crop)

    for ref_name, ref_embed in reference_data:
        sim = cosine_similarity(face_embed, ref_embed)

        if sim > best_score and sim >= threshold_similarity:
            best_score = sim
            display_name = ref_name
            print(f"Best match: {display_name} with score {best_score:.2f}")

    return display_name, best_score


def draw_boxes(image, bbox, name, score):
    x1, y1, x2, y2 = bbox

    color = (255, 0, 0) if name != "unregister" else (0, 0, 255)
    text = f"{name} ({score:.2f})" if name != "unregister" else "unregister"
    position_text = (x1, y1-10)
    box_h = y2 - y1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, min(1.0, box_h / 150))
    thickness = max(1, int(box_h / 120))
    thickness = max(1, int(box_h / 120))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image,text,position_text, font, font_scale, color, thickness)


def process_image(image_path, reference_data):
    image, image_rgb = load_image(image_path)

    results = detection_pipeline(image_rgb)

    for x1, y1, x2, y2, label, _ in results:
        if label != "mask":
            draw_boxes(image, (x1, y1, x2, y2), "unregister", 0.0)
            continue

        face_crop = image_rgb[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        name, score = recognizer(face_crop, reference_data)
        draw_boxes(image, (x1, y1, x2, y2), name, score)

        filename = os.path.splitext(os.path.basename(image_path))[0]
        out_path = f"{output_dir}/{filename}_recognition.jpg"

        cv2.imwrite(out_path, image)
        print(f"Output saved to: {out_path}")




def main():
    args = parse_args()

    reference_data = load_reference_embeddings(csv_reference)
    process_image(args.input, reference_data)


if __name__ == "__main__":
    main()
