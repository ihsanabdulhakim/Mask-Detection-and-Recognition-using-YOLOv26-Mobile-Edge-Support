import argparse
import cv2
import os
import csv
from src.pipeline import detection_pipeline, generate_mask_pipeline, embedding_pipeline

output_dir = "data/mask_recognition/csv/registered_image"
csv_reference = "data/mask_recognition/csv/withoutmask_generated_embed.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Mask and Registering Inference")
    parser.add_argument("--input", required=True, help="Input image")
    return parser.parse_args()

def load_image(imagepath):
    if not os.path.exists(imagepath):
        raise FileNotFoundError(f"Image not found: {imagepath}")

    image = cv2.imread(imagepath)
    if image is None:
        raise ValueError(f"Failed to read image: {imagepath}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image



def save_embedding(overlay_masked, imagename, output_csv):
    face_embed = embedding_pipeline(overlay_masked)
    with open(output_csv, "a", newline="") as f:
        csv.writer(f).writerow([imagename] + face_embed.tolist())



def process_face(img, bbox, imagename, output_img_dir, output_csv):
    x1, y1, x2, y2, label, score = bbox

    if label != "nomask":
        print(f"Skip face ({label})")
        return

    facecrop = img[y1:y2, x1:x2]

    overlay_masked = generate_mask_pipeline(facecrop)
    if overlay_masked is None:
        print("Mask overlay failed, skip embedding")
        return

    overlay_bgr = cv2.cvtColor(overlay_masked, cv2.COLOR_RGB2BGR)
    out_path = f"{output_img_dir}/{imagename}_maskgenerated.jpg"
    cv2.imwrite(out_path, overlay_bgr)
    print(f"Saved image: {out_path}")

    save_embedding(overlay_masked, imagename, output_csv)
    print(f"Saved embed: {output_csv}")


def main():
    args = parse_args()

    imagepath = args.input
    imagename = os.path.splitext(os.path.basename(imagepath))[0]

    img = load_image(imagepath)

    results = detection_pipeline(img)
    if not results:
        print("No face detected.")
        return

    for bbox in results:
        process_face(img,bbox,imagename,output_dir,csv_reference)

    print(f"Done {imagename}")


if __name__ == "__main__":
    main()