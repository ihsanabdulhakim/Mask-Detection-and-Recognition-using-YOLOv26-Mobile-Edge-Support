import argparse
import cv2
import os
from src.pipeline import detection_pipeline


IMAGE_EXTS = (".jpg", ".png", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


output_dir = "data/mask_detection/output"

def parse_args():
    parser = argparse.ArgumentParser(description="Face Mask Detection Inference")
    parser.add_argument("--input", required=True, help="Input image / folder / video")
    return parser.parse_args()

def draw_boxes(image, results):
    for x1, y1, x2, y2, label, score in results:
        text = f"{label} {score}"
        position_text = (x1, y1-10)
        box_h = y2 - y1
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.3, min(1.0, box_h / 150))
        thickness = max(1, int(box_h / 120))
        color = (120, 255, 0) if label == "mask" else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image,text,position_text, font, font_scale, color, thickness)
    return image

def process_image(imagepath, output_dir):
    image = cv2.imread(imagepath)
    if image is None:
        print(f"Failed to read {imagepath}")
        return

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detection_pipeline(img_rgb)

    image = draw_boxes(image, results)

    name = os.path.splitext(os.path.basename(imagepath))[0]
    out_path = os.path.join(output_dir, f"{name}_detected.jpg")
    cv2.imwrite(out_path, image)

    print(f"Image saved: {out_path}")

def process_video(videopath, output_dir):
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print(f"Failed to open video {videopath}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    name = os.path.splitext(os.path.basename(videopath))[0]
    out_path = f"{output_dir}/{name}_detected.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"Processing video {videopath}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detection_pipeline(frame_rgb)

        frame = draw_boxes(frame, results)
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Video saved: {out_path}")

def main():
    args = parse_args()
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    else:
        files = [args.input]

    for path in files:
        ext = path.lower()

        if ext.endswith(IMAGE_EXTS):
            process_image(path, output_dir)

        elif ext.endswith(VIDEO_EXTS):
            process_video(path, output_dir)

        else:
            print(f"Unsupported file: {path}")


if __name__ == "__main__":
    main()