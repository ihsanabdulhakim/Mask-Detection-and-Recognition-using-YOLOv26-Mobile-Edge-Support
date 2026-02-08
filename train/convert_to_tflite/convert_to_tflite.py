# This run_onnx2tf function uses the onnx2tf CLI
# with flags based on the documentation below:
# https://pinto0309.github.io/onnx2tf/


from ultralytics import YOLO
import subprocess, os

# Load model name
# modelname = "nomaskarea"
# modelname = "face"
modelname = "maskdetection"
model = YOLO(f"./modelpytorch/{modelname}.pt") # Load PyTorch model

# Export PyTorch YOLO to ONNX first
onnx_path = model.export(
    format="onnx",
    opset=12,     # ONNX operator set
    dynamic=False,
    simplify=True,
    imgsz=320,
    device="cpu" #device="cuda" for GPU usage
)

tflite_dir = f"./{modelname}_tflite"

def run_onnx2tf(extra_flags, onnx_path):
    cmd = [
        "onnx2tf",
        "-i", onnx_path,            # input path in ONNX
        "-b", "1",                  # batch size
        "-ois", "(1,3,320,320)",    # input parameter shape
        "-o", tflite_dir            # output
    ] + extra_flags
    return subprocess.call(cmd)

print("Float16") # Float 16 tensor shape
run_onnx2tf(["-oiqt", "-qt", "per-tensor"], onnx_path)

print("Per-channel quant") # Quantisized version
run_onnx2tf(["-oiqt", "-qt", "per-channel"], onnx_path)

prf_json = os.path.join(tflite_dir, "best_auto.json") # Fallback
if not os.path.exists(os.path.join(tflite_dir, "model_float32.tflite")) and os.path.exists(prf_json):
    print("Retrying with JSON auto-fix...")
    run_onnx2tf(["-prf", prf_json, "-oiqt", "-qt", "per-channel"], onnx_path)
