import cv2
import json
from tqdm import tqdm
from preprocess import compute_brightness, prepare_input
from model_loader import Model

YOLO_MODEL = "./models/yolov8.om"
PICO_MODEL = "./models/pp_picodet.om"

def main():
    cap = cv2.VideoCapture("./data/testvideo01.mp4")

    brightness_values = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        brightness_values.append(compute_brightness(frame))
    avg_brightness = sum(brightness_values) / len(brightness_values)
    use_yolo = avg_brightness > 80
    print(f"[INFO] 平均亮度: {avg_brightness:.2f} -> 使用: {'YOLOv8' if use_yolo else 'PP-PicoDet'}")

    model_path = YOLO_MODEL if use_yolo else PICO_MODEL
    model = Model(model_path)
    model.init()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    results = []
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break
        input_array = prepare_input(frame)
        output = model.infer(input_array)
        results.append(output.tolist())

    with open("./output/result.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[INFO] 推理结果已保存到 output/result.json")

    model.release()

if __name__ == "__main__":
    main()
