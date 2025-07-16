import cv2
import numpy as np

def compute_brightness(frame):
    return frame.mean()

def prepare_input(frame, input_size=(640, 640)):
    resized = cv2.resize(frame, input_size)
    input_data = resized.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)  # NCHW or NHWC，按模型来
    return input_data
