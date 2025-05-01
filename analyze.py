import torch
import cv2
import numpy as np
from torchvision.models.video import r2plus1d_18

def analyze(video_path):
    model = r2plus1d_18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 1),
        torch.nn.Sigmoid()
    )
    weights_path = "/Users/mac/cardio-echo/weights/r2plus1d_18_official.pth"
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict.pop('fc.0.weight', None)
    state_dict.pop('fc.0.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))  # [112, 112, 3]
        frame = frame / 255.0  # Normalisation
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("No frames extracted")

    # Convertir en array : [num_frames, 112, 112, 3]
    frames = np.array(frames)
    # Transposer pour obtenir [num_frames, 3, 112, 112]
    frames = frames.transpose(0, 3, 1, 2)
    # Convertir en tenseur et ajouter batch : [1, 3, num_frames, 112, 112]
    frames = torch.tensor(frames, dtype=torch.float32)
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 3, num_frames, 112, 112]

    with torch.no_grad():
        lvef = model(frames).item() * 100

    return f"Echocardiography video processed, predicted LVEF: {lvef:.1f}%"

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    print(analyze(video_path))
