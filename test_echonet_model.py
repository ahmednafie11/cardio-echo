import torch
import torchvision.models.video as video_models

try:
    # Load R2+1D model (placeholder)
    model = video_models.r2plus1d_18(weights="DEFAULT")
    model.eval()
    print("R2+1D model loaded successfully")
except Exception as e:
    print(f"Error: {str(e)}")
