# src/inference/predict.py
import torch
from torchvision import transforms, models
from PIL import Image
import json

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "brain_tumor_resnet18.pth"
CLASS_IDX_PATH = "classes.json"

# Load class index
with open(CLASS_IDX_PATH) as f:
    class_idx = json.load(f)

# Image preprocessing
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

# Load trained model
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict function
def predict(image_path, model):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1).squeeze().tolist()
        pred_idx = int(torch.tensor(probs).argmax().item())
    
    return {"class": class_idx[str(pred_idx)], "probs": probs}
