# brain-tumor-mvp
Brain MRI Tumor Classification MVP


File structure 1

brain_tumor_mvp/
│
├── app/
│   └── app.py
├── src/
│   ├── __init__.py
│   └── inference/
│       ├── __init__.py
│       ├── predict.py
│       └── gradcam.py
├── brain_tumor_resnet18.pth
├── classes.json
└── requirements.txt



File structure 2

brain_tumor_mvp/
│
├── app/
│   ├── app.py                  # Streamlit frontend
│   ├── predict.py              # Model loading & prediction
│   └── gradcam.py              # Grad-CAM helper (optional)
│
├── brain_tumor_resnet18.pth    # Trained model from Kaggle
├── classes.json                # Class index mapping
└── requirements.txt            # Python packages


