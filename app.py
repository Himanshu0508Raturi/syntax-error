import io
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn


MODEL_PATH  = "resnet50_disaster.pth"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    "Damaged_Infrastructure", "Fire_Disaster", "Human_Damage",
    "Land_Disaster", "Non_Damage", "Water_Disaster",
]
NUM_CLASSES = len(CLASS_NAMES)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

INTRA_TIERS = ["Contained", "Moderate", "Severe", "Catastrophic"]

TIER_THRESHOLDS = [0.35, 0.65, 0.85]   

INTRA_RATIONALE = {
    "Contained":    "Limited visual extent; event appears small or early-stage.",
    "Moderate":     "Partial coverage detected; response recommended.",
    "Severe":       "Large area affected; significant visual damage markers present.",
    "Catastrophic": "Near-total image coverage or extreme intensity; immediate action required.",
}


def _fire_proxy(img_rgb: np.ndarray) -> float:
    """
    High-intensity orange/red pixels in HSV space.
    A large inferno has many bright, saturated warm pixels.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, (0,  120, 100), (25,  255, 255))
    mask2 = cv2.inRange(hsv, (160, 120, 100), (180, 255, 255))
    fire_ratio = (mask1.sum() + mask2.sum()) / (255.0 * img_rgb.shape[0] * img_rgb.shape[1])
    brightness = hsv[..., 2].mean() / 255.0
    return float(np.clip(0.6 * fire_ratio + 0.4 * (brightness - 0.3), 0, 1))


def _water_proxy(img_rgb: np.ndarray) -> float:
    """
    Blue/grey desaturated coverage + dark low-areas typical of floods.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    water_mask = cv2.inRange(hsv, (90, 0, 30), (140, 180, 220))
    water_ratio = water_mask.sum() / (255.0 * img_rgb.shape[0] * img_rgb.shape[1])
    darkness = 1.0 - hsv[..., 2].mean() / 255.0
    return float(np.clip(0.55 * water_ratio + 0.45 * darkness, 0, 1))


def _land_proxy(img_rgb: np.ndarray) -> float:
    """
    Brown/grey earth tones + texture variance (rubble, landslide scars).
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((15, 15), np.float32) / 225
    mean_sq  = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
    mean_    = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_std = np.sqrt(np.maximum(mean_sq - mean_**2, 0))
    texture_score = local_std.mean() / 80.0   

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    earth_mask = cv2.inRange(hsv, (5, 20, 30), (35, 180, 200))
    earth_ratio = earth_mask.sum() / (255.0 * img_rgb.shape[0] * img_rgb.shape[1])

    return float(np.clip(0.5 * texture_score + 0.5 * earth_ratio, 0, 1))


def _infrastructure_proxy(img_rgb: np.ndarray) -> float:
    """
    Edge density (shattered structures = many edges) + grey/concrete tone coverage.
    """
    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    edge_density = edges.sum() / (255.0 * gray.shape[0] * gray.shape[1])

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    concrete_mask = cv2.inRange(hsv, (0, 0, 60), (180, 50, 200))
    concrete_ratio = concrete_mask.sum() / (255.0 * img_rgb.shape[0] * img_rgb.shape[1])

    return float(np.clip(0.6 * edge_density * 4 + 0.4 * concrete_ratio, 0, 1))


def _human_proxy(img_rgb: np.ndarray) -> float:
    """
    Skin-tone coverage + scene darkness (night disasters, smoke) + edge chaos.
    Many victims / crowded rescue scenes show more skin tone in frame.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    skin_mask = cv2.inRange(hsv, (0, 20, 70), (25, 180, 255))
    skin_ratio = skin_mask.sum() / (255.0 * img_rgb.shape[0] * img_rgb.shape[1])

    gray  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    chaos = edges.sum() / (255.0 * gray.shape[0] * gray.shape[1])

    return float(np.clip(0.45 * min(skin_ratio * 5, 1.0) + 0.55 * chaos * 3, 0, 1))


def _non_damage_proxy(_img_rgb: np.ndarray) -> float:
    """Non-damage is always a baseline — severity meaningless, return 0."""
    return 0.0


CLASS_VISUAL_PROXY: Dict[str, callable] = {
    "Fire_Disaster":          _fire_proxy,
    "Water_Disaster":         _water_proxy,
    "Land_Disaster":          _land_proxy,
    "Damaged_Infrastructure": _infrastructure_proxy,
    "Human_Damage":           _human_proxy,
    "Non_Damage":             _non_damage_proxy,
}


class GradCAM:
    """Minimal GradCAM for ResNet-50 last conv layer."""

    def __init__(self, model: nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None
        target_layer = model.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute(self, tensor: torch.Tensor, class_idx: int) -> float:
        """
        Returns a scalar [0,1] representing the fraction of the
        feature map strongly activated for `class_idx`.
        """
        self.model.zero_grad()
        output = self.model(tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = torch.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        spread = float((cam > 0.5).float().mean())
        return spread



def load_model() -> Tuple[nn.Module, GradCAM]:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE)
    m.train(False)      
    return m, GradCAM(m)


resnet, gradcam = load_model()


def _tier_from_score(score: float) -> str:
    for i, threshold in enumerate(TIER_THRESHOLDS):
        if score < threshold:
            return INTRA_TIERS[i]
    return INTRA_TIERS[-1]


def compute_intra_severity(
    predicted_class: str,
    confidence: float,
    probs: torch.Tensor,
    tensor: torch.Tensor,
    img_rgb: np.ndarray,
) -> Tuple[str, float, str]:
    """
    Returns (tier, score_0_to_1, rationale_string).

    Score = weighted combination of three signals:
      w1=0.40  GradCAM spread       — how much of the scene is activated
      w2=0.45  Class visual proxy   — class-specific CV heuristic
      w3=0.15  Confidence margin    — top-1 minus top-2 (model certainty)
    """
    if predicted_class == "Non_Damage":
        return "Contained", 0.0, "No disaster detected; severity not applicable."

    with torch.enable_grad():
        gradcam_score = gradcam.compute(tensor.clone().requires_grad_(True), 
                                        CLASS_NAMES.index(predicted_class))

    proxy_fn    = CLASS_VISUAL_PROXY.get(predicted_class, _non_damage_proxy)
    proxy_score = proxy_fn(img_rgb)

    sorted_probs   = probs.sort(descending=True).values
    conf_margin    = float(sorted_probs[0] - sorted_probs[1])  
    margin_score   = np.clip(conf_margin * 1.5, 0, 1)          

    raw_score = (0.40 * gradcam_score
               + 0.45 * proxy_score
               + 0.15 * float(margin_score))

    if confidence < 0.50:
        raw_score = 0.5 * raw_score + 0.5 * 0.35   
    
    final_score = float(np.clip(raw_score, 0.0, 1.0))
    tier        = _tier_from_score(final_score)

    rationale = (
        f"{INTRA_RATIONALE[tier]} "
        f"[GradCAM spread: {gradcam_score:.2f}, "
        f"visual proxy: {proxy_score:.2f}, "
        f"confidence margin: {conf_margin:.2f}]"
    )
    return tier, round(final_score, 4), rationale



app = FastAPI(
    title="Disaster Classification + Intra-Class Severity API",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class ClassScore(BaseModel):
    label:      str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_class:         str
    confidence:              float
    intra_class_severity:    str    
    intra_severity_score:    float  
    intra_severity_rationale:str    
    all_scores:              List[ClassScore]


def preprocess(image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot decode image. Use JPG/PNG.")
    img_rgb = np.array(img)
    tensor  = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return tensor, img_rgb


@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "model": "ResNet-50 + GradCAM severity",
            "classes": CLASS_NAMES, "docs": "/docs"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "device": str(DEVICE)}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Use image/jpeg or image/png.")

    tensor, img_rgb = preprocess(await file.read())

    with torch.no_grad():
        logits = resnet(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_idx   = int(probs.argmax())
    top_label = CLASS_NAMES[top_idx]
    top_conf  = float(probs[top_idx])

    intra_tier, intra_score, intra_rationale = compute_intra_severity(
        predicted_class = top_label,
        confidence      = top_conf,
        probs           = probs,
        tensor          = tensor,
        img_rgb         = img_rgb,
    )

    all_scores = sorted(
        [ClassScore(label=CLASS_NAMES[i], confidence=round(float(probs[i]), 4))
         for i in range(NUM_CLASSES)],
        key=lambda x: x.confidence, reverse=True,
    )

    return PredictionResponse(
        predicted_class          = top_label,
        confidence               = round(top_conf, 4),
        intra_class_severity     = intra_tier,
        intra_severity_score     = intra_score,
        intra_severity_rationale = intra_rationale,
        all_scores               = all_scores,
    )