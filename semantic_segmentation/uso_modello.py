import segmentation_models_pytorch as smp
import torch

num_classes = 10  
device = "cuda" if torch.cuda.is_available() else "cpu"


# Ricrea modello con stessa architettura
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,   # o "imagenet" se vuoi partire da pretrain
    classes=num_classes,
    activation=None
)
model.load_state_dict(torch.load("unet_weights.pth"))
model.to(device)
model.eval()  # fondamentale per inference
