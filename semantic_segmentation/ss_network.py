import torch
from u_net import UNet
from deep_labv3_plus import get_deeplabv3plus_model

# Scegli uno dei due modelli
model = UNet(in_channels=3, out_channels=8)
# oppure:
# model = get_deeplabv3plus_model(num_classes=8)

# Simula batch di input
input_tensor = torch.randn(4, 3, 128, 128)
output = model(input_tensor)  # [4, 8, 128, 128]

# Definizione della loss e target fittizio
loss_fn = torch.nn.CrossEntropyLoss()
target = torch.randint(0, 8, (4, 128, 128))  # classi da 0 a 7
loss_value = loss_fn(output, target)

print("Loss:", loss_value.item())
