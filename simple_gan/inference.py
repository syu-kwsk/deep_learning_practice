import torch
from torchvision.utils import save_image

import model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_g = model.Generator().to(device)
state_dict = torch.load("./model_g.pth")
model_g.load_state_dict(state_dict)

const_z = torch.randn(64, 128, 1, 1).to(device)
model_g.eval()

result_img = model_g(const_z)
save_image(result_img, "result.jpg")
