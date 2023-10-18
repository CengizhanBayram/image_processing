import torch
from torchvision import models, transforms
from PIL import Image

# İndirilmiş bir modeli yükle (örneğin, ResNet-18)
model = models.resnet18(pretrained=True)
model.eval()

# Görüntüyü yükle
image_path = 'test_image.jpg'
image = Image.open(image_path)

# Görüntüyü modelin giriş boyutlarına dönüştür
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# Görüntüyü model üzerinden geçir
with torch.no_grad():
    output = model(input_batch)

# Modelin tahmin ettiği sınıfı bul
_, predicted_idx = torch.max(output, 1)
