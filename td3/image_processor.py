import torchvision.transforms as transforms
from PIL import Image

# define the steps of image processor
preprocess = transforms.Compose([
    transforms.Resize((84, 84)),  # resolution transfor
    transforms.ToTensor(),       # trans to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalized
])

def process_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0)  # add one dim to match batch's dim
