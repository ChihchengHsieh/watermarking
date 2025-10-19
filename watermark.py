from torchvision import transforms  
import torch

def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0
