import os
from argparse import ArgumentParser
import datasets
from PIL import Image
import gradio as gr
import requests

import torch
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode, ToTensor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to imagenet-1k validation set")
    parser.add_argument("--name", type=str, default="vil2-tiny")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    return vars(parser.parse_args())




def main(data, name, device, batch_size, num_workers, precision):
    dataset = datasets.load_dataset(data, data_files="*valid*")["train"]
    # data = Path(data).expanduser()
    # assert data.exists() and data.is_dir(), f"invalid data path '{data.as_posix()}'"

    # init device
    print(f"using device: {device}")
    device = torch.device(device)

    # init data
    print(f"initializing ImageNet-1K validation set '{data}'")
    transform_fn = Compose(
            [
                Resize(size=224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(size=224),
                ToTensor(),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ],
        )
    label_names = requests.get("https://git.io/JJkYN")
    label_names = label_names.text.split("\n")

    def img_transform(image):
        image = image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB")
        image = transform_fn(image)
        return image

    if os.name != "nt":
        assert len(dataset) == 50000, f"dataset is not ImageNet-1K validation set (len(dataset) = {len(dataset)})"

    print(f"loading model '{name}'")
    model = torch.hub.load("nx-ai/vision-lstm", name)
    model = model.to(device).eval()

    # iterate over dataset
    def predict(img):
        img = img_transform(img)
        x = img.unsqueeze(0).to(device, non_blocking=True, memory_format=torch.contiguous_format).float()
        with torch.no_grad(), torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            prediction = model(x).softmax(-1)[0]
            confidences = {label_names[i]: float(prediction[i]) for i in range(1000)}
        return confidences

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        examples=dataset.select(range(100))["jpg"],
    )
    demo.launch()


if __name__ == "__main__":
    main(**parse_args())