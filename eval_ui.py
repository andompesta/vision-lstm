import os
from argparse import ArgumentParser
import datasets
from PIL import Image
import gradio as gr
import requests

import torch
import itertools
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

    iter_dataset = itertools.cycle(dataset.select(range(100)))

    # iterate over dataset
    def get_next_item():
        elem = next(iter_dataset)

        img = elem["jpg"].convert("RGB")
        x = transform_fn(img)
        x = x.unsqueeze(0).to(device, non_blocking=True, memory_format=torch.contiguous_format).float()
        with torch.no_grad(), torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16):
            prediction = model(x).softmax(-1)[0]
            confidences = {label_names[i]: float(prediction[i]) for i in range(1000)}
        return img, confidences

    # Build the Gradio Blocks interface.
    with gr.Blocks(
        head="""
            <script>
            // Function that triggers the click event on the button
            function continuouslyClickButton() {
                const button = document.getElementById("next-button");
                if (button) {
                    button.click();
                } else {
                    console.warn("Button not found");
                }
            }

            // Set the interval (e.g., every 1000 milliseconds)
            setInterval(continuouslyClickButton, 100);
            </script>
            """
    ) as demo:
        gr.Markdown(value="## Auto-Updating Image and Prediction Demo")

        # Output components: one for the image and one for the text prediction.
        image_component = gr.Image(label="Image", type="pil")
        prediction_component = gr.Label(label="Prediction", num_top_classes=3)

        # A hidden button that triggers the update.
        next_button = gr.Button("Next", visible=True, elem_id="next-button")
        next_button.click(fn=get_next_item, inputs=[], outputs=[image_component, prediction_component])

    # Launch the demo.
    demo.launch()


if __name__ == "__main__":
    main(**parse_args())