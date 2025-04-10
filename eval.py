import os
from argparse import ArgumentParser
import datasets
from PIL import Image


import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, InterpolationMode, ToTensor
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to imagenet-1k validation set")
    parser.add_argument("--name", type=str, default="vil2-tiny")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    return vars(parser.parse_args())


class NoopContext:
    def __enter__(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def collate_fn(examples):
    images = torch.stack([example["jpg"] for example in examples])
    images = images.to(memory_format=torch.contiguous_format).float()
    labels = torch.tensor([example["cls"] for example in examples])
    metadatas = [example["json"] for example in examples]
    return {"images": images, "labels": labels, "metadatas": metadatas}


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
    def img_transform(examples):
        images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples["jpg"]
        ]
        images = [transform_fn(image) for image in images]
        examples["jpg"] = images
        return examples

    dataset = dataset.with_transform(img_transform)

    if os.name != "nt":
        assert len(dataset) == 50000, f"dataset is not ImageNet-1K validation set (len(dataset) = {len(dataset)})"

    print(f"loading model '{name}'")
    model = torch.hub.load("nx-ai/vision-lstm", name)
    model = model.to(device)

    # precision
    if precision == "fp32":
        autocast_ctx = NoopContext()
    elif precision == "fp16":
        autocast_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.float16)
    elif precision == "bf16":
        autocast_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16)
    else:
        raise NotImplementedError

    # iterate over dataset
    print(f"batch_size: {batch_size}")
    print(f"num_workers: {num_workers}")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        # num_workers=num_workers,
    )
    num_correct = 0
    model = model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with autocast_ctx:
                y_hat = model(x).argmax(dim=1)
            num_correct += (y_hat == y).sum()
    accuracy = num_correct / len(dataset)
    print(f"accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main(**parse_args())