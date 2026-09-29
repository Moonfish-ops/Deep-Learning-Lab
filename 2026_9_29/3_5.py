"""教材 3.5：下载、读取并可视化 Fashion-MNIST。"""

from pathlib import Path
import time

import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils import data
from torchvision import transforms


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
OUTPUT_IMAGE = Path(__file__).resolve().parent / "fashion_mnist_samples.png"


def get_fashion_mnist_labels(labels):
    text_labels = [
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot",
    ]
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """用 Pillow 绘制样本，避免某些 Anaconda 环境的 OpenMP 冲突。"""
    cell_width = int(80 * scale)
    image_height = int(80 * scale)
    title_height = 22
    canvas = Image.new(
        "RGB", (num_cols * cell_width, num_rows * (image_height + title_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for i, img in enumerate(imgs[: num_rows * num_cols]):
        row, col = divmod(i, num_cols)
        pil_img = transforms.ToPILImage()(img).resize((image_height, image_height))
        x = col * cell_width + (cell_width - image_height) // 2
        y = row * (image_height + title_height)
        canvas.paste(pil_img.convert("RGB"), (x, y))
        if titles:
            draw.text((col * cell_width + 3, y + image_height + 3), titles[i], fill="black")
    return canvas


def get_dataloader_workers():
    # Windows 脚本中设为 0 最稳妥；教材使用 4 个进程。
    return 0


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(trans)
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, transform=transform, download=True
    )
    return (
        data.DataLoader(train_set, batch_size, shuffle=True,
                        num_workers=get_dataloader_workers()),
        data.DataLoader(test_set, batch_size, shuffle=False,
                        num_workers=get_dataloader_workers()),
    )


def main():
    train_iter, test_iter = load_data_fashion_mnist(256)
    print("训练集样本数:", len(train_iter.dataset))
    print("测试集样本数:", len(test_iter.dataset))

    start = time.perf_counter()
    X, y = next(iter(train_iter))
    elapsed = time.perf_counter() - start
    print("一个批次的形状:", X.shape, y.shape)
    print(f"读取首批耗时: {elapsed:.3f} 秒")

    image = show_images(
        X[:18].reshape(18, 28, 28), 2, 9,
        titles=get_fashion_mnist_labels(y[:18]),
    )
    image.save(OUTPUT_IMAGE)
    print("样本图已保存到:", OUTPUT_IMAGE)

    resized_train, _ = load_data_fashion_mnist(32, resize=64)
    resized_X, resized_y = next(iter(resized_train))
    print("resize=64 后:", resized_X.shape, resized_X.dtype,
          resized_y.shape, resized_y.dtype)


if __name__ == "__main__":
    main()
