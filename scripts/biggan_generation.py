# pip install pytorch-pretrained-biggan scipy pillow

import click
import logging

logging.basicConfig(level=logging.INFO)
import numpy as np
import os
import s3fs
import torch

from joblib import Parallel, delayed
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm, trange

from numpy.typing import NDArray
from torch.types import Tensor

from pytorch_pretrained_biggan import BigGAN, one_hot_from_int, truncated_noise_sample
from pytorch_pretrained_biggan.model import PRETRAINED_MODEL_ARCHIVE_MAP

MODELS = list(PRETRAINED_MODEL_ARCHIVE_MAP.keys())


def convert_to_images(tensor: NDArray | Tensor) -> list[Image.Image]:
    """Convert an output tensor from BigGAN in a list of images.
    Params:
        tensor: tensor or numpy array of shape (batch_size, channels, height, width)
    Output:
        list of Pillow Images of size (height, width)
    """

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().numpy()

    tensor = tensor.transpose((0, 2, 3, 1))
    tensor = np.clip(((tensor + 1) / 2.0) * 256, 0, 255)

    img = [Image.fromarray(np.asarray(np.uint8(out), dtype=np.uint8)) for out in tensor]
    return img


def save_batch(
    output: NDArray | Tensor,
    batch_idx: int,
    class_indices: NDArray | Tensor,
    batch_size: int,
    output_folder: Path,
    s3: s3fs.S3FileSystem | None = None,
):
    # Convert as images
    images = convert_to_images(output)

    for sample_idx, image in enumerate(images):
        image_idx = batch_idx * batch_size + sample_idx
        class_idx = class_indices[sample_idx]
        filename = output_folder / f"{image_idx:06d}.png"

        if s3:  # Save to S3
            with s3.open(filename, "wb") as f:
                image.save(f, format="PNG")
        else:  # Save to local disk
            image.save(output_folder / f"{image_idx:06d}_{class_idx=}.png")


def generate_batches(
    model: BigGAN,
    n_batches: int,
    batch_size: int,
    output_folder: Path,
    truncation: float = 0.4,
    device: str = "cuda",
    last_batch: int | None = None,
    fs: s3fs.S3FileSystem | None = None,
):
    if last_batch and last_batch > 0:
        n_batches += 1

    for batch_idx in trange(n_batches):
        if batch_idx == (n_batches - 1) and last_batch:
            batch_size = last_batch

        # Prepare a input
        class_indices = np.random.randint(0, 1000, size=batch_size)
        class_vector = one_hot_from_int(class_indices, batch_size=batch_size)
        noise_vector = truncated_noise_sample(
            truncation=truncation, batch_size=batch_size
        )

        # All in tensors
        noise_vector = torch.from_numpy(noise_vector)
        class_vector = torch.from_numpy(class_vector)

        # If you have a GPU, put everything on cuda
        noise_vector = noise_vector.to(device)
        class_vector = class_vector.to(device)

        # Generate an image
        with torch.no_grad():
            output = model(noise_vector, class_vector, truncation)

        # If you have a GPU put back on CPU
        output = output.to("cpu")

        yield delayed(save_batch)(
            output,
            batch_idx,
            class_indices,
            batch_size,
            output_folder,
            s3=fs,
        )


@click.command()
@click.argument("n_samples", type=int)
@click.option(
    "--arch",
    type=click.Choice(MODELS, case_sensitive=False),
    default="biggan-deep-256",
    help="BigGAN architecture to use.",
)
@click.option(
    "--truncation", type=float, default=0.4, help="Truncation value for BigGAN."
)
@click.option("--batch-size", type=int, default=48, help="Batch size for generation")
@click.option(
    "--output-folder",
    type=click.Path(writable=True, file_okay=False, dir_okay=True),
    default=Path("biggan-imagenet-256-samples"),
    help="Output folder to save generated images.",
)
@click.option(
    "--s3-endpoint-url",
    type=str,
    default=None,
    help="S3 endpoint URL. If provided, images will be saved to this S3 bucket.",
)
@click.option(
    "--s3-bucket",
    type=str,
    default=None,
    help="S3 bucket name. If provided, images will be saved to this S3 bucket.",
)
@click.option(
    "--n-jobs",
    type=int,
    default=12,
    help="Number of parallel jobs for saving images to disk.",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for generation (e.g., 'cuda' or 'cpu'). If not provided, will use 'cuda' if available, otherwise 'cpu'.",
)
def generate_biggan_samples(
    n_samples: int,
    arch: str = "biggan-deep-256",
    truncation: float = 0.4,
    batch_size: int = 48,
    output_folder: Path = Path("biggan-imagenet-256-samples"),
    s3_endpoint_url: str | None = None,
    s3_bucket: str | None = None,
    n_jobs: int = 12,
    seed: int = 42,
    device: str | None = None,
):
    # Load pre-trained model tokenizer (vocabulary)
    model = BigGAN.from_pretrained(arch)

    # Use CUDA if available
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    if s3_endpoint_url is not None and s3_bucket is not None:
        # Create filesystem object
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint_url})
        output_folder = Path(s3_bucket) / output_folder
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_batches = n_samples // batch_size
    last_batch = n_samples % batch_size

    model.to(device)
    model.eval()

    # Use joblib to parallelize saving images to disk
    with Parallel(
        n_jobs=n_jobs,
        backend="threading",
        return_as="generator",
    ) as parallel:
        # The main thread generates batches while worker threads
        # convert/save previously generated batches.
        for _ in parallel(
            generate_batches(
                model,
                n_batches,
                batch_size,
                output_folder,
                last_batch=last_batch,
                truncation=truncation,
                device=device,
                fs=fs if s3_endpoint_url and s3_bucket else None,
            )
        ):
            pass

    logging.info(
        f"Generated {n_samples} samples and saved to {s3_bucket}/{output_folder}"
        if s3_bucket
        else f"Generated {n_samples} samples and saved to {output_folder}"
    )


if __name__ == "__main__":
    generate_biggan_samples()
