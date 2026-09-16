import click
import s3fs  # for minIO

from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm
from typing import List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.types import Tensor
import torchvision
from torchvision import transforms
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F

from datasets import load_dataset

from torch_fidelity.registry import FEATURE_EXTRACTORS_REGISTRY
from torch_fidelity.feature_extractor_base import FeatureExtractorBase
from torch_fidelity.utils import (
    create_feature_extractor,
    resolve_feature_layer_for_metric,
)

# https://github.com/toshas/torch-fidelity/blob/5e211a950a7b45206bd4976813ffd6aed6cf4ccc/torch_fidelity/registry.py#L172
BACKBONES = list(FEATURE_EXTRACTORS_REGISTRY.keys())


class SimpleImageFolder(Dataset):
    """
    Dataset subclass that reads all images from a local folder.

    Args:
        folder: path to the folder to read from
        extensions: allowed file extensions (default: .png, .jpg)
    """

    def __init__(
        self,
        folder: Path | str,
        extensions: List[str] = [".png", ".jpg"],
    ):
        self.folder = Path(folder)
        self.extensions = extensions

        # List all files in the folder and filter by extension
        self.filelist = [
            f for f in self.folder.iterdir() if f.suffix.lower() in self.extensions
        ]

    def __len__(self) -> int:
        # Helper length function
        return len(self.filelist)

    def __getitem__(self, idx: int) -> Tensor:
        # Read item at position idx in the filelist
        filename = self.filelist[idx]
        im = Image.open(filename)
        im.load()
        # Convert to torch.Tensor
        return F.pil_to_tensor(im)

class MinIODataset(Dataset):
    """
    Dataset subclass that reads all images from an S3 bucket.

    Args:
        s3_endpoint_url: URL to the S3 endpoint
        s3_bucket: path to the bucket to read from
        extensions: allowed file extensions (default: .png, .jpg)
    """

    def __init__(
        self,
        s3_endpoint_url: str,
        s3_bucket: Path | str,
        extensions: List[str] = [".png", ".jpg"],
    ):
        self.s3_endpoint = s3_endpoint_url
        self.extensions = extensions
        self.s3_bucket = s3_bucket

        # Instantiate connection to S3 bucket
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_endpoint_url})

        # List all s3 files and filter by extension
        s3_files = fs.ls(s3_bucket)
        self.filelist = self._list_files()

        # Remove fs object for fork-safe multiprocessing
        self.fs = None

    def _list_files(self):
        # Helper function to list allowed files in the bucket
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": self.s3_endpoint})

        files = fs.ls(self.s3_bucket)

        return [file for file in files if Path(file).suffix.lower() in self.extensions]

    def __len__(self) -> int:
        # Helper length function
        return len(self.filelist)

    def _get_fs(self):
        # Lazy loading of the S3 file system for fork-compatibility
        if self.fs is None:
            self.fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": self.s3_endpoint}
            )
        return self.fs

    def __getitem__(self, idx: int) -> Tensor:
        # Read item at position idx in the filelist
        filename = self.filelist[idx]
        with self._get_fs().open(filename, "rb") as f:
            im = Image.open(f)
            im.load()
        # Convert to torch.Tensor
        return F.pil_to_tensor(im)

# Modified from torch-fidelity
def extract_features_from_dataset(
    input: SimpleImageFolder | MinIODataset | VisionDataset,
    feat_extractor: FeatureExtractorBase,
    batch_size: int = 32,
    cuda: bool = True,
    verbose: bool = True,
    num_workers: int = 0,
) -> Tensor:
    """
    Extract features from a dataset using a feature extractor.
    
    Args:
        input: Dataset to extract features from. Can be a SimpleImageFolder, MinIODataset, or a torchvision VisionDataset.
        feat_extractor: torch-fidelity backbone model to use for feature extraction.
        batch_size: Batch size for feature extraction.
        cuda: Whether to use GPU for feature extraction.
        verbose: Whether to print progress information.
        num_workers: Number of workers for data loading. Set to 0 for no multiprocessing (default).

    Returns:
        out: Dictionary of extracted features, where keys are feature names and values are tensors of shape (N, D),
             with N being the number of samples and D the feature dimension.
    """

    
    if batch_size > len(input):
        batch_size = len(input)

    dataloader = DataLoader(
        input,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=cuda,
    )

    out = None

    with tqdm(
        disable=not verbose,
        leave=False,
        unit="samples",
        total=len(input),
        desc="Processing samples",
    ) as t, torch.no_grad():
        for bid, batch in enumerate(dataloader):

            # Ignore labels if present (e.g., in ImageFolder)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            elif isinstance(batch, dict):
                batch = batch["image"]

            # Move to GPU if available
            if cuda:
                batch = batch.cuda(non_blocking=True)

            features = feat_extractor(batch)
            featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
            featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            if out is None:
                out = featuresdict
            else:
                out = {k: out[k] + featuresdict[k] for k in out.keys()}
            t.update(batch.shape[0])

    if verbose:
        print("Processing samples")

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}

    return out


# S3_ENDPOINT_URL = "https://" + os.environ["AWS_S3_ENDPOINT"]
# s3_bucket = Path("nshaud/biggan-imagenet-256-samples")


@click.command()
@click.argument("backbone", type=click.Choice(BACKBONES, case_sensitive=False))
@click.argument(
    "filename", type=click.Path(writable=True, file_okay=True, dir_okay=False)
)
@click.option(
    "--folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Path to a local folder containing images. \
          If provided, features will be extracted from images in this folder.",
)
@click.option(
    "--torchvision-dataset",
    type=str,
    required=False,
    help="Torchvision dataset name (e.g., 'CIFAR10', 'MNIST'). \
          Split can be provided as 'CIFAR10:train' or 'CIFAR10:test'. \
          Use train split by default.",
)
@click.option(
    "--torchvision-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=False,
    help="Root directory for torchvision datasets. \
          If not provided, the current directory will be used.",
)
@click.option(
    "--hf-dataset",
    type=str,
    required=False,
    help="HuggingFace dataset name (e.g., 'cassiekang/cub200_dataset'). \
          Split can be provided as 'cassiekang/cub200_dataset:train' or 'cassiekang/cub200_dataset:test'. \
          Use train split by default.",
)

@click.option(
    "--s3-bucket",
    type=str,
    required=False,
    help="S3 bucket name. If provided, features will be extracted from images in this bucket.",
)
@click.option(
    "--s3-endpoint",
    type=str,
    required=False,
    help="S3 endpoint URL. Useless if --s3-bucket is not provided.",
)
@click.option(
    "--batch-size", type=int, default=32, help="Batch size for feature extraction."
)
@click.option(
    "--verbose/--no-verbose",
    default=True,
    help="Enable verbose output and progress bar.",
)
@click.option("--cuda/--no-cuda", default=True, help="Use GPU if available")
@click.option(
    "--num-workers",
    type=int,
    default=0,
    help="Number of workers for data loading. Set to 0 for no multiprocessing (default).",
)
def extract_features(
    backbone: str,
    filename: str,
    folder: Optional[Path] = None,
    torchvision_dataset: Optional[str] = None,
    torchvision_root: Optional[Path] = None,
    hf_dataset: Optional[str] = None,
    s3_bucket: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    batch_size: int = 32,
    verbose: bool = True,
    cuda: bool = True,
    num_workers: int = 0,
):
    """
    Extract features from images in a local folder, a torchvision dataset or an S3 bucket using the specified model.

    Args:
        backbone (str): Backbone model to use for feature extraction, as provided by torch-fidelity.
        filename (str): Path to the output file where extracted features will be saved.
    """
    if folder:
        # Load images from local folder
        dataset = SimpleImageFolder(folder)
    elif torchvision_dataset:
        name, split = (
            torchvision_dataset.split(":")
            if ":" in torchvision_dataset
            else (torchvision_dataset, "train")
        )
        # Load images from torchvision dataset
        try:
            dataset = torchvision.datasets.__dict__[name](
                root=torchvision_root or ".", split=split, download=True, transform=transforms.PILToTensor()
            )
        except TypeError: # maybe this dataset uses "train : bool = True" instead of "split"
            dataset = torchvision.datasets.__dict__[name](
                root=torchvision_root or ".", train=(split == "train"), download=True, transform=transforms.PILToTensor()
            )
    elif hf_dataset:
        name, split = (
            hf_dataset.split(":")
            if ":" in hf_dataset
            else (hf_dataset, "train")
        )
        # Load images from HuggingFace dataset
        dataset = load_dataset(name).with_format("torch")[split]
    elif s3_bucket and s3_endpoint:
        # Load images from S3 bucket
        dataset = MinIODataset(s3_endpoint, s3_bucket)
    else:
        raise ValueError(
            "Either --folder, --torchvision-dataset, or both --s3-bucket and --s3-endpoint must be provided."
        )

    feat_layer_name = resolve_feature_layer_for_metric(
        "fid", feature_extractor=backbone
    )
    feat_extractor = create_feature_extractor(backbone, [feat_layer_name])

    features = extract_features_from_dataset(
        dataset,
        feat_extractor,
        batch_size=batch_size,
        verbose=verbose,
        cuda=cuda,
        num_workers=num_workers,
    )

    # Save features to disk
    torch.save(features, f"{filename}_features.pt")
    print(f"Features extracted and saved to {filename}_features.pt")

    return features


if __name__ == "__main__":
    extract_features()
