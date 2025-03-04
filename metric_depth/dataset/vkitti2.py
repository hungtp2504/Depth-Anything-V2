import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Resize, NormalizeImage, PrepareForNet, Crop


import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .transform import Resize, NormalizeImage, PrepareForNet, Crop


class VKITTI2(Dataset):
    def __init__(self, root_dir, filelist_path, mode, size=(518, 518)):
        """
        VKITTI2 Dataset Loader

        Args:
            root_dir (str): Base directory for dataset (e.g., 'datasets/vkitti/vkitti_depth').
            filelist_path (str): Path to the file containing image-depth file pairs.
            mode (str): 'train' or 'val', affects data augmentation.
            size (tuple): Target image size (width, height).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.size = size

        net_w, net_h = size
        self.transform = Compose(
            [
                Resize(
                    width=net_w,
                    height=net_h,
                    resize_target=True,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
            + ([Crop(size[0])] if self.mode == "train" else [])
        )

        # 🔹 Read and filter file list
        with open(filelist_path, "r") as f:
            lines = f.read().splitlines()

        # 🔹 Only keep paths containing "Scene18/morning"
        self.filelist = [
            (os.path.join(self.root_dir, line.split(" ")[0]),  # Image Path
             os.path.join(self.root_dir, line.split(" ")[1]))  # Depth Path
            for line in lines
        ]

        if len(self.filelist) == 0:
            raise ValueError(f"No matching data found in {filelist_path}'")

    def __getitem__(self, index):
        img_path, depth_path = self.filelist[index]

        # 🔹 Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        # 🔹 Load and preprocess depth map (Convert cm to meters)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0  

        # 🔹 Apply transformations
        sample = self.transform({"image": image, "depth": depth})

        # 🔹 Convert to tensors
        sample["image"] = torch.from_numpy(sample["image"])
        sample["depth"] = torch.from_numpy(sample["depth"])
        sample["valid_mask"] = sample["depth"] <= 80
        sample["image_path"] = img_path  # Keep track of image paths

        return sample

    def __len__(self):
        return len(self.filelist)
