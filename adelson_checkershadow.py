from pathlib import Path

import numpy as np
from PIL import Image

image_dir = Path(__file__).parents[0]

image = np.array(Image.open(image_dir / "adelson_checkershadow.bmp").convert("L"))
mask = np.array(Image.open(image_dir / "adelson_checkershadow_mask.bmp").convert("L"))

for idx, val in enumerate(np.unique(mask)):
    mask[mask == val] = idx

checkershadow = {
    "img": image,
    "target_mask": mask,
}
