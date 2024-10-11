import random
import string
from dataclasses import asdict, fields
from typing import List

import numpy as np
from PIL import Image


def convert_dataclass_for_serialization(data_instance) -> dict:
    """Convert dataclass instance to a dictionary, handling ndarrays specifically."""
    d = asdict(data_instance)
    for field in fields(data_instance):
        if isinstance(d[field.name], np.ndarray):
            # Convert ndarray to list
            d[field.name] = d[field.name].tolist()
    return d


def get_all_rotated_images(image: Image.Image, angle: int) -> List[Image.Image]:
    """Get all rotated image_scene of the input image by the given angle."""
    assert 360 % angle == 0, "Angle must be a divisor of 360."
    rotated_images = []
    for i in range(0, 360, angle):
        rotated_images.append(image.rotate(i))
    return rotated_images


def get_image_pyramid(image: Image.Image, scale_factor: float, min_size: int) -> List[Image.Image]:
    """Get image pyramid of the input image."""
    image_pyramid = []
    while image.size[0] >= min_size and image.size[1] >= min_size:
        image_pyramid.append(image)
        image = image.resize((int(image.size[0] * scale_factor), int(image.size[1] * scale_factor)))
    return image_pyramid


def generate_random_string(length=20):
    # Define the characters to choose from (letters and digits)
    characters = string.ascii_letters + string.digits

    # Generate a random string of the specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))

    return random_string
