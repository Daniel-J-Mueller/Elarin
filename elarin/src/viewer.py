"""Utility for visualizing what Elarin "sees" with text overlay."""

from PIL import Image, ImageDraw
from typing import Union
import numpy as np


def render(frame: Union[Image.Image, np.ndarray], text: str = "") -> Image.Image:
    """Return image with a text bar at the bottom."""
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)
    bar_height = 30
    result = Image.new("RGB", (frame.width, frame.height + bar_height))
    result.paste(frame, (0, 0))
    draw = ImageDraw.Draw(result)
    draw.rectangle([(0, frame.height), (frame.width, frame.height + bar_height)], fill=(0, 0, 0))
    draw.text((5, frame.height + 5), text, fill=(255, 255, 255))
    return result
