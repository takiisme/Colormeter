from enum import Enum
from tueplots.constants.color import rgb

class ColorSpace(Enum):
    RGB = 'rgb'
    LAB = 'lab'
    HSV = 'hsv'
    HEX = 'hex'

    def get_channels(self) -> list[str]:
        if self == ColorSpace.RGB:
            return ['R', 'G', 'B']
        elif self == ColorSpace.LAB:
            return ['l', 'a', 'b']
        elif self == ColorSpace.HSV:
            return ['H', 'S', 'V']
        else:
            raise ValueError(f"Unsupported color space: {self}")


class LightingCondition(Enum):
    DAYLIGHT = 'daylight'
    DARK = 'dark'


COLOR = {
    'raw': rgb.tue_gray,
    'color': rgb.tue_gray,
    'scaling': rgb.tue_red,
    'reduced': rgb.tue_blue,
    'full': rgb.tue_gold,
}
