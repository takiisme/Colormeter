from enum import Enum

class ColorSpace(Enum):
    RGB = 'rgb'
    LAB = 'lab'
    HSV = 'hsv'

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
