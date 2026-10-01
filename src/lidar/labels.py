"""Label spaces.

Three levels:
  raw       SemanticKITTI ids stored in .label files (lower 16 bits), e.g. 40 road, 252 moving-car
  train     19 classes the network predicts (standard SemanticKITTI benchmark set), IGNORE = 255
  category  7 map categories the 2.5D grid stores, ordered so a larger value = higher safety priority
"""
import numpy as np

IGNORE = 255

TRAIN_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist",
    "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]
NUM_CLASSES = len(TRAIN_NAMES)

# raw id -> train id (anything not listed is ignored)
RAW_TO_TRAIN = {
    10: 0, 252: 0,                       # car, moving-car
    11: 1,                               # bicycle
    15: 2,                               # motorcycle
    18: 3, 258: 3,                       # truck
    13: 4, 16: 4, 20: 4, 256: 4, 257: 4, 259: 4,  # bus / on-rails / other-vehicle
    30: 5, 254: 5,                       # person
    31: 6, 253: 6,                       # bicyclist
    32: 7, 255: 7,                       # motorcyclist
    40: 8, 60: 8,                        # road, lane-marking
    44: 9,                               # parking
    48: 10,                              # sidewalk
    49: 11,                              # other-ground
    50: 12,                              # building
    51: 13,                              # fence
    70: 14,                              # vegetation
    71: 15,                              # trunk
    72: 16,                              # terrain
    80: 17,                              # pole
    81: 18,                              # traffic-sign
}
MOVING_RAW = (252, 253, 254, 255, 256, 257, 258, 259)

# Map categories, in increasing safety priority.
UNKNOWN, DRIVABLE, TERRAIN, VEGETATION, STATIC, VEHICLE, PEDESTRIAN = range(7)
CATEGORY_NAMES = ["unknown", "drivable", "terrain", "vegetation", "static-obstacle", "vehicle", "pedestrian"]
NUM_CATEGORIES = len(CATEGORY_NAMES)
GROUND_CATEGORIES = (DRIVABLE, TERRAIN)
DYNAMIC_CATEGORIES = (VEHICLE, PEDESTRIAN)

TRAIN_TO_CATEGORY = np.array([
    VEHICLE, VEHICLE, VEHICLE, VEHICLE, VEHICLE,   # car .. other-vehicle
    PEDESTRIAN, PEDESTRIAN, PEDESTRIAN,            # person, bicyclist, motorcyclist
    DRIVABLE, DRIVABLE,                            # road, parking
    TERRAIN, TERRAIN,                              # sidewalk, other-ground
    STATIC, STATIC,                                # building, fence
    VEGETATION,                                    # vegetation
    STATIC,                                        # trunk
    TERRAIN,                                       # terrain (grass / soil)
    STATIC, STATIC,                                # pole, traffic-sign
], dtype=np.uint8)

# RGB per category for visualization
CATEGORY_COLORS = np.array([
    [70, 76, 90],     # unknown
    [138, 147, 166],  # drivable
    [176, 141, 87],   # terrain
    [63, 174, 90],    # vegetation
    [79, 124, 255],   # static obstacle
    [255, 159, 28],   # vehicle
    [255, 59, 107],   # pedestrian
], dtype=np.uint8)

_raw_lut = np.full(1 << 16, IGNORE, dtype=np.uint8)
for raw, train in RAW_TO_TRAIN.items():
    _raw_lut[raw] = train
_moving_lut = np.zeros(1 << 16, dtype=bool)
_moving_lut[list(MOVING_RAW)] = True

_category_lut = np.full(256, UNKNOWN, dtype=np.uint8)
_category_lut[:NUM_CLASSES] = TRAIN_TO_CATEGORY


def raw_to_train(raw: np.ndarray) -> np.ndarray:
    """uint32 .label contents -> uint8 train ids."""
    return _raw_lut[raw & 0xFFFF]


def raw_is_moving(raw: np.ndarray) -> np.ndarray:
    return _moving_lut[raw & 0xFFFF]


def train_to_category(train: np.ndarray) -> np.ndarray:
    return _category_lut[train]
