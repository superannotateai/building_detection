from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
import enum
from pathlib import Path
import numpy as np
import warnings
import rasterio
import cv2


# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger("MainLogger")
logger.setLevel(INFO)
logger.addHandler(handler) 

# Fix seed for reproducibility
np.random.seed(1145141919)


def create_dir_if_doesnt_exist(path):
    # Create working dir if not exists.
    directory= Path(path)
    if not directory.exists():
        directory.mkdir(parents=True)


def datapath_to_prefix(datapath):
    if datapath is None:
        return 'no_prefix'
    dir_name = Path(datapath).name
    if dir_name.startswith('AOI_2_Vegas'):
        return 'AOI_2_Vegas'
    elif dir_name.startswith('AOI_3_Paris'):
        return 'AOI_3_Paris'
    elif dir_name.startswith('AOI_4_Shanghai'):
        return 'AOI_4_Shanghai'
    elif dir_name.startswith('AOI_5_Khartoum'):
        return 'AOI_5_Khartoum'
    elif dir_name.startswith('ALL_IN_ONE'):
        return 'ALL_IN_ONE'
    else:
        return 'no_prefix'
    

class DataSplit(enum.Enum):
   Train = 1
   Validation = 2
   Test = 3


def image_ids_to_slice_ids(image_list, slice_count):
    """ If current classifier slices images, converts image list of slice list, otherwise returns the same list.
    """
    if slice_count == 1:
        return image_list
    slice_id_list = []
    for image_id in image_list:
        for slice_pos in range(slice_count):
            slice_id_list.append('{}_{}'.format(image_id, slice_pos))
    return slice_id_list


def load_raster_image(path):
    with rasterio.open(path, 'r') as f:
        values = f.read().astype(np.float32)
    return values


def reduce_polygon_points(cnt, max_num):
    # cnt = np.reshape(cnt, (-1, 1, 2))
    out = []
    if max_num is None or len(cnt) <= max_num:
        out = cnt.copy()
        return out
    max_distance = 1
    while True:
        out = cv2.approxPolyDP(cnt, max_distance, True)
        out = np.squeeze(out)
        # print('Out', out.dtype, out.shape)
        if len(out) <= max_num:
            return out
        max_distance += 1

