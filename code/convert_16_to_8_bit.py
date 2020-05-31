# -*- coding: utf-8 -*-
"""
Tiff image conversion 16-bit to 8-bit.

Author: martun
"""
from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
import argparse
import math
import glob
import warnings

import click
import imageio
import numpy as np
import skimage.transform
import rasterio
#import shapely.wkt

# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger('spacenet2')
logger.setLevel(INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('datapath', type=str)
def to_8_bit(datapath):
    """ Converts all tiff images in a given directory to 8-bit images. """
    
    logger.info("Converting images in directory " + datapath)

    # Load all images from given directory.
    image_paths = glob.glob(datapath + "/*.tif")

    channel_max = {0:float('-inf'), 1:float('-inf'), 2:float('-inf')}
    channel_min = {0:float('inf'), 1:float('inf'), 2:float('inf')}

    with click.progressbar(image_paths, label='Computing channel min/max') as progress_bar:
        for image_path in progress_bar:
            with rasterio.open(image_path, 'r') as f:
                values = f.read().astype(np.float32)
                for chan_i in range(3):
                    # logger.info("min {} max {}".format(channel_min[chan_i], channel_max[chan_i]))
                    channel_min[chan_i] = min(channel_min[chan_i], np.min(values[chan_i]))
                    channel_max[chan_i] = max(channel_max[chan_i], np.max(values[chan_i]))

    with click.progressbar(image_paths, label='Converting images') as progress_bar:
        for image_path in progress_bar:
            with rasterio.open(image_path, 'r') as f:
                values = f.read().astype(np.float32)
                for chan_i in range(3):
                    values[chan_i] = np.clip(values[chan_i], channel_min[chan_i],
                            channel_max[chan_i])
                    values[chan_i] = (values[chan_i] -  channel_min[chan_i]) / (channel_max[chan_i] - channel_min[chan_i]) * 255
                    # logger.info("Sum is {}".format(np.mean(values[chan_i])))
                # Move channel to the end. CHW
                values = np.swapaxes(values, 0, 2)
                values = np.swapaxes(values, 0, 1)

                # Save as image with same name + _8_bit at the end.
                output_image_path = image_path[:-4] + "_8_bit.jpg"

                imageio.imwrite(output_image_path, np.uint8(values), 'RGB')


if __name__ == '__main__':
    logger.addHandler(handler)
    cli()
