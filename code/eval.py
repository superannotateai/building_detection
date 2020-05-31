from tqdm import tqdm
import numpy as np
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from jaccard_loss import jaccard_coeff

# Logger
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))

logger = getLogger("MainLogger")
logger.setLevel(INFO)

# Fix seed for reproducibility
np.random.seed(1145141919)


if __name__ == '__main__':
    logger.addHandler(handler) 


def eval_net(net, loader, device,
        criterion=nn.BCEWithLogitsLoss().cuda(), world_size=1, rank=0):
    """Evaluation without the densecrf with the jaccard coefficient"""
    net.eval()
    mask_type = torch.float32
    n_val = len(loader.dataset) / world_size  # the number of images for this worker
    tot = 0
    loss = 0
    pbar = tqdm(total=n_val, desc='Validation round', unit='images', leave=False) if rank==0 else None
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        loss += criterion(mask_pred, true_masks)
        mask_pred = torch.sigmoid(mask_pred)
        pred = (mask_pred > 0.5).float()
        tot += jaccard_coeff(pred, true_masks).item() * pred.shape[0]
        if pbar is not None:
            pbar.update(imgs.shape[0])

    if pbar is not None:
        pbar.close()
        del pbar
    return tot / n_val, loss


def predict_generator(net, validation_dataset, batch_size, device, world_size=1, rank=0):
    """Run predictions and yield."""
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
    	validation_dataset,
    	num_replicas=world_size,
    	rank=rank
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        pin_memory=True, drop_last=False, sampler=validation_sampler)

    net.eval()
    mask_type = torch.float32
    n_val = len(validation_dataset)  # the number of batch
    tot = 0

    pbar = tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) if rank==0 else None
    for batch in validation_loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)
            mask_pred = torch.sigmoid(mask_pred)

        # Move the result back to CPU.
        mask_pred = mask_pred.cpu().numpy()
        # Martun: No need for the following line.
        #mask_pred = (mask_pred > 0.5).astype(np.uint8)
        yield mask_pred
        if pbar is not None:
            pbar.update(imgs.shape[0] * world_size)

    if pbar is not None:
        pbar.close()
        del pbar


def predict_generator_test(net, dataset, batch_size, device, maximal_batch_size=32):
    """Run predictions and yield. Because we want to get results for full images, batch_size can be large for large images. If batch_size > 64, divide it into sub-batches of 64 and run on them, then merge. """
    subbatch_size = batch_size if batch_size < maximal_batch_size else maximal_batch_size
    logger.info("subbatch_size is {}".format(str(subbatch_size)))
    loader = DataLoader(
        dataset, batch_size=subbatch_size,
        shuffle=False, num_workers=0,
        pin_memory=True, drop_last=False)

    net.eval()
    mask_type = torch.float32
    n_val = len(dataset)  # the number of images.
    logger.info("dataset size is {}".format(str(n_val)))
    tot = 0

    with tqdm(total=n_val, desc='Running on test images', unit='images', leave=False) as pbar:
        mask_pred_accumulated = None
        for batch in loader:
            imgs = batch['image']
            imgs = imgs.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred = torch.sigmoid(mask_pred)

            # Move the result back to CPU.
            mask_pred = mask_pred.cpu().numpy()
            # Martun: No need for the following line.
            #mask_pred = (mask_pred > 0.5).astype(np.uint8)
            if mask_pred_accumulated is None:
                mask_pred_accumulated = mask_pred
            else:
                mask_pred_accumulated = np.concatenate(
                    (mask_pred_accumulated, mask_pred), axis=0)
            pbar.update(imgs.shape[0])
            # If one full batch is done, yield it.
            if mask_pred_accumulated.shape[0] >= batch_size:
                masks = mask_pred_accumulated[:batch_size]
                mask_pred_accumulated = mask_pred_accumulated[batch_size:]
                yield masks
        if mask_pred_accumulated is not None and mask_pred_accumulated.shape[0] != 0:
            yield mask_pred_accumulated
