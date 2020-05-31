from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from tqdm import tqdm
import string
import random

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributed import all_reduce
from torch.distributed import ReduceOp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import imgaug as ia
import imgaug.augmenters as iaa
# import imageio

import utils
import dice_loss
from eval import eval_net

logger = getLogger("MainLogger")


def augment_batch(images, masks):
    #initial_image = images

    images = (images.numpy() * 255).astype(np.uint8)
    images = np.swapaxes(images, 1, 3)
    images = np.swapaxes(images, 1, 2)
    masks = masks.numpy().astype(np.int32)
    masks = np.swapaxes(masks, 1, 3)
    masks = np.swapaxes(masks, 1, 2)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.5), # vertically flip 50% of all images
            # change brightness of images (by -10 to 40 of original value)
            iaa.Add((-20, 60)),
            iaa.AddToSaturation((-20, 20)), # change saturation
            iaa.Multiply((0.8, 1.6)),
            # improve or worsen the contrast
            iaa.LinearContrast((0.8, 1.2))
        ],
    )
    images, masks = seq(images=images, segmentation_maps=masks)
    images = np.swapaxes(images, 1, 3)
    images = np.swapaxes(images, 2, 3)
    images = images.astype(np.float32) / 255.0
    masks = np.swapaxes(masks, 1, 3)
    masks = np.swapaxes(masks, 2, 3)
    masks = masks.astype(np.uint8)
    images, masks = torch.tensor(images), torch.tensor(masks)

    # Uncomment next lines if you want to see a few examples of augmented images.
    #output_folder_path = "augmented_images"
    #utils.create_dir_if_doesnt_exist(output_folder_path)
    #random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(10))
    #for i in range(images.shape[0]):
    #    image = images[i]
    #    image = np.swapaxes(image, 0, 2)
    #    image = np.swapaxes(image, 0, 1)
    #    init_image = initial_image[i]
    #    init_image = np.swapaxes(init_image, 0, 2)
    #    init_image = np.swapaxes(init_image, 0, 1)

    #    imageio.imwrite("{}/{}_{}.jpg".format(output_folder_path, random_str, i), np.uint8(image * 255), 'RGB')
    #    imageio.imwrite("{}/{}_{}_init.jpg".format(output_folder_path, random_str, i), np.uint8(init_image * 255), 'RGB')

    return images, masks


def train_net(net,
              training_dataset,
              validation_dataset,
              optimizer,
              device=None,
              epochs=5,
              batch_size=64,
              save_after_each_epoch=True, 
              model_file_path_format="",
              world_size=1,
              rank=0):
    if device==None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_val = len(validation_dataset)
    n_train = len(training_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	training_dataset,
    	num_replicas=world_size,
    	rank=rank
    )
    train_loader = DataLoader(
        training_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        pin_memory=True, sampler=train_sampler)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(
    	validation_dataset,
    	num_replicas=world_size,
    	rank=rank
    )
    val_loader = DataLoader(
        validation_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0,
        pin_memory=True, drop_last=True, sampler=validation_sampler)

    if rank == 0:
        writer = SummaryWriter(comment=f'BS_{batch_size}')

        logger.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_after_each_epoch}
            Device:          {device.type}
        ''')
    else:
        writer = None

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', factor=0.3, patience=2)
    criterion = nn.BCEWithLogitsLoss().cuda()
    # criterion = dice_loss.BinaryDiceLoss()
    global_step = 0
    for epoch in range(epochs):
        net.train()
        pbar = tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') if rank==0 else None
        for batch_id, batch in enumerate(train_loader):
            global_step += 1
            imgs, true_masks = batch['image'], batch['mask']
            imgs, true_masks = augment_batch(imgs, true_masks)

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            # if rank == 0:
                # writer.add_images('masks/true', true_masks.detach().clone(), epoch)
                # masks_pred_cpu = masks_pred.detach().cpu().numpy()
                # writer.add_image('masks/pred', pred[0], epoch)
 
            masks_pred = net(imgs)
            loss = criterion(masks_pred, true_masks)
            if rank == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)

            if pbar is not None:
                pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            # Next line helps to overcome exploding gradient problem.
            # nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            if pbar is not None:
                pbar.update(imgs.shape[0] * world_size)
        if pbar is not None:
            pbar.close()
            del pbar
        
        # Compute validation IoU and loss.
        val_jaccard_coef, val_loss = eval_net(
            net, val_loader, device,
            criterion=criterion, world_size=world_size, rank=rank)

        val_jaccard_coef_tensor = torch.Tensor([val_jaccard_coef]).cuda()
        val_loss_tensor = torch.Tensor([val_loss]).cuda()

        # Compute training set IoU and loss.
        train_jaccard_coef, train_loss = eval_net(
            net, train_loader, device,
            criterion=criterion, world_size=world_size, rank=rank)

        train_jaccard_coef_tensor = torch.Tensor([train_jaccard_coef]).cuda()
        train_loss_tensor = torch.Tensor([train_loss]).cuda()

        train_jaccard_coef_mine = train_jaccard_coef
        train_loss_mine = train_loss
        val_jaccard_coef_mine = val_jaccard_coef
        val_loss_mine = val_loss

        # only process with rank 0 will receive the values.
        torch.distributed.all_reduce(train_jaccard_coef_tensor, op=ReduceOp.SUM)
        torch.distributed.all_reduce(train_loss_tensor, op=ReduceOp.SUM)
        torch.distributed.all_reduce(val_jaccard_coef_tensor, op=ReduceOp.SUM)
        torch.distributed.all_reduce(val_loss_tensor, op=ReduceOp.SUM)

        # Check if model parameters are in sync.
        # for p in net.parameters():
        #     gathered_data = [torch.ones_like(p.data)] * world_size
        #     torch.distributed.all_gather(gathered_data, p.data)
        #     for data in gathered_data:
        #         if not torch.all(torch.eq(p.data, data)):
        #             logger.info("Parameters unequal!!!")
        val_jaccard_coef = val_jaccard_coef_tensor.item() / world_size
        val_loss = val_loss_tensor.item()
        train_jaccard_coef = train_jaccard_coef_tensor.item() / world_size
        train_loss = train_loss_tensor.item()

        scheduler.step(val_jaccard_coef)

        if rank == 0:
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)

            logger.info('Validation Jaccard Coeff: {}'.format(val_jaccard_coef))
            logger.info('Training Jaccard Coeff: {}'.format(train_jaccard_coef))

            writer.add_scalars('Jaccard/IoU',
                    {'train': train_jaccard_coef, 'validation': val_jaccard_coef, 
                     'train_gpu_0': train_jaccard_coef_mine, 'validation_gpu_0': val_jaccard_coef_mine}, epoch)
            writer.add_scalars('epoch_losses/train_test',
                {'train': train_loss / n_train, 'validation': val_loss / n_val, 'train_gpu_0': train_loss_mine / n_train * world_size, 'validation_gpu_0': val_loss_mine / n_val * world_size}, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # Save the weights only from the first process
            if save_after_each_epoch:
                torch.save(net.state_dict(), model_file_path_format.format(epoch=epoch))
                logger.info(f'Checkpoint {epoch + 1} saved !')

    if rank == 0:
        writer.close()

