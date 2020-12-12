import os

import hydra
import torch
import logging
logger = logging.getLogger(__name__)
import omegaconf
import importlib
from tqdm import tqdm
from utils import AverageMeter, ModelWrapper
import dataset
import utils


def train(cfg):
    KeypointDataset = getattr(dataset, 'Keypoint{}Dataset'.format(cfg.task.capitalize()))

    log_dir = os.path.curdir

    train_dataset = KeypointDataset(cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    val_dataset = KeypointDataset(cfg, 'val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    cfg.num_classes = train_dataset.nclasses
    model_impl = getattr(importlib.import_module('.{}'.format(cfg.network.name), package='models'), '{}Model'.format(cfg.task.capitalize()))(cfg).cuda()
    model = ModelWrapper(model_impl).cuda()
    
    logger.info('Start training on {} keypoint detection...'.format(cfg.task))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )
    criterion = getattr(utils, '{}Criterion'.format(cfg.task.capitalize()))().cuda()
    
    meter = AverageMeter()
    best_loss = 1e10
    for epoch in range(cfg.max_epoch + 1):
        train_iter = tqdm(train_dataloader)

        # Training
        meter.reset()
        model.train()
        for i, data in enumerate(train_iter):
            outputs = model(data)
            loss = criterion(data, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())
            
        logger.info(
                f'Epoch: {epoch}, Average Train loss: {meter.avg}'
            )
        # validation loss
        model.eval()
        meter.reset()
        val_iter = tqdm(val_dataloader)
        for i, data in enumerate(val_iter):
            with torch.no_grad():
                outputs = model(data)
                loss = criterion(data, outputs)

            val_iter.set_postfix(loss=loss.item())
            meter.update(loss.item())

        if meter.avg < best_loss:
            logger.info("best epoch: {}".format(epoch))
            best_loss = meter.avg
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pth'))

        logger.info(
                f'Epoch: {epoch}, Average Val loss: {meter.avg}'
            )


@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.log_path = '{}_log'.format(cfg.task)
    logger.info(cfg.pretty())
    train(cfg)


if __name__ == '__main__':
    main()