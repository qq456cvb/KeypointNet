from model.benchmark import BenchMarkMetric as Metric
from model.benchmark import BenchMarkLoss as Loss
from model.benchmark import BenchMark as Model
from dataset.data_loader import KeypointDataset
from utils.tools import *
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from tensorboardX import SummaryWriter
import numpy as np
import logging
import os
import hydra
logger = logging.getLogger(__name__)

import torch.utils.data as DT
import torch


def train(config, **kwargs):
    trainer = Trainer(config, **kwargs)
    trainer.run()

def test(config, **kwargs):
    evaluator = Evaluator(config, **kwargs)
    evaluator.run()


@hydra.main(config_path='config/config.yaml', strict=False)
def main(cfg):
    log_content = "\nUsing Configuration:\n{\n"
    for key in cfg:
        log_content += "    {}: {}\n".format(key, cfg[key])
    logger.info(log_content + '}')
    
    writer = SummaryWriter(os.path.curdir)

    train_dataset = KeypointDataset(cfg, cfg.data.train_txt)
    validate_dataset = KeypointDataset(cfg, cfg.data.val_txt)
    test_dataset = KeypointDataset(cfg, cfg.data.test_txt)
    cfg.num_kps = train_dataset.nclasses
    
    logger.info('maximum number of keypoints: {}'.format(cfg.num_kps))

    model = Model(cfg).cuda()
    criterion = Loss(cfg)
    metric = Metric(cfg)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    train_data = DT.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    val_data = DT.DataLoader(
        dataset=validate_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    test_data = DT.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    trainer_config = dict(
        model=model,
        criterion=criterion,
        metric=metric,
        scheduler=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    if cfg.task == 'train':
        train(cfg, logger=logger, writer=writer, trainer_config=trainer_config)
    else:
        test(cfg, logger=logger, writer=writer, trainer_config=trainer_config)


if __name__ == "__main__":
    main()