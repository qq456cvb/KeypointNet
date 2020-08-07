from model.benchmark import BenchMarkMetric as Metric
from model.benchmark import BenchMarkLoss as Loss
from model.benchmark import BenchMark as Model
from dataset.data_loader import KeypointDataset, my_collate
from utils.tools import *
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from tensorboardX import SummaryWriter
import numpy as np
import logging
import os
import pickle
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
    
BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')

# airplane,bathtub,bed,bottle,cap,car,chair,guitar,helmet,knife,laptop,motorcycle,mug,skateboard,table,vessel
# pointnet,pointnet2,rscnn,dgcnn,rsnet,spidercnn,pointconv,graphcnn
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
    
    # load geodists, without model normalization
    if os.path.exists(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cfg.class_name))):
        logger.info('Found geodesic cache...')
        geo_dists = pickle.load(open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cfg.class_name)), 'rb'))
    else:
        geo_dists = {}
        logger.info('Generating geodesics, this may take some time...')
        for i in tqdm(range(len(test_dataset.mesh_names))):
            if test_dataset.mesh_names[i] not in geo_dists:
                geo_dists[test_dataset.mesh_names[i]] = gen_geo_dists(test_dataset.pcds[i]).astype(np.float32)
        pickle.dump(geo_dists, open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(cfg.class_name)), 'wb'))

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
        collate_fn=my_collate
    )

    val_data = DT.DataLoader(
        dataset=validate_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=my_collate
    )

    test_data = DT.DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=my_collate
    )

    trainer_config = dict(
        model=model,
        criterion=criterion,
        metric=metric,
        scheduler=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        geo_dists=geo_dists
    )

    if cfg.task == 'train':
        train(cfg, logger=logger, writer=writer, trainer_config=trainer_config)
    else:
        test(cfg, logger=logger, writer=writer, trainer_config=trainer_config)


if __name__ == "__main__":
    main()