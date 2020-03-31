from model.benchmark import BenchMarkMetric as Metric
from model.benchmark import BenchMarkLoss as Loss
from model.benchmark import BenchMark as Model
from dataset.data_loader import KPDataset as DataSet
from utils.tools import *
from utils.trainer import Trainer
from utils.evaluator import Evaluator
from tensorboardX import SummaryWriter
import numpy as np
import argparse

import torch.utils.data as DT
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(0)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='pointconv')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=str, default='8')
    parser.add_argument('--num_workers', type=str, default='0')
    parser.add_argument('--lr', type=str, default='0.01')
    parser.add_argument('--lr_decay', type=str, default='0.5')
    parser.add_argument('--lr_step', type=str, default='5') 
    parser.add_argument('--prob_threshold', type=float, default=0.1)
    parser.add_argument('--geo_threshold', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=str, default='-1')
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_flag', type=int, default=1)
    parser.add_argument('--mode', type=str, default='run', choices=['run', 'debug'])
    parser.add_argument('--data_type', type=str, default='geo', choices=['geo', 'gt', 'nms'])
    parser.add_argument('--task_type', type=str, default='pck', choices=['pck'])
    parser.add_argument('--category', type=str, default='airplane')

    args = parser.parse_args()
    cfg = get_cfg(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg["gpu"]

    if cfg["mode"] == "debug":
        cfg["name"] = cfg["mode"]
    elif cfg["mode"] == "run":
        cfg["name"] = cfg["net"] + "_" + cfg["task_type"] + "_" + cfg["data_type"] + "_" + cfg["category"]
    return cfg

def my_collect_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def train(config):
    trainer = Trainer(config)
    trainer.run()

def test(config):
    evaluator = Evaluator(config)
    evaluator.run()

if __name__ == '__main__':
    cfg = arg_parse()

    if cfg["mode"] == "debug" and cfg["train_flag"]:
        clean_logs_and_checkpoints(cfg)

    logger = get_logger(cfg)
    log_content = "\nUsing Configuration:\n{\n"
    for key in cfg:
        log_content += "    {}: {}\n".format(key, cfg[key])
    logger.info(log_content + '}')

    cfg['logger'] = logger
    cfg['writer'] = SummaryWriter(os.path.join(cfg['log_path'], cfg['name']))

    train_dataset = DataSet(cfg, 'train')
    validate_dataset = DataSet(cfg, 'validate')
    test_dataset = DataSet(cfg, 'test')
    cfg["num_kps"] = train_dataset.keypoints.shape[1]

    model = Model(cfg).cuda()
    criterion = Loss(cfg)
    metric = Metric(cfg)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    train_data = DT.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        drop_last=True,
        batch_size=int(cfg['batch_size']),
        num_workers=int(cfg['num_workers']),
        collate_fn=my_collect_fn
    )

    val_data = DT.DataLoader(
        dataset=validate_dataset,
        batch_size=16,
        num_workers=int(cfg['num_workers']),
        collate_fn=my_collect_fn
    )

    test_data = DT.DataLoader(
        dataset=test_dataset,
        batch_size=32,
        num_workers=int(cfg['num_workers']),
        collate_fn=my_collect_fn
    )

    cfg['trainer_config'] = dict(
        model=model,
        criterion=criterion,
        metric=metric,
        scheduler=optimizer,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
    )

    if cfg["train_flag"] == 1:
        train(cfg)
    else:
        test(cfg)
