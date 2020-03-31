import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.tools import *

class Evaluator:
    def __init__(self, cfg):
        self.model = cfg['trainer_config']['model']
        self.criterion = cfg['trainer_config']['criterion']
        self.scheduler = cfg['trainer_config']['scheduler']
        self.train_data = cfg['trainer_config']['train_data']
        self.cat = cfg["category"]
        self.logger = cfg['logger']
        self.writer = cfg['writer']
        self.max_epoch = int(cfg["max_epoch"])
        self.eval_step = int(cfg["eval_step"])
        self.save_step = int(cfg["save_step"])
        self.lr_step = int(cfg['lr_step'])
        self.lr_decay = float(cfg["lr_decay"])
        self.task = cfg['task_type']
        self.cfg = cfg
        self.loss = 0
        self.loss_dict = {'total': 0.}
        self.metric_value = 0
        self.best_score = 0
        self.iteration = 0
        self.epoch = 0
        self.steps = 0
        self.best_val_loss = sys.maxsize
        self.img_save_dir = os.path.join(
            cfg['root_path'], cfg['log_path'], cfg['name'], 'images')

        self.ckpt_path = os.path.join(cfg["root_path"], cfg["checkpoint_path"], cfg["name"], 'model_best.pth.tar')
        # self.ckpt_path = os.path.join(cfg["root_path"], cfg["checkpoint_path"], cfg["name"], 'iou_checkpoint_100.pth.tar')
        self.fout = open(os.path.join(cfg["log_path"], "test_results_"+cfg["net"]+"_"+self.task+".txt"), "a")

        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

    def run(self):
        self.model.load_state_dict(torch.load(self.ckpt_path)['state_dict'])
        tic = time.time()
        self.evaluate()
        toc = time.time()
        self.logger.info("Evaluation time is {}s".format(toc - tic))
        self.fout.close()

    def evaluate(self):
        self.model.eval()
        test_data = self.cfg['trainer_config']['test_data']
        metric = self.cfg['trainer_config']['metric']
        self.logger.info("\n-------------------Testing model----------------------")
        res = 0
        cnt = 0
        if self.task == 'pck':
            results_list = []
            for batch_data in tqdm(test_data):
                with torch.no_grad():
                    logits = self.model(batch_data)
                pred_index = torch.argmax(logits, dim=1)
                pcds, kp_index, _ = batch_data
                input = (pcds.cuda(), kp_index.cuda(), pred_index)
                corr_list = metric(input)
                results_list.append(corr_list)

                loss_dict = self.criterion((logits, kp_index))
                for k, v in loss_dict.items():
                    if k not in self.loss_dict:
                        self.loss_dict[k] = v
                    else:
                        self.loss_dict[k] += v
                cnt += 1

            loss_avg = loss_dict['total'] / cnt
            self.writer.add_scalar('test-loss', loss_avg, self.iteration)

            self.fout.write("{}".format(self.cat))
            results_list = np.mean(results_list, axis=0)
            for i, corr in enumerate(results_list):
                self.writer.add_scalar('correspondence/{:.2f}'.format(i*0.01), corr, self.iteration)
                log_content = metric.__name__ + '_{:.2f}: {:.8f}'.format(i*0.01, corr)
                self.fout.write("\t{:.8f}".format(corr))
                self.logger.info(log_content)
            self.fout.write("\n")
            for k, v in loss_dict.items():
                self.logger.info("Test loss: {}: {}".format(k, v))


        elif self.task == 'iou':
            iou = 0.
            map = 0.
            for batch_data in tqdm(test_data):
                with torch.no_grad():
                    logits = self.model(batch_data)
                pcds, kp_index, geo_dists = batch_data
                input = (kp_index, logits, geo_dists)
                iou_, map_ = metric(input)
                iou += iou_
                map += map_
                cnt += 1
            iou /= cnt
            map /= cnt
            self.logger.info("Validation IoU: {}".format(iou))
            self.logger.info("Validation mAP: {}".format(map))
            self.fout.write("{}\t{}\t{}\n".format(self.cat, iou, map))

        else:
            raise NotImplementedError

if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler("test.log"))
    logger.info("test")
