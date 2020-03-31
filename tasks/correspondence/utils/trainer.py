import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.tools import *

class Trainer:
    def __init__(self, cfg):
        self.model = cfg['trainer_config']['model']
        self.criterion = cfg['trainer_config']['criterion']
        self.scheduler = cfg['trainer_config']['scheduler']
        self.train_data = cfg['trainer_config']['train_data']
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
        self.best_corr = -1
        self.best_iou = -1
        self.best_map = -1
        self.img_save_dir = os.path.join(
            cfg['root_path'], cfg['log_path'], cfg['name'], 'images')

        if not os.path.exists(self.img_save_dir):
            os.mkdir(self.img_save_dir)

    def run(self):
        self.logger.info('-' * 20 + "New Training Starts" + '-' * 20)
        for epoch in range(self.max_epoch):
            if epoch == 0:
                self.evaluate()

            tic = time.time()
            self.epoch += 1
            self.logger.info("Epoch {}/{}:".format(epoch, self.max_epoch))
            self.step()
            if self.epoch % self.eval_step == 0:
                self.evaluate()

            for k, v in self.loss_dict.items():
                self.logger.info("training loss: {}: {}".format(k, v))
            toc = time.time()
            self.logger.info("Elapsed time is {}s".format(toc - tic))

    def step(self):
        self.loss = 0
        for k in self.loss_dict.keys():
            self.loss_dict[k] = 0.

        self.iteration += 1
        for idx, group in enumerate(self.scheduler.param_groups):
            if self.iteration % self.lr_step == 0:
                self.scheduler.param_groups[idx]['lr'] *= self.lr_decay
            self.logger.info('Learning rate is {}.'.format(group['lr']))

        cnt = 0
        self.model.train()
        for batch_data in tqdm(self.train_data):
            self.steps += 1
            self.scheduler.zero_grad()

            logits = self.model(batch_data)
            pcds, kp_index, _ = batch_data
            loss_dict = self.criterion((logits, kp_index))
            loss = loss_dict['total']

            for k, v in loss_dict.items():
                if k not in self.loss_dict:
                    self.loss_dict[k] = v
                else:
                    self.loss_dict[k] += v

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.scheduler.step()
            self.loss += loss.data
            cnt += 1

        self.loss = self.loss / cnt
        for k, v in self.loss_dict.items():
            self.loss_dict[k] = self.loss_dict[k] / cnt

        self.writer.add_scalar('training-loss', self.loss, self.iteration)


    def evaluate(self):
        self.model.eval()
        val_data = self.cfg['trainer_config']['val_data']
        metric = self.cfg['trainer_config']['metric']
        print("-------------------Evaluating model----------------------")
        res = 0
        cnt = 0
        if self.task == 'pck':
            results_list = []
            for batch_data in tqdm(val_data):
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
            self.writer.add_scalar('val-loss', loss_avg, self.iteration)

            results_list = np.mean(results_list, axis=0)
            for i, corr in enumerate(results_list):
                self.writer.add_scalar('correspondence/{:.2f}'.format(i*0.01), corr, self.iteration)
                log_content = metric.__name__ + '_{:.2f}: {:.8f}'.format(i*0.01, corr)
                self.logger.info(log_content)

            is_best = False
            # if loss_avg < self.best_val_loss:
            #     self.best_val_loss = loss_avg
            #     is_best = True
            #     self.logger.info("Validation best epoch: {}".format(self.epoch))

            if results_list[0] > self.best_corr:
                self.best_corr = results_list[0]
                is_best = True
                self.logger.info("Validation best epoch: {}".format(self.epoch))

            # if self.epoch % self.save_step == 0 or is_best:
            if is_best:
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'optimizer': self.scheduler.state_dict(), },
                    is_best,
                    os.path.join(self.cfg["root_path"], self.cfg["checkpoint_path"], self.cfg["name"],
                                 'checkpoint_{}.pth.tar'.format(self.epoch)))
            for k, v in loss_dict.items():
                self.logger.info("Validation loss: {}: {}".format(k, v))
            res = -1

        elif self.task == 'iou':
            iou = 0.
            map = 0.
            for batch_data in tqdm(val_data):
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

            is_best = False
            if iou > self.best_iou:
                is_best = True
                self.best_iou = iou

            if self.epoch % self.save_step == 0 or is_best:
                self.logger.info("Validation best IoU epoch: {}".format(self.epoch))
                save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_val_iou': self.best_iou,
                    'optimizer': self.scheduler.state_dict(), },
                    is_best,
                    os.path.join(self.cfg["root_path"], self.cfg["checkpoint_path"], self.cfg["name"],
                                 'iou_checkpoint_{}.pth.tar'.format(self.epoch)))
            # if map > self.best_map:
            #     self.best_map = map
            #     self.logger.info("Validation best mAP epoch: {}".format(self.epoch))
            #     save_checkpoint({
            #         'epoch': self.epoch,
            #         'state_dict': self.model.state_dict(),
            #         'best_val_map': self.best_map,
            #         'optimizer': self.scheduler.state_dict(), },
            #         True,
            #         os.path.join(self.cfg["root_path"], self.cfg["checkpoint_path"], self.cfg["name"],
            #                      'map_checkpoint_{}.pth.tar'.format(self.epoch)))
        else:
            raise NotImplementedError

        # self.model.train()
        # log_info = dict(
        #     metric_name=metric.__name__,
        #     value=res
        # )
        # return log_info


if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.FileHandler("test.log"))
    logger.info("test")
