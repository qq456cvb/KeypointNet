import os
import sys
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

sys.path.append('..')
from utils.tools import *

class Trainer:
    def __init__(self, cfg, **kwargs):
        self.max_epoch = cfg.max_epoch
        self.eval_step = cfg.eval_step
        self.save_step = cfg.save_step
        self.lr_step = cfg.optimizer.lr_step
        self.lr_decay = cfg.optimizer.lr_decay
        self.cfg = cfg
        self.loss = 0
        self.loss_dict = {'total': 0.}
        self.iteration = 0
        self.epoch = 0
        self.best_corr = -1
        
        for key in kwargs:
            setattr(self, key, kwargs[key])
            
        self.model = self.trainer_config['model']
        self.criterion = self.trainer_config['criterion']
        self.scheduler = self.trainer_config['scheduler']
        self.train_data = self.trainer_config['train_data']

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
            self.scheduler.zero_grad()

            logits = self.model(batch_data)
            pcds, kp_index = batch_data
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
        val_data = self.trainer_config['val_data']
        metric = self.trainer_config['metric']
        self.logger.info("-------------------Evaluating model----------------------")
        cnt = 0
        
        results_list = []
        for batch_data in tqdm(val_data):
            with torch.no_grad():
                logits = self.model(batch_data)
            pred_index = torch.argmax(logits, dim=1)
            pcds, kp_index = batch_data
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
            log_content = 'Validation pck' + '-{:.2f}: {:.8f}'.format(i*0.01, corr)
            self.logger.info(log_content)

        is_best = False
        if results_list[0] > self.best_corr:
            self.best_corr = results_list[0]
            is_best = True
            self.logger.info("Validation best epoch: {}".format(self.epoch))

        if is_best:
            torch.save(self.model.state_dict(), 'pck_best.pth')
            
        for k, v in loss_dict.items():
            self.logger.info("Validation loss: {}: {}".format(k, v))

