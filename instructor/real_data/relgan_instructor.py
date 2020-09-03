# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : relgan_instructor.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from time import time
import config as cfg
from instructor.real_data.instructor import BasicInstructor
from models.RelGAN_D import RelGAN_D, GradNorm
from models.RelGAN_G import RelGAN_G
from utils.helpers import get_fixed_temperature, get_losses
from tensorboardX import SummaryWriter


class RelGANInstructor(BasicInstructor):
    def __init__(self, opt):
        super(RelGANInstructor, self).__init__(opt)
        norm = opt.norm
        assert norm in ['none', 'spectral', 'gradnorm']
        # generator, discriminator
        print('norm ', norm)
        self.gen = RelGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx,gpu=cfg.CUDA)
        self.dis = RelGAN_D(cfg.dis_embed_dim, cfg.max_seq_len, cfg.num_rep, cfg.vocab_size, cfg.padding_idx,
                            gpu=cfg.CUDA,norm=norm ).cuda()
        if norm == 'gradnorm':
            print('use gradnorm')
            self.dis = GradNorm(self.dis).cuda()

        self.init_model()

        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.dis_opt = optim.Adam(self.dis.parameters(), lr=cfg.dis_lr)

        os.makedirs(cfg.log_filename.replace('.txt', ''), exist_ok=True)

        self.logger = SummaryWriter(
            cfg.log_filename.replace('.txt', '')+'_'+norm
        )

    def _run(self):
        # ===PRE-TRAINING (GENERATOR)===
        if os.path.exists(cfg.pretrained_gen_path):
            checkpoint = torch.load(cfg.pretrained_gen_path)
            generation_weights = self.gen.state_dict()
            match = True
            for key, value in checkpoint.items():
                if key not in generation_weights:
                    match = False
                elif generation_weights[key].shape != checkpoint[key].shape:
                    match = False
            if match:
                self.gen.load_state_dict(checkpoint)
                print('Load pre-trained generator: {}'.format(cfg.pretrained_gen_path))
        elif not cfg.gen_pretrain:
            self.log.info('Starting Generator MLE Training...')
            self.pretrain_generator(cfg.MLE_train_epoch)
            if cfg.if_save and not cfg.if_test:
                torch.save(self.gen.state_dict(), cfg.pretrained_gen_path)
                print('Save pre-trained generator: {}'.format(cfg.pretrained_gen_path))

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')

        progress = tqdm(range(cfg.ADV_train_epoch), dynamic_ncols=True)

        for adv_epoch in progress:
            self.sig.update()
            if self.sig.adv_sig:
                start = time()
                g_loss = self.adv_train_generator(cfg.ADV_g_step)  # Generator
                d_loss = self.adv_train_discriminator(cfg.ADV_d_step)  # Discriminator
                self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature

                progress.set_description(
                    'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.temperature))
                if adv_epoch % 10 == 0:
                    self.logger.add_scalar('train/d_loss', float(d_loss), adv_epoch)
                    self.logger.add_scalar('train/g_loss', float(g_loss), adv_epoch)
                    self.logger.add_scalar('train/temperature', self.gen.temperature, adv_epoch)

                # TEST
                if adv_epoch % cfg.adv_log_step == 0 and adv_epoch > 0:
                    metrics = self.cal_metrics(fmt_str=False)
                    for key, value in metrics.items():
                        if isinstance(value, list):
                            for idx, v in enumerate(value):
                               self.logger.add_scalar('train/'+key+'/'+str(idx), v, adv_epoch)
                        else:
                            self.logger.add_scalar('train/'+key, value, adv_epoch)

                    self.logger.flush()
                    self.log.info('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f' % (
                        adv_epoch, g_loss, d_loss ))

                    if cfg.if_save and not cfg.if_test:
                        self._save('ADV', adv_epoch)
            else:
                self.log.info('>>> Stop by adv_signal! Finishing adversarial training...')
                progress.close()
                break

    def _test(self):
        print('>>> Begin test...')

        self._run()
        pass

    def pretrain_generator(self, epochs):
        """
        Max Likelihood Pre-training for the generator
        """
        for epoch in range(epochs):
            self.sig.update()
            if self.sig.pre_sig:
                # ===Train===
                pre_loss = self.train_gen_epoch(self.gen, self.train_data.loader, self.mle_criterion, self.gen_opt)

                # ===Test===
                if (epoch % cfg.pre_log_step == 0 or epoch == epochs - 1) and epoch > 0:
                    metrics = self.cal_metrics(fmt_str=False)
                    for key, value in metrics.items():
                        if isinstance(value, list):
                            for idx, v in enumerate(value):
                               self.logger.add_scalar('pretrain/'+key+'/'+str(idx), v, epoch)
                        else:
                            self.logger.add_scalar('pretrain/'+key, value, epoch)
                    self.logger.add_scalar('pretrain/loss', pre_loss, epoch)
                    self.logger.flush()
                    self.log.info('[MLE-GEN] epoch %d : pre_loss = %.4f' % (
                        epoch, pre_loss ))

                    if cfg.if_save and not cfg.if_test:
                        self._save('MLE', epoch)
            else:
                self.log.info('>>> Stop by pre signal, skip to adversarial training...')
                break

    def adv_train_generator(self, g_step):
        total_loss = 0
        for step in range(g_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.gen_adv_opt, g_loss, self.gen)
            total_loss += g_loss.item()

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step):
        total_loss = 0
        for step in range(d_step):
            real_samples = self.train_data.random_batch()['target']
            gen_samples = self.gen.sample(cfg.batch_size, cfg.batch_size, one_hot=True)
            if cfg.CUDA:
                real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()
            real_samples = F.one_hot(real_samples, cfg.vocab_size).float()

            # ===Train===
            d_out_real = self.dis(real_samples)
            d_out_fake = self.dis(gen_samples)
            _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

            self.optimize(self.dis_opt, d_loss, self.dis)
            total_loss += d_loss.item()

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
