# -*- coding:utf-8 -*-
# author: Awet H. Gebrehiwot
# at 8/10/22
# --------------------------|
import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.pc_dataset import get_label_name, update_config
from utils.load_save_util import load_checkpoint
from utils.metric_util import per_class_iu, fast_hist_crop
import pickle
import torch.optim as optim

import copy


def yield_target_dataset_loader(n_epochs, target_train_dataset_loader):
    for e in range(n_epochs):
        for i_iter_train, (_, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) \
                in enumerate(target_train_dataset_loader):
            yield train_vox_label, train_grid, train_pt_fea, ref_st_idx, ref_end_idx, lcw


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 unique_label,
                 unique_label_str,
                 lovasz_softmax,
                 loss_func,
                 ignore_label,
                 save_progress_iters,
                 checkpoint_save_path,
                 progres_save_path,
                 scheduler,
                 scheduler_save_path,
                 optimizer_save_path,
                 model_save_dir,
                 progress_dict=None,
                 pytorch_device=0,
                 summary_writer=None,
                 save_val_vis=False,
                 val_vis_save_path='',
                 model_config=None
                 ):
        self.model = model
        self.optimizer = optimizer
        self.unique_label = unique_label
        self.unique_label_str = unique_label_str
        self.lovasz_softmax = lovasz_softmax
        self.loss_func = loss_func
        self.ignore_label = ignore_label
        self.pytorch_device = pytorch_device
        self.val = False
        self.best_val_miou = 0
        self.progress_value = 100
        self.progress_dict = dict()
        self.progress_dict['miou_per_epoch'] = []
        self.progress_dict['train_loss_list'] = []
        self.progress_dict['val_loss_list'] = []
        self.global_iter = 0
        self.epoch = 0
        if progress_dict:
            self.progress_dict = progress_dict
            print('Progress dictionary loaded')
            self.epoch = self.progress_dict['epoch']
            self.best_val_miou = self.progress_dict['best_val_miou']
            self.global_iter = self.progress_dict['global_iter']
        self.progres_save_path = progres_save_path
        self.progres_save_iters = save_progress_iters
        self.checkpoint_save_path = checkpoint_save_path
        self.model_save_path = model_save_dir
        self.scheduler = scheduler
        self.scheduler_save_path = scheduler_save_path
        self.optimizer_save_path = optimizer_save_path
        self.summary_writer = summary_writer
        self.save_val_vis = save_val_vis
        self.val_vis_save_path = val_vis_save_path
        self.model_config = model_config

        self.ssl=None

    def criterion(self, outputs, point_label_tensor, lcw=None):
        if self.ssl:
            lcw_tensor = torch.FloatTensor(lcw).to(self.pytorch_device)

            loss = self.lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                                       ignore=self.ignore_label, lcw=lcw_tensor) \
                   + self.loss_func(outputs, point_label_tensor, lcw=lcw_tensor)
        else:
            loss = self.lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                                       ignore=self.ignore_label) \
                   + self.loss_func(outputs, point_label_tensor)
        return loss

    def fast_iou_aprox(self, vox_predict_labels: torch.Tensor, vox_val_labels: torch.Tensor, ignore_label):
        res = ''
        iou_list = []
        id = 0
        tfpn_stats = None
        for label in self.unique_label:
            predict_positives = vox_predict_labels == label
            predict_negatives = torch.logical_and(vox_predict_labels != label, vox_predict_labels != ignore_label)
            val_positives = vox_val_labels == label
            val_negatives = torch.logical_and(vox_val_labels != label, vox_val_labels != ignore_label)

            true_positives = torch.sum(torch.logical_and(predict_positives, val_positives))
            false_positives = torch.sum(torch.logical_and(predict_positives, val_negatives))
            false_negatives = torch.sum(torch.logical_and(predict_negatives, val_positives))

            if label == 1:
                tfpn_stats = np.array([true_positives, false_positives, false_negatives])

            iou = true_positives / (true_positives + false_positives + false_negatives)
            if not torch.sum(val_positives) == 0:
                res += (f' {self.unique_label_str[id]} : {round(iou.item(), 3)} |')
            else:
                res += (f' {self.unique_label_str[id]} : NaN |')
                iou = np.nan
            id += 1
            iou_list.append(iou)
        return res, np.array(iou_list).reshape(-1, 1), tfpn_stats

    def label_pts(self, grid_ten, vox_label):
        for i in range(vox_label.shape[0]):
            point_labels_ten = vox_label[i, :]
            grid = grid_ten[i].cpu()
            pt_lab_lin = point_labels_ten.reshape(-1)
            grid_lin = np.ravel_multi_index(grid.numpy().T, self.model_config['output_shape'])
            out_labels = pt_lab_lin[torch.from_numpy(grid_lin)]
            labels = out_labels.reshape(-1, 1)
        return labels

    def map_outputs_to_pts(self, grid_indices: np.ndarray, outputs: np.ndarray):
        pt_vox_outputs = outputs[0, :, grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
        return pt_vox_outputs

    def validate(self, my_model, val_dataset_loader, val_batch_size):
        hist_list = []
        val_loss_list = []
        val_iou_list = []
        my_model.eval()
        val_pbar = tqdm(total=len(val_dataset_loader))

        tfpn_stats = np.array([0, 0, 0])

        with torch.no_grad():
            for i_iter_val, (
                    xyzil, val_vox_label, val_grid, val_pt_labs, val_pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
                val_dataset_loader):
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                  val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(self.pytorch_device)

                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                # aux_loss = loss_fun(aux_outputs, point_label_tensor)

                inp = val_label_tensor.size(0)

                # TODO: check if this is correctly implemented
                # hack for batch_size mismatch with the number of training example
                predict_labels = predict_labels[:inp, :, :, :, :]
                outputs = self.map_outputs_to_pts(val_grid[0], predict_labels.clone().cpu().numpy())

                loss = self.criterion(predict_labels, val_label_tensor, lcw)
                predict_labels = torch.argmax(predict_labels, dim=1)

                approx_class_iou, iou_list, cur_tfpn_stats = self.fast_iou_aprox(predict_labels.detach().cpu(), val_vox_label, ignore_label=self.ignore_label)

                # save the sample
                if self.save_val_vis:
                    xyz_labels = self.label_pts(val_grid_ten, predict_labels)
                    spacer_dim = np.zeros_like(xyz_labels[:, 0].cpu().numpy()).reshape(-1, 1)
                    xyzill = np.concatenate([xyzil[0, :, :], spacer_dim, xyz_labels.cpu().numpy().reshape(-1, 1)], axis=1)
                    name_iou = list(str(iou_list[-1][0] * 100).split('.'))[0]
                    vis_sample_name = f'vis_sample_{name_iou}_{i_iter_val}.npz'
                    vis_sample_path = os.path.join(self.val_vis_save_path, vis_sample_name)
                    np.savez(vis_sample_path, data=xyzill, original_data=xyzil, iou=np.array(iou_list), outputs=outputs)

                #print(predict_labels.shape, val_vox_label.shape)
                val_iou_list.append(iou_list)
                tfpn_stats += cur_tfpn_stats
                predict_labels = predict_labels.cpu().detach().numpy()
                for count, i_val_grid in enumerate(val_grid):
                    hist_list.append(fast_hist_crop(predict_labels[
                                                        count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        val_grid[count][:, 2]], val_pt_labs[count],
                                                        self.unique_label))
                val_loss_list.append(loss.detach().cpu().numpy())
                val_pbar.set_description(f'val_iou: {approx_class_iou}')
                val_pbar.update(1)
        my_model.train()
        val_pbar.close()
        return hist_list, val_loss_list, val_iou_list, tfpn_stats

    def fit(self, n_epochs, source_train_dataset_loader, train_batch_size, val_dataset_loader,
            val_batch_size, test_loader=None):

        global_iter = 1

        for epoch in range(self.epoch, n_epochs):
            pbar = tqdm(total=len(source_train_dataset_loader))
            # train the model
            loss_list = []
            self.model.train()
            # training with multi-frames and ssl:
            '''for i_iter_train, (
                    xyzil, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
                source_train_dataset_loader):

                z_diff = xyzil[:, 2].max() - xyzil[:, 2].min()
                if z_diff > 60:
                    print(f'skipping due to large z_diff: {z_diff}')
                    pbar.update(1)
                    continue

                #print(np.isnan(train_vox_label).any(), np.isnan(train_grid).any(), np.isnan(train_pt_fea[0]).any(), )
                #print(xyzil[0, :, :3].mean(axis=0))
                if xyzil.shape[1] > 300000:# or xyzil.shape[1] < 5000:
                    print(f'skipping due to too much points: {xyzil.shape[1]}')
                    pbar.update(1)
                    continue

                # call the validation and inference with
                train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(self.pytorch_device) for i in
                                    train_pt_fea]
                # train_grid_ten = [torch.from_numpy(i[:,:2]).to(self.pytorch_device) for i in train_grid]
                train_vox_ten = [torch.from_numpy(i).to(self.pytorch_device) for i in train_grid]
                point_label_tensor = train_vox_label.type(torch.LongTensor).to(self.pytorch_device)

                # forward + backward + optimize
                outputs = self.model(train_pt_fea_ten, train_vox_ten, train_batch_size)

                if torch.isnan(outputs).any():
                    print('outputs NaN')

                inp = point_label_tensor.size(0)
                # print(f"outputs.size() : {outputs.size()}")
                # TODO: check if this is correctly implemented
                # hack for batch_size mismatch with the number of training example
                outputs = outputs[:inp, :, :, :, :]
                ################################

                loss = self.criterion(outputs, point_label_tensor, lcw)

                # TODO: check --> to mitigate only one element tensors can be converted to Python scalars
                loss = loss.mean()

                if torch.isnan(loss).any():
                    print(f'ERROR: loss NaN')
                    self.optimizer.zero_grad()
                    print(xyzil.shape)
                    print((xyzil[0, :, :3].mean(axis=0)))
                    np.savez('./error_input.npz', data=xyzil)
                    exit(-1)

                loss.backward()
                self.optimizer.step()

                predict_labels = torch.argmax(outputs, dim=1)
                temp_iou, _, _ = self.fast_iou_aprox(predict_labels.clone().cpu(), train_vox_label, ignore_label=self.ignore_label)

                self.optimizer.zero_grad()

                loss_list.append(loss.item())
                if self.summary_writer:
                    self.summary_writer.add_scalar('train/loss', loss.item(), self.global_iter)

                if self.global_iter % self.progress_value == 0 and self.global_iter >= self.progress_value:
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' % (epoch, i_iter_train, np.mean(loss_list)))
                    else:
                        print('loss error')
                    self.progress_dict['train_loss_list'].append(np.mean(loss_list))

                if self.global_iter % self.progres_save_iters == 0 and self.global_iter >= self.progres_save_iters:
                    torch.save(self.model.state_dict(), self.checkpoint_save_path)
                    torch.save(self.optimizer.state_dict(), self.optimizer_save_path)
                    if self.scheduler:
                        torch.save(self.scheduler.state_dict(), self.scheduler_save_path)
                    self.progress_dict['latest_model'] = self.checkpoint_save_path
                    self.progress_dict['epoch'] = epoch
                    self.progress_dict['global_iters'] = self.global_iter
                    with open(self.progres_save_path, 'wb') as file:
                        pickle.dump(self.progress_dict, file)

                self.global_iter += 1
                pbar.update(1)
                pbar.set_description(f'loss: {loss.item():.{4}f} | iou: {temp_iou}')'''

            # ----------------------------------------------------------------------#
            # Evaluation/validation
            with torch.no_grad():
                hist_list, val_loss_list, iou_list, tfpn_stats = self.validate(self.model, val_dataset_loader, val_batch_size)

            # ----------------------------------------------------------------------#
            # Print validation mIoU and Loss
            print(f"--------------- epoch: {epoch} ----------------")
            iou = per_class_iu(sum(hist_list))
            true_positives, false_positives, false_negatives = tfpn_stats
            # tps / (tps + fps + fns)
            total_iou = true_positives / (true_positives + false_positives + false_negatives)
            print(f'Total lane mark IoU is: {total_iou}')
            print('Validation per class iou: ')
            iou2 = np.concatenate(iou_list, axis=1)
            for class_name, class_iou in zip(self.unique_label_str, iou):
                print('%s : %.2f%%' % (class_name, class_iou * 100))

            class_iou = np.nanmean(iou2, axis=1)
            print(f'class iou: {class_iou}')
            val_miou = np.nanmean(class_iou) * 100
            # del val_vox_label, val_grid, val_pt_fea

            # save model if performance is improved
            if self.best_val_miou < val_miou:
                self.progress_dict['best_val_miou'] = val_miou
                self.best_val_miou = val_miou
                torch.save(self.model.state_dict(), self.model_save_path)
                torch.save(self.optimizer.state_dict(), self.optimizer_save_path)
                torch.save(self.scheduler.state_dict(), self.scheduler_save_path)
                self.progress_dict['latest_model'] = self.model_save_path
                self.progress_dict['best_model'] = self.model_save_path

            # save model details
            self.progress_dict['miou_per_epoch'].append(val_miou)
            self.progress_dict['epoch'] = epoch
            self.progress_dict['global_iters'] = self.global_iter

            # save loss progress
            val_loss = np.mean(val_loss_list)
            self.progress_dict['val_loss_list'].append(val_loss)

            # save the progredd dictionary
            with open(self.progres_save_path, 'wb') as file:
                pickle.dump(self.progress_dict, file)

            print('Current val miou is %.3f while the best val miou is %.3f' %
                  (val_miou, self.best_val_miou))
            print('Current val loss is %.3f' % (np.mean(val_loss_list)))
            if self.summary_writer:
                self.summary_writer.add_scalar('val/loss', np.mean(val_loss_list), self.global_iter)
                self.summary_writer.add_scalar('val/miou', val_miou, self.global_iter)
                self.summary_writer.add_scalar('val/iou_lane', iou[1], self.global_iter)
                self.summary_writer.add_scalar('val/iou_other', iou[0], self.global_iter)

            # Uncomment to use the learning rate scheduler
            if self.scheduler:
                self.scheduler.step()
