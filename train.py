# -*- coding:utf-8 -*-
# author: Awet

import argparse
import os
import pickle
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
from utils.trainer_function import Trainer
#from tensorboardX import SummaryWriter
import shutil

# clear/empty cached memory used by caching allocator
#torch.cuda.empty_cache()
#torch.cuda.memory_summary(device='cuda:0', abbreviated=False)

# training
epoch = 0
best_val_miou = 0
global_iter = 0


def main(args):
    # pytorch_device = torch.device("cuda:2") # torch.device('cuda:2')
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'true'
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '9994'
    # os.environ['RANK'] = "0"
    # If your script expects `--local_rank` argument to be set, please
    # change it to read from `os.environ['LOCAL_RANK']` instead.
    # args.local_rank = os.environ['LOCAL_RANK']

    print('running')
    #os.environ['OMP_NUM_THREADS'] = "2"


    pytorch_device = 'cuda:0'

    config_path = args.config_path

    configs = load_config_data(config_path)
    # send configs parameters to pc_dataset
    # update_config(configs)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']
    test_dataloader_config = configs['test_data_loader']

    train_pt_dataset_config = configs['train_dataset']
    val_pt_dataset_config = configs['val_dataset']
    test_pt_dataset_config = configs['test_dataset']

    source_val_batch_size = val_dataloader_config['batch_size']
    source_train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_path = train_hypers['model_load_path']

    # NB: no ignored class
    unique_label = np.array([0, 1])
    unique_label_str = ['other', 'lane']

    # copy the configs into the folder
    base_name = train_hypers['checkpoint_save_path'][:-3]
    cpy_config_path = base_name + '_config.yaml'
    cpy_train_filenames = base_name + '_tr_files.pkl'
    cpy_val_filenames = base_name + '_val_filenames.pkl'
    shutil.copy(configs['train_dataset']['filenames_file'], cpy_train_filenames)
    shutil.copy(configs['val_dataset']['filenames_file'], cpy_val_filenames)
    shutil.copy(config_path, cpy_config_path)

    # build and load model
    model = model_builder.build(model_config)
    model = model.to(pytorch_device)
    if os.path.exists(model_path):
        model = load_checkpoint(model_path, model, map_location=pytorch_device[-1])
    else:
        print('model npt found --> starting from scratch')

    # get loss criterion
    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label,
                                                   weights=False, fl=False)

    # get dataloaders
    source_train_dataset_loader, source_val_dataset_loader, source_test_dataset_loader = data_builder.build(
        dataset_config,
        train_dataloader_config,
        val_dataloader_config,
        train_dataset_confg=train_pt_dataset_config,
        val_dataset_config=val_pt_dataset_config,
        test_dataset_config=None,
        test_dataloader_config=None,
        grid_size=grid_size,
        train_hypers=train_hypers)

    # get optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_hypers["learning_rate"],
        weight_decay=train_hypers['weight_decay']
    )

    # get scheduler
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
    #                                                 steps_per_epoch=len(source_train_dataset_loader),
    #                                                 epochs=train_hypers["max_num_epochs"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.7)

    global global_iter, best_val_miou, epoch
    print("|-------------------------Training started-----------------------------------------|")

    #summary_writer = SummaryWriter('./final_test03/log01')
    summary_writer = None
    # Define training mode and function
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        unique_label=unique_label,
        unique_label_str=unique_label_str,
        lovasz_softmax=lovasz_softmax,
        loss_func=loss_func,
        ignore_label=ignore_label,
        checkpoint_save_path=train_hypers['checkpoint_save_path'],
        scheduler=scheduler,
        model_save_dir=train_hypers['model_save_path'],
        pytorch_device=pytorch_device,
        save_val_vis=train_hypers['save_vis'],
        val_vis_save_path=train_hypers['vis_save_path'],
        summary_writer=summary_writer,
        model_config=model_config
        )

    print('trainer constructed')

    trainer.fit(train_hypers["max_num_epochs"],
                source_train_dataset_loader,
                source_train_batch_size,
                source_val_dataset_loader,
                source_val_batch_size,
                test_loader=None
                )


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path',
                        default='config/waymo.yaml')
    parser.add_argument('-g', '--mgpus', action='store_true', default=False)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)

