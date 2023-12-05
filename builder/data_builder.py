# -*- coding:utf-8 -*-

import torch
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV, collate_fn_BEV_tta
from dataloader.pc_dataset import get_pc_model_class
from dataloader.complete_dataset import CompleteDataset

def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          train_dataset_confg,
          val_dataset_config,
          test_dataset_config,
          test_dataloader_config=None,
          grid_size=None,
          train_hypers=None):

    train_pt_dataset = CompleteDataset(train_dataset_confg)
    val_pt_dataset = CompleteDataset(val_dataset_config)

    train_dataset = get_model_class(dataset_config['dataset_type'])(
        train_pt_dataset,
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        #scale_aug=True,
        #transform_aug=True,
        return_test=True
    )

    val_dataset = get_model_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        #scale_aug=True,
        return_test=True
    )

    if test_dataloader_config is not None and test_dataset_config is not None:
        test_pt_dataset = CompleteDataset(test_dataset_config)
        test_dataset = get_model_class(dataset_config['dataset_type'])(
            test_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
            rotate_aug=True,
            #scale_aug=True,
            return_test=True
        )

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       pin_memory=True,
                                                       prefetch_factor=5,
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     pin_memory=True,
                                                     prefetch_factor=5,
                                                     num_workers=val_dataloader_config["num_workers"])
    test_dataset_loader = None
    if test_dataloader_config is not None:
        test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=test_dataloader_config["batch_size"],
                                                          collate_fn=collate_fn_BEV,
                                                          shuffle=test_dataloader_config["shuffle"],
                                                          pin_memory=True,
                                                          prefetch_factor=1,
                                                          num_workers=test_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader, test_dataset_loader
