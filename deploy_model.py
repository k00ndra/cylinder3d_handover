

from dataloader.deployment_dataset import DeploymentDataset
from builder import model_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
from dataloader.dataset_semantickitti import get_model_class, collate_fn_BEV
import numpy as np
import torch
from tqdm import tqdm
import sys
import os
import torch.nn.functional as F

DEVICE = 'cuda:0'


SAVE_DEBUG_PATCHES = False
RETURN_UPSAMPLED_MASK = False
SELECTION_STRATEGY = 'last'  # 'last', 'or'
# TODO confidence not yet working

def get_dataloader(source_path: str, partition_size: float, voxel_size: float, configs):
    pt_dataset = DeploymentDataset(
        source_npz_file=source_path,
        partition_size=partition_size,
        voxel_size=voxel_size
    )

    masks, grid_min_coords, grid_indices, downsamled_point_cloud = pt_dataset.generate_split()

    dataset = get_model_class('voxel_dataset')(
        pt_dataset,
        grid_size=configs['model_params']['output_shape'],
        flip_aug=False,
        fixed_volume_space=configs['dataset_params']['fixed_volume_space'],
        max_volume_space=configs['dataset_params']['max_volume_space'],
        min_volume_space=configs['dataset_params']['min_volume_space'],
        ignore_label=configs['dataset_params']["ignore_label"],
        rotate_aug=False,
        return_test=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=collate_fn_BEV,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=1,
        num_workers=3
    )

    return dataloader, masks, grid_min_coords, grid_indices, downsamled_point_cloud

def upsample_mask(full_point_cloud: np.ndarray, downsampled_mask: np.ndarray, downsampled_indices: np.ndarray, min_coords: np.ndarray, voxel_size: float):
    def get_mask(subset: np.ndarray, source: np.ndarray):
        b = subset
        a = source
        baseval = np.max([a.max(), b.max()]) + 1
        n_cols = a.shape[1]
        a = a * baseval ** np.array(range(n_cols))
        b = b * baseval ** np.array(range(n_cols))
        c = np.isin(np.sum(a, axis=1), np.sum(b, axis=1))
        return c

    full_grid_indices = np.floor((full_point_cloud[:, :3] - min_coords.reshape(-1, 1).T) / voxel_size).astype(np.int64)
    valid_downsampled_indices = downsampled_indices[downsampled_mask.astype(bool)]
    full_mask = get_mask(np.unique(valid_downsampled_indices, axis=0), full_grid_indices)
    return full_mask

def run_model(model, dataloader, model_config):
    def label_pts(self, grid_ten, vox_label):
        labels = []
        for i in range(vox_label.shape[0]):
            point_labels_ten = vox_label[i, :]
            grid = grid_ten[i].cpu()
            pt_lab_lin = point_labels_ten.reshape(-1)
            grid_lin = np.ravel_multi_index(grid.numpy().T, self.model_config['output_shape'])
            out_labels = pt_lab_lin[torch.from_numpy(grid_lin)]
            labels.append(out_labels.reshape(-1, 1))
        return labels

    def map_outputs_to_pts(grid_indices: np.ndarray, outputs: np.ndarray):
        pt_vox_outputs = outputs[0, :, grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]]
        return pt_vox_outputs

    print(f'Getting model output -- {len(dataloader)}')

    if SAVE_DEBUG_PATCHES and not os.path.isdir('./debug_patches'):
        os.makedirs('./debug_patches/')
        print('saving debug outputs to: ./debug_patches')

    outputs_dict = dict()
    model.eval()
    pbar = tqdm(total=len(dataloader))
    for i_iter_train, (
            xyzil, train_vox_label, train_grid, _, train_pt_fea, ref_st_idx, ref_end_idx, lcw) in enumerate(
        dataloader):

        index = int(xyzil[0, 0, -1].item())
        pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in train_pt_fea]
        vox_ten = [torch.from_numpy(i).to(DEVICE) for i in train_grid]
        label_tensor = train_vox_label.type(torch.LongTensor).to(DEVICE)
        grid_ten = [torch.from_numpy(i).to(DEVICE) for i in train_grid]

        outputs = model(pt_fea_ten, vox_ten, 1)

        if torch.isnan(outputs).any():
            print('outputs NaN')

        inp = label_tensor.size(0)
        outputs = outputs[:inp, :, :, :, :]
        predict_probabilities = F.softmax(outputs, dim=1)
        predict_labels = torch.argmax(outputs, dim=1)
        labels = label_pts(grid_ten, predict_labels)
        outputs_dict[index] = [xyzil, labels]

        if SAVE_DEBUG_PATCHES:
            patch_path = os.path.join('./debug_patches', f'patch_{index}.npz')
            patch_pts = np.concatenate([xyzil[0, :, :], labels.cpu().numpy().reshape(-1, 1)], axis=1)
            np.savez(patch_path, data=patch_pts)

        pbar.update(1)
    pbar.close()
    return outputs_dict

def get_labels(outputs_dict: dict, downsampled_point_cloud: np.ndarray, masks: list):

    # TODO implement overlap decision heuristics - use softmax probabilities
    label_dim = np.zeros_like(downsampled_point_cloud[:, 0])
    confidences = -np.ones_like(label_dim)

    for index in range(len(masks)):
        _, labels = outputs_dict[index]
        mask = masks[index]

        # TODO temp
        if SELECTION_STRATEGY == 'last':
            label_dim[mask] = labels.reshape(-1)
        elif SELECTION_STRATEGY == 'or':
            label_dim[mask] = np.logical_or(label_dim[mask].astype(bool), labels.astype(bool))
        else:
            print('unknown strategy')

        #prob_diff_mask = prob_diffs > confidences[mask]
        #label_dim[mask][prob_diff_mask] = labels.reshape(-1)[prob_diff_mask]

    labeled_point_cloud = np.concatenate([downsampled_point_cloud, label_dim.reshape(-1, 1)], axis=1)
    return labeled_point_cloud

def deploy_model(config_path: str, source_path: str, partition_size: float, voxel_size: float):

    # load configuration
    configs = load_config_data(config_path)
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    model_path = train_hypers['model_load_path']

    # load model
    model = model_builder.build(model_config)
    model = model.to(DEVICE)
    model = load_checkpoint(model_path, model, map_location=DEVICE[-1])

    # load dataset
    dataloader, masks, grid_min_coords, grid_indices, downsampled_point_cloud = get_dataloader(
        source_path=source_path,
        partition_size=partition_size,
        voxel_size=voxel_size,
        configs=configs
    )

    outputs_dict = run_model(model, dataloader, model_config)

    print('Labeling outputs')
    labeled_downsample = get_labels(outputs_dict, downsampled_point_cloud, masks)

    if RETURN_UPSAMPLED_MASK:
        labels = labeled_downsample[:, -2]
        full_point_cloud = np.load(source_path)['data'][:, :3]
        full_labels = upsample_mask(
            full_point_cloud,
            labels,
            grid_indices,
            grid_min_coords,
            voxel_size
        )

        return labeled_downsample, full_labels
    else:
        return labeled_downsample


if __name__ == '__main__':

    config_path = sys.argv[1]
    source_path = sys.argv[2]
    partition_size = float(sys.argv[3])
    voxel_size = float(sys.argv[4])
    output_path = sys.argv[5]

    result = deploy_model(config_path, source_path, partition_size, voxel_size)
    if RETURN_UPSAMPLED_MASK:
        labeled_point_cloud, full_labels = result
    else:
        labeled_point_cloud = result
        full_labels = np.array([-1])
    print(f'Saving output to {output_path}')
    np.savez(output_path, data=labeled_point_cloud, full_labels=full_labels)

    print('done')








