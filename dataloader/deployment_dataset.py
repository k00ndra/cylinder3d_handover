import pickle

import numpy as np
from torch.utils import data
import os
from pyntcloud import PyntCloud

class DeploymentDataset(data.Dataset):

    def __init__(self, source_npz_file: str, partition_size: float, voxel_size: float):
        self.source_point_cloud = np.load(source_npz_file)['data'][:, :4]  # take xyzi
        #self.source_point_cloud = self.load_pcd_to_numpy(source_npz_file)[:,:4]
        self.partition_size = partition_size
        self.voxel_size = voxel_size
        self.downsamled_point_cloud = None
        self.masks = None

    def generate_split(self):
        print(f'Downsampling pointcloud with shape: {self.source_point_cloud.shape}')
        self.downsamled_point_cloud, grid_min_coords, grid_indices = self.voxel_downsample(self.source_point_cloud, self.voxel_size)
        print(f'Downsampled point cloud to {self.downsamled_point_cloud.shape[0]} points')
        self.masks = self.generate_split_masks(self.downsamled_point_cloud, self.partition_size)
        return self.masks, grid_min_coords, grid_indices, self.downsamled_point_cloud

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):

        sample_mask = self.masks[index]
        sample_xyzi = self.downsamled_point_cloud[sample_mask, :]

        sample_xyzi[:, :3] -= sample_xyzi[:, :3].mean(axis=0)

        xyz = sample_xyzi[:, :3]
        sig = sample_xyzi[:, -1].reshape(-1, 1) / 255
        index_labels = index * np.ones_like(sig).reshape(-1, 1)

        data_tuple = (xyz, index_labels.astype(np.uint8), sig)

        return data_tuple

    def load_pcd_to_numpy(self, pcd_filename: str):
        pcd_raw = PyntCloud.from_file(pcd_filename)
        pcd_numpy = pcd_raw.points.to_numpy()
        return pcd_numpy

    def voxel_downsample(self, point_cloud: np.ndarray, voxel_size: float):
        # extract the features
        xyz = point_cloud[:, :3]
        intensity = point_cloud[:, 3]

        # compute grid boundaries
        max_coords = xyz.max(axis=0)
        min_coords = xyz.min(axis=0)
        grid_size = np.ceil((max_coords - min_coords) / voxel_size).astype(np.int64) + 1  # x_size / y_size / z_size

        # compute the point to voxel indices
        centered_xyz = xyz - min_coords
        point_indices = np.floor(centered_xyz / voxel_size).astype(np.int64)

        # prepare indices and conversion table
        multi_index = (point_indices[:, 0], point_indices[:, 1], point_indices[:, 2])
        lin_point_indices = np.ravel_multi_index(multi_index, grid_size)
        unique_lin_indices, inverse_indices = np.unique(lin_point_indices, return_inverse=True)
        temp_indices = np.arange(unique_lin_indices.shape[0])

        indices = temp_indices[inverse_indices]

        # compute the counts of each voxel idx in pointcloud
        index_counts = np.zeros_like(unique_lin_indices)
        np.add.at(index_counts, indices, 1)

        # compute the centroids
        buffer_x = np.zeros_like(unique_lin_indices).astype(np.float64)
        buffer_y = buffer_x.copy()
        buffer_z = buffer_x.copy()
        np.add.at(buffer_x, indices, xyz[:, 0])
        np.add.at(buffer_y, indices, xyz[:, 1])
        np.add.at(buffer_z, indices, xyz[:, 2])
        centroids = np.concatenate([
            buffer_x.reshape(-1, 1), buffer_y.reshape(-1, 1), buffer_z.reshape(-1, 1)
        ], axis=1)
        centroids /= index_counts.reshape(-1, 1)

        # compute intensity average
        intensity_buffer = np.zeros_like(unique_lin_indices).astype(np.float64)
        np.add.at(intensity_buffer, indices, intensity)
        intensity_average = intensity_buffer / index_counts

        # assemble the pointcloud
        downsampled_point_cloud = np.concatenate([
            centroids, intensity_average.reshape(-1, 1)
        ], axis=1)

        grid_indices = np.floor((downsampled_point_cloud[:, :3] - min_coords) / voxel_size).astype(np.int64)

        return downsampled_point_cloud.astype(np.float64), min_coords, grid_indices

    def sub_partition_xy(self, point_cloud: np.ndarray, sub_partition_size: float):
        min_coords = point_cloud[:, :2].min(axis=0)
        sub_part_indices = np.floor((point_cloud[:, :2] - min_coords) / sub_partition_size).astype(np.int64)
        grid_size = sub_part_indices.max(axis=0) + 1

        return grid_size, sub_part_indices

    def choose_cover(self, grid_idx: np.ndarray, grid_indices: np.ndarray, grid_size: np.ndarray):

        def fits_grid(idx: np.ndarray, grid_size: np.ndarray):
            fits = idx[0] >= 0 and idx[1] >= 0 and idx[0] < grid_size[0] and idx[1] < grid_size[1]
            return fits

        offset_groups = [
            [
                np.array([0, 0]),
                np.array([-1, 0]),
                np.array([-1, 1]),
                np.array([0, 1])
            ],
            [
                np.array([0, 0]),
                np.array([0, 1]),
                np.array([1, 1]),
                np.array([1, 0])
            ],
            [
                np.array([0, 0]),
                np.array([1, 0]),
                np.array([1, -1]),
                np.array([0, -1])
            ],
            [
                np.array([0, 0]),
                np.array([0, -1]),
                np.array([-1, -1]),
                np.array([-1, 0])
            ]
        ]

        best_cover_mask = None
        best_score = -1

        for offset_group in offset_groups:
            cover_mask = np.zeros_like(grid_indices[:, 0], dtype=bool)
            for idx_offset in offset_group:
                new_idx = grid_idx + idx_offset
                if fits_grid(new_idx, grid_size):
                    # compute the number of matching indices -- points
                    cur_cover_mask = (grid_indices == new_idx.reshape(-1, 1).T).all(axis=1)
                    cover_mask = np.logical_or(cover_mask, cur_cover_mask)
            score = np.sum(cover_mask)
            if score > best_score:
                best_score = score
                best_cover_mask = cover_mask

        return best_cover_mask

    def delete_duplicate_masks(self, masks_list: list):
        stacked_masks = np.vstack(masks_list)
        unique_masks = np.unique(stacked_masks, axis=0)
        unique_masks_list = [unique_masks[i, :] for i in range(unique_masks.shape[0])]
        print(unique_masks_list[0].shape)
        return unique_masks_list

    def check_mask_covering(self, masks_list: list):
        check_mask = np.zeros_like(masks_list[0]).astype(bool)
        for mask in masks_list:
            check_mask = np.logical_or(check_mask, mask)

        covered = np.sum(check_mask)
        if covered != check_mask.shape[0]:
            print(f'ERROR: {check_mask.shape[0] - covered} out of {check_mask.shape[0]} points are not covered')

        return covered == np.sum(check_mask)

    def generate_split_masks(self, point_cloud: np.ndarray, partition_size: float):
        grid_size, grid_indices = self.sub_partition_xy(point_cloud, partition_size / 2)
        unique_grid_indices = np.unique(grid_indices, axis=0)
        print(unique_grid_indices.shape)
        cover_masks = []
        for temp_idx in range(unique_grid_indices.shape[0]):
            grid_idx = unique_grid_indices[temp_idx, :]
            cur_cover_mask = self.choose_cover(grid_idx, grid_indices, grid_size)
            cover_masks.append(cur_cover_mask)

        unique_cover_masks = self.delete_duplicate_masks(cover_masks)

        if not self.check_mask_covering(unique_cover_masks):
            exit(-1)

        return unique_cover_masks

    def fits_partition(self, points: np.ndarray, partition_size: float, epsilon: float = 0.1):
        fit_x = (np.abs(points[:, 0]) <= partition_size / 2 + epsilon).all()
        fit_y = (np.abs(points[:, 1]) <= partition_size / 2 + epsilon).all()
        return fit_x and fit_y

    def center_points(self, points: np.ndarray, partition_size: float):
        min_coords = points[:, :2].min(axis=0)
        shifted_points = points.copy()
        center_translation = min_coords + np.array([[partition_size / 2, partition_size / 2]])
        shifted_points[:, :2] -= center_translation
        return shifted_points


