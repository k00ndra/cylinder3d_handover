import numpy as np
import os
import sys
import pickle
from sklearn.linear_model import RANSACRegressor
import torch
from pytorch3d.ops import ball_query
from tqdm import tqdm
import yaml


def histogram_equalization(intensities: np.ndarray):
    # Calculate histogram
    hist, bins = np.histogram(intensities, bins=255, range=[0, 1], density=True)
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()

    # Perform histogram equalization
    equalized_intensities = np.interp(intensities, bins[:-1], cdf_normalized)

    return equalized_intensities


def voxel_downsample(point_cloud: np.ndarray, voxel_size: float):

    # extract the features
    xyz = point_cloud[:, :3]
    intensity = point_cloud[:, 3]
    labels = point_cloud[:, -1]

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

    # compute the counts
    label_buffer = np.zeros_like(unique_lin_indices).astype(int)
    label_mask = (labels > 0).astype(int)
    np.add.at(label_buffer, indices, label_mask)

    label_modus = label_buffer > 0

    # assemble the pointcloud
    downsampled_point_cloud = np.concatenate([
        centroids, intensity_average.reshape(-1, 1), label_modus.reshape(-1, 1)
    ], axis=1)

    return downsampled_point_cloud.astype(np.float64)


def refine_labels(sample: np.ndarray,  label_col: int, radius: float, plane_inl_tresh: float, use_ball_query: bool = False):
    valid_points = sample[sample[:, label_col].astype(bool),:]
    valid_mask = sample[:, label_col].astype(bool)

    if np.count_nonzero(valid_mask) <= 20:
        return valid_mask

    #visualize_points(valid_points)

    ransac = RANSACRegressor(residual_threshold=plane_inl_tresh)
    ransac.fit(valid_points[:,:2], valid_points[:, 2])
    plane_mask = np.abs(ransac.predict(sample[:, :2]) - sample[:, 2]) <= plane_inl_tresh

    #print(np.count_nonzero(plane_mask))
    #visualize_points(sample[plane_mask,:])

    if use_ball_query:
        query_ten = torch.from_numpy(sample[plane_mask,:3][valid_mask[plane_mask],:]).unsqueeze(0).cuda().float()
        source_ten = torch.from_numpy(sample[plane_mask,:3][~valid_mask[plane_mask],:]).unsqueeze(0).cuda().float()

        knn = ball_query(query_ten, source_ten, radius=radius, return_nn=False)
        indices = knn[1].squeeze(0).cpu().numpy()

        intensities = np.ones_like(indices)
        valid_indices = indices[indices != -1]
        flat_intensities = np.take(sample[:,3], valid_indices)
        indices[indices != -1] = flat_intensities

        reference_intensities = sample[np.logical_and(plane_mask, valid_mask),3].reshape(-1, 1)
        intensity_diffs = np.abs(intensities - reference_intensities)
        new_added_mask = np.logical_and(intensity_diffs < 10, indices != -1)
        new_added_mask = np.logical_and(new_added_mask, intensities >= 95)

        added_pt_indices = np.unique(indices[new_added_mask])

    final_mask = valid_mask.copy()
    final_mask[~plane_mask] = 0
    if use_ball_query:
        final_mask[np.logical_and(~valid_mask, plane_mask)][added_pt_indices] = 1

    return final_mask


def to_supervoxels(point_cloud: np.ndarray, supervoxel_size: float, min_pts_limit: int, downsample_voxel_size: float, label_col: int,  target_folder: str, subfolder_name: str,):
    min_coords = point_cloud[:, :2].min(axis=0)
    max_coords = point_cloud[:, :2].max(axis=0)

    grid_dimensions = np.ceil((max_coords - min_coords) / supervoxel_size)
    num_voxels = np.prod(grid_dimensions, axis=0)
    print(num_voxels)
    voxel_indices = np.floor((point_cloud[:, :2] - min_coords) / supervoxel_size).astype(np.int64)

    centered_supervoxel_points = point_cloud.copy()
    centered_supervoxel_points[:, :2] -= min_coords
    centered_supervoxel_points[:, :2] -= supervoxel_size * voxel_indices
    centered_supervoxel_points[:, :2] -= 0.5 * supervoxel_size

    lin_voxel_indices = voxel_indices[:, 0] * grid_dimensions[1] + voxel_indices[:, 1]

    supervoxels = []
    pbar = tqdm(total=num_voxels)
    sample_id = 0
    for supervoxel_id in range(num_voxels.astype(int)):
        supervoxel_mask = lin_voxel_indices == supervoxel_id
        supervoxel = centered_supervoxel_points[supervoxel_mask, :]
        if supervoxel.shape[0] >= min_pts_limit:
            supervoxel_sample = np.concatenate([
                supervoxel[:, :4], supervoxel[:, label_col].reshape(-1, 1)
            ], axis=1)
            if equalize:
                supervoxel_sample[:, 3] = histogram_equalization(supervoxel_sample[:, 3] / 255)
            else:
                supervoxel_sample[:, 3] = supervoxel_sample[:, 3] / 255
            supervoxel_sample_down = voxel_downsample(supervoxel_sample, downsample_voxel_size)
            supervoxel_sample_down[:, 2] -= supervoxel_sample_down[:, 2].mean()
            if supervoxel_sample_down.shape[0] >= 100:
                sample_id = save_supervoxel(supervoxel_sample_down, target_folder, subfolder_name, sample_id)
        pbar.update(1)
    pbar.close()

    return supervoxels


def accumulate_point_cloud(sequence_folder: str, label_col: int, bq_radius: float, plane_inl_tresh: float, use_bq: bool):

    pc_buffer = []
    pbar = tqdm(total=len(os.listdir(sequence_folder)))
    for file_name in os.listdir(sequence_folder):
        if file_name.endswith('.npz'):
            file_path = os.path.join(sequence_folder, file_name)
            with np.load(file_path, mmap_mode='r') as pc_file:
                cur_pc = pc_file['data']
            refined_labels = refine_labels(cur_pc, label_col, bq_radius, plane_inl_tresh, use_ball_query=use_bq)
            cur_sample = np.concatenate([cur_pc[:, :4], refined_labels.reshape(-1, 1)], axis=1)
            pc_buffer.append(cur_sample)
        pbar.update(1)
    pbar.close()
    accumulated_point_cloud = np.concatenate(pc_buffer, axis=0)
    return accumulated_point_cloud

def save_supervoxel(supervoxel: list, target_folder: str, subfolder_name: str, sample_id: int):

    target_subfolder_path = os.path.join(target_folder, subfolder_name)
    if not os.path.isdir(target_subfolder_path):
        os.makedirs(target_subfolder_path)
    sample_name = f'sample_{sample_id}.npz'
    sample_path = os.path.join(target_subfolder_path, sample_name)
    np.savez(sample_path, data=supervoxel)
    return sample_id + 1




def create_accumulated_dataset(source_folder: str, target_folder: str, supervoxel_size: float, min_pts_limit: int, voxel_size: float, label_col: int, bq_radius: float, plane_inl_tresh: float, use_bq: bool, equalize: bool):

    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        if os.path.isdir(folder_path):
            print(f'processing folder: {folder_name}')
            accumulated_pc = accumulate_point_cloud(folder_path, label_col, bq_radius, plane_inl_tresh, use_bq)
            print('    accumulated point cloud')
            supervoxels = to_supervoxels(accumulated_pc, supervoxel_size, min_pts_limit, voxel_size, label_col, target_folder, folder_name)
            print('    split to supervoxels')
    print('done')



if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path, 'rb') as file:
        config = yaml.safe_load(file)

    source_folder = config['source_folder']
    target_folder = config['target_folder']
    supervoxel_size = config['supervoxel_size']
    min_pts_tresh = config['min_pts_tresh']
    voxel_size = config['voxel_size']
    label_col = config['label_col']
    use_bq = config['use_bq']
    bq_radius = config['bq_radius']
    plane_inl_tresh = config['plane_inl_tresh']
    equalize = config['equalize']

    create_accumulated_dataset(source_folder, target_folder, supervoxel_size, min_pts_tresh, voxel_size, label_col,bq_radius, plane_inl_tresh, use_bq, equalize)



