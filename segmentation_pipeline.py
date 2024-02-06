import numpy as np
import sys
from ruamel.yaml import YAML
from plot_segmentation import visualize_points
from deploy_model import deploy_model, parse_input, merge_point_cloud


def segmentation_main(data_dict, config):

    point_cloud = data_dict['data']
    print(f'Segmenting pointcloud with shape: {point_cloud.shape}')

    model_results = deploy_model(
        point_cloud=point_cloud,
        pipeline_config=config
    )


    print('Saving model results')

    # using upsampled mask --> mask compatible with original pointcloud
    mask_model_results, downsampled_model_results = model_results
    final_mask = mask_model_results.astype(bool)
    final_mask = np.logical_and(final_mask, point_cloud[:, 3] >= config['POSTPROCESSING_INTENSITY_TRESH'])
    data_dict['segmentation_mask'] = final_mask
    data_dict['segmentation'] = np.concatenate([point_cloud[final_mask, :3], point_cloud[final_mask, -1].reshape(-1, 1)], axis=1)

    # save downsampled outputs for debugging
    downsampled_final_mask = downsampled_model_results[:, -2].astype(bool)
    downsampled_point_cloud = downsampled_model_results[:, :4]
    downsampled_final_mask = np.logical_and(downsampled_final_mask, downsampled_point_cloud[:, 3] >= config['POSTPROCESSING_INTENSITY_TRESH']).reshape(-1)
    data_dict['segmentation_downsample'] = downsampled_point_cloud[downsampled_final_mask, :]

    if config['ANIMATION']:
        visualize_points(point_cloud, point_cloud[final_mask, :])

if __name__ == '__main__':

    npz_path = sys.argv[1]
    pipeline_config_path = sys.argv[2]

    # load point cloud
    scan_list, pose_list = parse_input(npz_path)
    data_dict = dict()
    data_dict['data'] = merge_point_cloud(frames_list=scan_list, pose_list=pose_list, pipeline_config=config)
    data_dict['poses'] = pose_list
    print(f"pointcloud loaded {data_dict['data'].shape}")

    # load config
    yaml = YAML()
    yaml.default_flow_style = False
    with open(pipeline_config_path, "r") as f:
        config = yaml.load(f)
    print('config loaded')

    # run the pipeline
    segmentation_main(data_dict, config)

    # save the results
    output_file_path = npz_path[:-4] + '-segmentation.npz'
    np.savez(output_file_path, data=data_dict['data'], labels=data_dict['segmentation_mask'])
    print('done')


