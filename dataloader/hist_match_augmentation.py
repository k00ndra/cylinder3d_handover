import numpy as np
import random
import os
from tqdm import tqdm
import pickle

class HistMatchAug:

    def __init__(self, matching_dataset_folder: str, load: bool=True, load_path: str='./augmentation_histograms.pkl'):
        filenames = os.listdir(matching_dataset_folder)
        valid_filenames = []
        for filename in filenames:
            if filename.endswith('.npz'):
                valid_filenames.append(filename)
        self.source_folder = matching_dataset_folder
        self.matching_filenames = valid_filenames

        if load and os.path.isfile(load_path):
            print('loading histograms')
            with open(load_path,'rb') as file:
                self.histograms = pickle.load(file)
                print(len(self.histograms))
        else:
            print('processing histograms')
            pbar = tqdm(total=len(self.matching_filenames))
            self.histograms = []
            for filename in self.matching_filenames:
                pth = os.path.join(self.source_folder, filename)
                with np.load(pth) as pcd_file:
                    pcd = pcd_file['data']
                intensities = pcd[:, 3]
                hist, bins = np.histogram(intensities, bins=256, range=(0, 256), density=True)
                self.histograms.append([hist, bins])
                pbar.update(1)

            with open(load_path, 'wb') as file:
                pickle.dump(self.histograms, file)

    def histogram_match(self, pointcloud2: np.ndarray):
        match_idx = random.randint(0, len(self.histograms) - 2)

        intensity2 = pointcloud2[:, 3]

        # Compute histograms for both intensity values
        hist1, bins1 = self.histograms[match_idx]
        hist2, bins2 = np.histogram(intensity2, bins=256, range=(0, 256), density=True)

        # Compute cumulative distribution functions (CDFs) for both histograms
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()

        # Interpolate the intensity values from pointcloud2 to match the CDF of pointcloud1
        matched_intensity2 = np.interp(intensity2, bins1[:-1], cdf1)

        # Replace the intensity values in pointcloud2 with the matched values
        pointcloud2[:, 3] = matched_intensity2 * 255

        return pointcloud2

    def augment(self, point_cloud: np.ndarray):

        # perform histogram matching
        aug_point_cloud = self.histogram_match( point_cloud)

        return aug_point_cloud