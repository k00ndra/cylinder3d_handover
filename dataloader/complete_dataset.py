import pickle

import numpy as np
from torch.utils import data
import os
from bend_augmentation import SingleRandomBend
import random
from pyntcloud import PyntCloud


class CompleteDataset(data.Dataset):

    def __init__(self, dataset_config):
        self.config = dataset_config
        self.filenames = []
        self.parse_valeo()

        if self.config['use_bending']:
            self.bend_augmentation = SingleRandomBend(
                self.config['bend_max_len'],
                self.config['bend_max_k'],
                test_print=False
            )

    def load_pcd_to_numpy(self, pcd_filename: str):
        pcd_raw = PyntCloud.from_file(pcd_filename)
        pcd_numpy = pcd_raw.points.to_numpy()
        return pcd_numpy

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):

        sample_path = self.filenames[index]

        sample_xyzi, sample_labels = self.get_valeo_sample(sample_path)

        if sample_xyzi.shape[0] > 2:
            if self.config['use_bending']:
                sample_xyzi = self.bend_augmentation(sample_xyzi)
                sample_xyzi[:, :3] -= sample_xyzi[:, :3].mean(axis=0)

            if self.config['use_intensity_shift']:
                sample_xyzi = self.shift_intensity(sample_xyzi)

            if self.config['use_intensity_jitter']:
                sample_xyzi = self.intensity_jitter(sample_xyzi, 0.1, section_size=7)
        else:
            print(f'ERROR: sample too small {sample_xyzi.shape}')
        # normalize the intensities
        xyz = sample_xyzi[:, :3]
        #print(xyz.shape)
        sig = sample_xyzi[:, 3]

        # TODO temp
        labels = sample_labels.reshape(-1, 1).astype(np.uint8)
        data_tuple = (xyz, labels, sig)  # TODO check order
        return data_tuple

    def align_z_axis(self, sample: np.ndarray, perc: float = 15):
        bottom_value = np.percentile(sample[:, 2], perc)
        bottomed_sample = sample.copy()
        bottomed_sample[:, 2] -= bottom_value
        return bottomed_sample

    def load_sample(self, file_path: str):
        sample = self.load_pcd_to_numpy(file_path)

        nan_mask = np.isnan(sample).any(axis=1)
        if nan_mask.sum() > 0:
            print('NaNs in sample')
            sample[nan_mask, :] = 0

        sample_center = np.mean(sample[:, :3], axis=0)
        sample[:, :3] -= sample_center
        #sample = self.align_z_axis(sample)

        # normalize intensity if needed
        if (sample[:, 3] > 2).any():
            sample[:, 3] /= 255

        return sample

    def get_valeo_sample(self, file_path: str):

        sample = self.load_sample(file_path)
        sample_xyzi = sample[:, :4]
        sample_labels = sample[:, -1]

        if self.config['use_gamma']:
            sample_xyzi = self.random_gamma_correction(sample_xyzi)

        if self.config['use_rsj']:
            sample_xyzi = self.random_spatial_jitter(sample_xyzi)

        return sample_xyzi, sample_labels


    def parse_valeo(self):

        filenames_file = self.config['filenames_file']
        with open(filenames_file, 'rb') as file:
            filenames = pickle.load(file)

        valid_counter = 0
        invalid_counter = 0

        for file_name in filenames:

            if file_name.endswith('.pcd'):
                #file_path = os.path.join(source_folder, file_name)
                file_path = file_name
                self.filenames.append(file_path)
                valid_counter += 1
            else:
                invalid_counter += 1
        print(f'Parsed Valeo dataset --> found {valid_counter} valid subfolders and {invalid_counter} invalid subfolders')


    def intensity_jitter(self, sample: np.ndarray, rng: float, section_size = 1):
        intensities = sample[:, 3]
        xy = sample[:, :2]
        section_indices = np.floor((xy - xy.min(axis=0)) / section_size).astype(np.int64)
        noise = np.random.uniform(-rng, rng, section_indices.max(axis=0) + 7)
        augmented_intensities = intensities.copy()
        augmented_intensities += noise[section_indices[:, 0], section_indices[:, 1]]

        augmented_sample = sample.copy()
        augmented_sample[:, 3] = augmented_intensities
        return augmented_sample

    def shift_intensity(self, sample: np.ndarray):
        int_shift = random.uniform(-0.15, 0.15)
        #print(int_shift)
        intensities = sample[:, 3]
        augmented_intensities = intensities + int_shift
        augmented_intensities[augmented_intensities > 1] = 1
        augmented_intensities[augmented_intensities < 0] = 0
        augmented_sample = sample.copy()
        augmented_sample[:, 3] = augmented_intensities
        return augmented_sample

    def random_gamma_correction(self, sample: np.ndarray, low_gamma: float = 0.9, upper_gamma: float = 1.1):

        def gamma_correction(sample: np.ndarray, gamma: float):
            intensities = sample[:, 3]
            corrected_intensities = np.power(intensities, gamma)
            cor_sample = sample.copy()
            cor_sample[:, 3] = corrected_intensities
            return cor_sample

        gamma = np.random.uniform(low_gamma, upper_gamma, 1)
        corrected_sample = gamma_correction(sample, gamma)
        return corrected_sample

    def random_spatial_jitter(self, sample: np.ndarray, max_len: float = 0.03):
        augmented_sample = sample.copy()

        def generate_random_unit_vectors(n):
            # Generate random points on the unit sphere
            phi = np.random.uniform(0, np.pi, n)  # Polar angle
            theta = np.random.uniform(0, 2 * np.pi, n)  # Azimuthal angle

            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)

            return np.column_stack((x, y, z))

        def generate_random_vectors_with_length(n, max_length):
            # Generate random unit vectors
            random_unit_vectors = generate_random_unit_vectors(n)

            lengths = np.random.uniform(0, max_length, random_unit_vectors.shape[0]).reshape(-1, 1)

            # Scale the unit vectors to the desired length
            random_vectors = np.multiply(random_unit_vectors, lengths)

            return random_vectors

        random_jitter_vectors = generate_random_vectors_with_length(sample.shape[0], max_len)
        augmented_sample[:, :3] += random_jitter_vectors
        return augmented_sample

