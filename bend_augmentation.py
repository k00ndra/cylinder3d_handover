import numpy as np
class SingleRandomBend:

    def __init__(self, max_len: float, max_k: float, test_print: bool = False):

        self.max_len = max_len
        self.max_coefficient = max_k
        self.test_print = test_print

    def __call__(self, x):
        point_cloud = x

        # select axis to bend:
        temp = np.random.rand(1).item()
        from_axis = 'y'
        to_axis = 'x'
        neutral_axis = 'z'

        # select length
        length = np.random.rand(1).item() * self.max_len * 2

        # select coefficient
        coefficient = np.random.rand(1) * self.max_coefficient
        #coefficient = 0.04
        if abs(coefficient) < 1e-3:
            return point_cloud

        transformed_point_cloud = self.tranform_to_pca_coords(point_cloud)

        # select start and end y coordinates
        min_start = transformed_point_cloud[:, 1].min().item()
        max_start = transformed_point_cloud[:, 1].max().item()
        start = np.random.rand(1).item() * (max_start - min_start) + min_start
        end = start + length

        if self.test_print:
            print('bend parameters', start, end, coefficient, from_axis, to_axis)

        dir = 1
        if np.random.rand(1) > 0.5:
            dir = -1

        bent_point_cloud = self.bend(transformed_point_cloud, start, end, dir * coefficient, from_axis, to_axis, neutral_axis)
        return bent_point_cloud

    def get_pca(self, point_cloud: np.ndarray):
        centroid = np.mean(point_cloud[:, :3], axis=0)
        centered_point_cloud = point_cloud[:, :3].copy() - centroid
        covariance_matrix = np.dot(centered_point_cloud.T, centered_point_cloud) / (centered_point_cloud.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        return eigenvalues.real, eigenvectors.real

    def get_dominant_direction(self, point_cloud: np.ndarray):
        eigen_values, eigen_vectors = self.get_pca(point_cloud)
        dominant_idx = np.argmax(eigen_values)
        dominant_direction_vec = eigen_vectors[dominant_idx, :]
        dominant_direction_vec /= np.linalg.norm(dominant_direction_vec)
        return dominant_direction_vec

    def get_perp_vec_in_plane(self, vector: np.ndarray, plane_normal: np.ndarray = np.array([0, 0, 1], dtype=np.float32)):
        # normalize the vectors

        query_vector = vector.copy() / np.linalg.norm(vector)
        reference_vector = plane_normal.copy() / np.linalg.norm(plane_normal)

        perpendicular_vector = np.cross(query_vector, reference_vector)
        perpendicular_vector /= np.linalg.norm(perpendicular_vector)
        return perpendicular_vector

    def get_direction_distances(self, point_cloud: np.ndarray, direction_vec: np.ndarray):
        dot_products = np.matmul(point_cloud[:, :3], direction_vec.T)
        return dot_products

    def center_point_cloud_xy(self, point_cloud: np.ndarray):
        center = np.mean(point_cloud[:, :2], axis=0)
        centered_pc = point_cloud.copy()
        centered_pc[:, :2] -= center
        return centered_pc

    def tranform_to_pca_coords(self, point_cloud: np.ndarray):

        transformed_point_cloud = self.center_point_cloud_xy(point_cloud)

        y_vec = self.get_dominant_direction(transformed_point_cloud)
        x_vec = self.get_perp_vec_in_plane(y_vec)
        z_vec = np.array([0, 0, 1], dtype=np.float32)

        x = self.get_direction_distances(transformed_point_cloud[:, :3], x_vec)
        y = self.get_direction_distances(transformed_point_cloud[:, :3], y_vec)
        z = self.get_direction_distances(transformed_point_cloud[:, :3], z_vec)

        transformed_point_cloud[:, 0] = x
        transformed_point_cloud[:, 1] = y
        transformed_point_cloud[:, 2] = z

        return transformed_point_cloud

    def bend(self, transformed_point_cloud: np.ndarray, start: float, end: float, coefficient: float, from_axis: str, to_axis: str, neutral_axis: str):

        config_dict = {
            'x': 0,
            'y': 1,
            'z': 2
        }

        f_idx = config_dict[from_axis]
        t_idx = config_dict[to_axis]
        n_idx = config_dict[neutral_axis]

        from_coord = transformed_point_cloud[:, f_idx]
        to_coord = transformed_point_cloud[:, t_idx]
        neutral_coord = transformed_point_cloud[:, n_idx]

        theta = np.zeros_like(transformed_point_cloud[:, 0])
        theta[from_coord > start] = coefficient * (from_coord[from_coord > start] - start)
        theta[from_coord >= end] = coefficient * (end - start)

        rho = 1 / coefficient

        new_neutral_coord = neutral_coord.copy()

        new_from_coord = from_coord.copy()
        new_from_coord[from_coord >= start] = start - np.sin(theta[from_coord >= start]) * (to_coord[from_coord >= start] - rho)
        new_from_coord[from_coord > end] = start - np.sin(theta[from_coord > end]) * (to_coord[from_coord > end] - rho) + np.cos(theta[from_coord > end]) * (from_coord[from_coord > end] - end)

        new_to_coord = to_coord.copy()
        new_to_coord[from_coord >= start] = rho + np.cos(theta[from_coord >= start]) * (to_coord[from_coord >= start] - rho)
        new_to_coord[from_coord > end] = rho + np.cos(theta[from_coord > end]) * (to_coord[from_coord > end] - rho) + np.sin(theta[from_coord > end]) * (from_coord[from_coord > end] - end)

        bent_point_cloud = transformed_point_cloud.copy()
        bent_point_cloud[:, f_idx] = new_from_coord
        bent_point_cloud[:, t_idx] = new_to_coord
        bent_point_cloud[:, n_idx] = new_neutral_coord

        return bent_point_cloud

if __name__ == '__main__':
    from utils_temp import visualize_points
    point_cloud = np.load('/home/koondra/full_accumulated_dataset/0803/sample_103.npz')['data']
    bender = SingleRandomBend(60, 0.07)
    # visualize_points(point_cloud)
    print('vis')

    bend_point_cloud = bender(point_cloud)
    visualize_points(bend_point_cloud)


