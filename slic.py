import math
from matplotlib import pyplot as plt
from skimage import io, color
import numpy as np
from tqdm import trange


class Cluster:
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return f"{self.h},{self.w}:{self.l} {self.a} {self.b}"

    def __repr__(self):
        return self.__str__()


class SLICProcessor:
    @staticmethod
    def open_image(path):
        """
        Open an image and convert it to LAB color space.
        :param path: Path to the image.
        :return: LAB image as a 3D numpy array.
        """
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        """
        Convert LAB image back to RGB and save it.
        :param path: Path to save the image.
        :param lab_arr: LAB image array.
        """
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def __init__(self, filename, K, M):
        """
        Initialize the SLIC processor.
        :param filename: Path to the input image.
        :param K: Number of superpixels.
        :param M: Compactness factor.
        """
        self.K = K
        self.M = M

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w, self.data[h][w][0], self.data[h][w][1], self.data[h][w][2])

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height:
                    continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width:
                        continue
                    L, A, B = self.data[h][w]
                    Dc = math.sqrt((L - cluster.l) ** 2 + (A - cluster.a) ** 2 + (B - cluster.b) ** 2)
                    Ds = math.sqrt((h - cluster.h) ** 2 + (w - cluster.w) ** 2)
                    D = math.sqrt(Dc**2 + (Ds / self.S)**2) * self.M
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        for cluster in self.clusters:
            sum_h = sum_w = number = 0
            for p in cluster.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
            if number > 0:
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_array = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_array[p[0]][p[1]] = [cluster.l, cluster.a, cluster.b]
        self.save_lab_image(name, image_array)

    def compute_aggregation_matrix(self):
        num_superpixels = len(self.clusters)
        num_pixels = self.image_height * self.image_width
        aggregation_matrix = np.zeros((num_superpixels, num_pixels), dtype=np.float32)

        for idx, cluster in enumerate(self.clusters):
            for h, w in cluster.pixels:
                pixel_index = h * self.image_width + w
                aggregation_matrix[idx, pixel_index] = 1

        row_sums = aggregation_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        aggregation_matrix = aggregation_matrix / row_sums
        np.save("aggregation_matrix.npy", aggregation_matrix)
        print("Aggregation matrix saved as 'aggregation_matrix.npy'.")

    def compute_adjacency_matrix(self):
        adjacency_matrix = np.zeros((len(self.clusters), len(self.clusters)))

        cluster_map = np.full((self.image_height, self.image_width), -1, dtype=int)
        for idx, cluster in enumerate(self.clusters):
            for h, w in cluster.pixels:
                cluster_map[h, w] = idx

        for idx, cluster in enumerate(self.clusters):
            for h, w in cluster.pixels:
                for dh, dw in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nh, nw = h + dh, w + dw
                    if 0 <= nh < self.image_height and 0 <= nw < self.image_width:
                        neighbor_idx = cluster_map[nh, nw]
                        if neighbor_idx != -1 and neighbor_idx != idx:
                            adjacency_matrix[idx, neighbor_idx] = 1

        np.save("adjacency_matrix.npy", adjacency_matrix)
        print("Adjacency matrix saved as 'adjacency_matrix.npy'.")

    def iterate_10times(self):
        self.init_clusters()
        self.move_clusters()
        for i in trange(10):
            self.assignment()
            self.update_cluster()

        # Generate the final superpixel label map
        labels = np.zeros((self.image_height, self.image_width), dtype=int)
        for cluster in self.clusters:
            for (h, w) in cluster.pixels:
                labels[h, w] = cluster.no

        return labels
