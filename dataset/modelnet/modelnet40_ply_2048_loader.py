"""Modified from DeepGCN and DGCNN
Reference: https://github.com/lightaime/deep_gcns_torch/tree/master/examples/classification
"""
import os
import glob
import h5py
import numpy as np
import pickle
import logging
import ssl
import urllib
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import extract_archive, check_integrity
from ..build import DATASETS
import pandas as pd
from sklearn import preprocessing
from pyntcloud import PyntCloud



def download_and_extract_archive(url, path, md5=None):
    # Works when the SSL certificate is expired for the link
    path = Path(path)
    extract_path = path
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / Path(url).name
        if not file_path.exists() or not check_integrity(file_path, md5):
            print(f'{file_path} not found or corrupted')
            print(f'downloading from {url}')
            context = ssl.SSLContext()
            with urllib.request.urlopen(url, context=context) as response:
                with tqdm(total=response.length) as pbar:
                    with open(file_path, 'wb') as file:
                        chunk_size = 1024
                        chunks = iter(lambda: response.read(chunk_size), '')
                        for chunk in chunks:
                            if not chunk:
                                break
                            pbar.update(chunk_size)
                            file.write(chunk)
            extract_archive(str(file_path), str(extract_path))
    return extract_path


def load_data(data_dir, split, url):
    download_and_extract_archive(url, data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % split)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0).squeeze(-1)
    return all_data, all_label


@DATASETS.register_module()
class ModelNet40Ply2048(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.
    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    dir_name = 'modelnet40_ply_hdf5_2048'
    md5 = 'c9ab8e6dfb16f67afdab25e155c79e59'
    url = f'https://shapenet.cs.stanford.edu/media/{dir_name}.zip'
    classes = ['airplane',
               'bathtub',
               'bed',
               'bench',
               'bookshelf',
               'bottle',
               'bowl',
               'car',
               'chair',
               'cone',
               'cup',
               'curtain',
               'desk',
               'door',
               'dresser',
               'flower_pot',
               'glass_box',
               'guitar',
               'keyboard',
               'lamp',
               'laptop',
               'mantel',
               'monitor',
               'night_stand',
               'person',
               'piano',
               'plant',
               'radio',
               'range_hood',
               'sink',
               'sofa',
               'stairs',
               'stool',
               'table',
               'tent',
               'toilet',
               'tv_stand',
               'vase',
               'wardrobe',
               'xbox']

    def __init__(self,
                 num_points=1024,
                 data_dir="/mnt/nvme0n1/Datasets/MN_new/",
                 split='train',
                 transform=None
                 ):
        data_dir = os.path.join(
            os.getcwd(), data_dir) if data_dir.startswith('.') else data_dir
        self.split = 'train' if split.lower() == 'train' else 'test'  # val = test
        self.data, self.label = load_data(data_dir, self.split, self.url)
        self.num_points = num_points
        logging.info(f'==> sucessfully loaded {self.split} data')
        self.transform = transform

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        if self.split == 'train':
            np.random.shuffle(pointcloud)
        data = {'pos': pointcloud,
                'y': label
                }
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' in data.keys():
            data['x'] = torch.cat((data['pos'], data['heights']), dim=1)
        else:
            data['x'] = data['pos']
        return data

    def __len__(self):
        return self.data.shape[0]

    @property
    def num_classes(self):
        return np.max(self.label) + 1

    """ for visulalization
    from openpoints.dataset import vis_multi_points
    import copy
    old_points = copy.deepcopy(data['pos'])
    if self.transform is not None:
        data = self.transform(data)
    new_points = copy.deepcopy(data['pos'])
    vis_multi_points([old_points, new_points.numpy()])
    End of visulization """


@DATASETS.register_module()
class IntrA(Dataset):
    """
    This is the data loader for the IntrA dataset.
    IntrA contains annotated and generated vessel data.
    Args:
        data_dir: Path to the directory containing point cloud data.
        num_points: Number of points per sample (default: 2048).
        split: Data split mode (train/test/interpret/non_annotated/all).
        cls_state: Whether to load data with classification labels (default: True).
        transform: Data augmentation or transformation to apply.
        choice: Fold choice for cross-validation.
    """
    classes = ['healthy', 'aneurysm']
    def __init__(self,
                 data_dir='/mnt/nvme0n1/Datasets/IntrA/',
                 num_points=1024,
                 split="train",
                 cls_state=True,
                 transform=None,
                 choice=0):
        assert data_dir is not None, "data_dir cannot be None"
        self.num_points = num_points
        self.cls_state = cls_state
        self.split = split
        self.transform = transform
        self.datapath = []
        self.labels = {}

        fold_csv = pd.read_csv(os.path.join(data_dir, f"folds/fold_{choice}.csv"))
        print("training intra dataset")

        # Prepare labels and datapaths
        if self.cls_state:
            self.labels[0] = glob.glob(os.path.join(data_dir, "generated/vessel/ad/*.ad"))
            self.labels[1] = glob.glob(os.path.join(data_dir, "generated/aneurysm/ad/*.ad")) + \
                             glob.glob(os.path.join(data_dir, "annotated/ad/*.ad"))

            train_set = [os.path.join(data_dir, i.split("IntrA/")[-1])
                         for i in fold_csv[fold_csv["Split"] == "train"]["Path"].tolist()]
            val_set = [os.path.join(data_dir, i.split("IntrA/")[-1])
                       for i in fold_csv[fold_csv["Split"] == "validation"]["Path"].tolist()]
            train_set += val_set
            test_set = [os.path.join(data_dir, i.split("IntrA/")[-1])
                        for i in fold_csv[fold_csv["Split"] == "test"]["Path"].tolist()]
            annotated = glob.glob(os.path.join(data_dir, "annotated/ad/*.ad"))
        else:
            annotated = glob.glob(os.path.join(data_dir, "annotated/ad/*.ad"))
            train_set = [os.path.join(data_dir, i.split("IntrA/")[-1])
                         for i in fold_csv[fold_csv["Split"] == "train"]["Path"].tolist()
                         if os.path.join(data_dir, i.split("IntrA/")[-1]) in annotated]
            val_set = [os.path.join(data_dir, i.split("IntrA/")[-1])
                       for i in fold_csv[fold_csv["Split"] == "validation"]["Path"].tolist()
                       if os.path.join(data_dir, i.split("IntrA/")[-1]) in annotated]
            train_set += val_set
            test_set = [os.path.join(data_dir, i.split("IntrA/")[-1])
                        for i in fold_csv[fold_csv["Split"] == "test"]["Path"].tolist()
                        if os.path.join(data_dir, i.split("IntrA/")[-1]) in annotated]
            non_annotated = [os.path.join(data_dir, i.split("IntrA/")[-1])
                             for i in fold_csv["Path"].tolist()
                             if os.path.join(data_dir, i.split("IntrA/")[-1]) not in annotated]

        # Assign split-specific datapaths
        if split == "train":
            self.datapath = train_set
        elif split == "test":
            self.datapath = test_set
        elif split == "all":
            self.datapath = train_set + test_set
        elif split == "interpret":
            self.datapath = annotated
        elif split == "non_annotated":
            self.datapath = non_annotated
        else:
            raise ValueError(f"Invalid split mode: {split}")
        self.split = split

    def __len__(self):
        return len(self.datapath)

    @property
    def num_classes(self):
        return 2

    def __getitem__(self, idx):
        curr_file = self.datapath[idx]
        point_set = np.loadtxt(curr_file)[:, :-1].astype(np.float32)
        seg = np.loadtxt(curr_file)[:, -1].astype(np.int64)
        seg[np.where(seg == 2)] = 1  # Relabel boundary lines as aneurysm (label 1)

        # Random choice to sample points
        if point_set.shape[0] < self.num_points:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=False)
        point_set = point_set[choice, :]
        seg = seg[choice]


        # Normalize to unit ball
        point_set[:, :3] -= np.mean(point_set[:, :3], axis=0)
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)))
        point_set[:, :3] /= dist


        if self.split == 'train':
            np.random.shuffle(point_set)

        if self.cls_state:
            if self.split == 'interpret':
                data = {'pos': point_set, 'seg': seg}
                data['x'] = data['pos']
                return data

            cls = None
            if self.cls_state:
                if curr_file in self.labels[0]:
                    cls = torch.from_numpy(np.array([0]).astype(np.int64))

                elif curr_file in self.labels[1]:
                    cls = torch.from_numpy(np.array([1]).astype(np.int64))
                else:
                    print("Error found!!!")
                    exit(-1)
            data = {'pos': point_set, 'y': cls}
            # if self.transform is not None:
            #     data = self.transform(data)

            data['x'] = data['pos']
            return data
        else:
            return {'pos': point_set, 'seg': seg}


@DATASETS.register_module()
class RBC(Dataset):
    """
    This is the data loader for the RBC dataset.
    RBC dataset contains annotated point clouds for classification and segmentation.
    Args:
        data_dir: Path to the directory containing point cloud data.
        num_points: Number of points per sample (default: 2048).
        split: Data split mode (train/test/all).
        cls_state: Whether to load data with classification labels (default: True).
        transform: Data augmentation or transformation to apply.
        choice: Fold choice for cross-validation.
    """
    classes = ['stomatocyte',
               'cell cluster',
               'keratocyte',
               'knizocyte',
               'multilobate cell',
               'acanthocyte',
               'spherocyte',
               'discocyte',
                'echinocyte'
               ]
    def __init__(self,
                 data_dir=None,
                 num_points=1024,
                 split="train",
                 cls_state=True,
                 transform=None,
                 choice=2):
        assert data_dir is not None, "data_dir cannot be None"
        self.num_points = num_points
        self.cls_state = cls_state
        self.split = split
        self.transform = transform
        self.datapath = []

        fold_csv = pd.read_csv(os.path.join(data_dir, f"folds_9/fold_{choice}.csv"))
        print(f"We are training fold {fold_csv}")

        train_set = [
            os.path.join(data_dir, "ad", i.split("ad/")[-1])
            for i in fold_csv[fold_csv["Split"] == "train"]["Path"].tolist()
        ]
        train_labels = fold_csv[fold_csv["Split"] == "train"]["Label"].tolist()

        val_set = [
            os.path.join(data_dir, "ad", i.split("ad/")[-1])
            for i in fold_csv[fold_csv["Split"] == "validation"]["Path"].tolist()
        ]
        val_labels = fold_csv[fold_csv["Split"] == "validation"]["Label"].tolist()

        train_set += val_set
        train_labels += val_labels

        test_set = [
            os.path.join(data_dir, "ad", i.split("ad/")[-1])
            for i in fold_csv[fold_csv["Split"] == "test"]["Path"].tolist()
        ]
        test_labels = fold_csv[fold_csv["Split"] == "test"]["Label"].tolist()

        if split == "train":
            self.datapath = (train_set, train_labels)
        elif split == "test":
            self.datapath = (test_set, test_labels)
        elif split == "all":
            self.datapath = (train_set + test_set, train_labels + test_labels)
        else:
            raise ValueError(f"Invalid split mode: {split}")

    def __len__(self):
        return len(self.datapath[0])

    def __getitem__(self, idx):
        curr_file = self.datapath[0][idx]
        cls = self.datapath[1][idx]

        point_set = np.loadtxt(curr_file)[:, :-1].astype(np.float32)
        seg = np.loadtxt(curr_file)[:, -1].astype(np.int64)
        seg[np.where(seg == 2)] = 1  # Relabel boundary lines as aneurysm (label 1)

        # Random choice to sample points
        if point_set.shape[0] < self.num_points:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=True)
        else:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=False)
        point_set = point_set[choice, :]
        seg = seg[choice]

        # Normalize to unit ball
        point_set[:, :3] -= np.mean(point_set[:, :3], axis=0)
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)))
        point_set[:, :3] /= dist

        data = {'pos': point_set, 'y': cls}
        # if self.transform is not None:
        #     data = self.transform(data)

        data['x'] = data['pos']

        if self.cls_state:
            return data
        else:
            return data


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

@DATASETS.register_module()
class Drugs(Dataset):
    """
    This is the data loader for the RBC dataset.
    RBC dataset contains annotated point clouds for classification and segmentation.
    Args:
        data_dir: Path to the directory containing point cloud data.
        num_points: Number of points per sample (default: 2048).
        split: Data split mode (train/test/all).
        cls_state: Whether to load data with classification labels (default: True).
        transform: Data augmentation or transformation to apply.
        choice: Fold choice for cross-validation.
    """
    classes = ['Binimetinib',
               'Blebbistatin',
               'CK666',
               'DMSO',
               'H1152',
               'MK1775',
               'Nocodazole',
               'No Treatment',
               'Palbocyclib',
               'PF228'
               ]
    def __init__(self,
                 data_dir=None,
                 num_points=1024,
                 split="train",
                 cls_state=True,
                 transform=None,
                 choice=0,
                 inference=False):
        assert data_dir is not None, "data_dir cannot be None"
        self.num_points = num_points
        self.cls_state = cls_state
        self.split = split
        self.transform = transform
        self.datapath = []
        self.partition = split
        self.data_dir = data_dir
        self.inference = inference


        self.annot_df = pd.read_csv(os.path.join(data_dir, f"new_drugs_convext_hull_distal/fold0.csv"))

        if self.partition != "all":
            self.new_df = self.annot_df[(self.annot_df.Splits == self.partition) &
                                        ((self.annot_df.Treatment == "No Treatment") |
                                         (self.annot_df.Treatment == "Nocodazole") |
                                         (self.annot_df.Treatment == "Blebbistatin"))
                                         # (self.annot_df.Treatment == "Binimetinib"))
            ].reset_index(drop=True)
        else:
            self.new_df = self.annot_df.reset_index(drop=True)

        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.new_df["Treatment"].values)

    def __len__(self):
        return len(self.new_df)

    def __getitem__(self, idx):
        # read the image
        treatment = self.new_df.loc[idx, "Treatment"]
        # class_id = self.new_df.loc[idx, "Class"]
        plate_num = "Plate" + str(self.new_df.loc[idx, "PlateNumber"])
        # Always load 4096 points
        num_str = '_4096'

        component_path = "stacked_pointcloud" + num_str

        img_path = os.path.join(
            self.data_dir,
            plate_num,
            component_path,
            treatment,
            self.new_df.loc[idx, "serialNumber"],
        )
        image = PyntCloud.from_file(img_path + ".ply")
        point_set = image.points.values

        if point_set.shape[0] < self.num_points:
            choice = np.random.choice(point_set.shape[0], self.num_points, replace=True)
            point_set = point_set[choice, :]
        else:
            point_set = farthest_point_sample(point_set, self.num_points)


        # Normalize to unit ball
        point_set[:, :3] -= np.mean(point_set[:, :3], axis=0)
        dist = np.max(np.sqrt(np.sum(point_set[:, :3] ** 2, axis=1)))
        point_set[:, :3] /= dist

        cls = torch.squeeze(torch.tensor(self.le.transform([treatment])))

        data = {'pos': point_set, 'y': cls}
        # if self.transform is not None:
        #     data = self.transform(data)

        data['x'] = data['pos']
        if self.inference:
            data['serial'] = self.new_df.loc[idx, "serialNumber"]
            data['Treatment'] = treatment
        # data['serial'] = self.new_df.loc[idx, "serialNumber"]
        # data['Treatment'] = treatment

        if self.cls_state:
            return data
        else:
            return data

