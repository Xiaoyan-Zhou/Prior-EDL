import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class DatasetLoader(Dataset):

    def __init__(self, setname, args=None):
        DATASET_DIR = args.data_path
        print('DATASET_DIR', DATASET_DIR)

        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
        elif setname == 'val':
            THE_PATH = osp.join(DATASET_DIR, 'val')
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        data = []
        label = []


        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if setname == 'train':
            image_size = 90
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.1245, 0.1442, 0.1820), (0.1373, 0.1375, 0.1537))])

        else:
            image_size = 90
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1245, 0.1442, 0.1820), (0.1373, 0.1375, 0.1537))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class DatasetLoader_sample(Dataset):

    def __init__(self, setname, random_seed, args=None):
        DATASET_DIR = args.data_path

        # Set the path according to train, val and test
        if setname == 'train':
            THE_PATH = osp.join(DATASET_DIR, 'train')
            label_list = os.listdir(THE_PATH)
            print('label_list', label_list)
        elif setname == 'test':
            THE_PATH = osp.join(DATASET_DIR, 'test')
            label_list = os.listdir(THE_PATH)
        # The training set and the test set are not divided, and the remaining part of the training part is naturally used as the test set
        elif setname == 'all-train' or setname == 'all-test':
            THE_PATH = DATASET_DIR
            label_list = os.listdir(THE_PATH)
        elif setname == 'test_15':
            THE_PATH ='./data/MSTAR_REAL/fdf8-EOC/15'
            label_list = os.listdir(THE_PATH)
        elif setname == 'test_17':
            THE_PATH ='./data/MSTAR_REAL/fdf8-EOC/17'
            label_list = os.listdir(THE_PATH)
        elif setname == 'test_30':
            THE_PATH ='./data/MSTAR_REAL/fdf8-EOC/30'
            label_list = os.listdir(THE_PATH)
        elif setname == 'test_45':
            THE_PATH ='./data/MSTAR_REAL/fdf8-EOC/45'
            label_list = os.listdir(THE_PATH)
        # Test sets with out-of-distribution data
        elif setname == 'uncertainty_oodtest_17':
            THE_PATH ='./data/MSTAR_OOD/17'
            label_list = os.listdir(THE_PATH)
        # Test sets with out-of-distribution data
        elif setname == 'uncertainty_oodtest_15':
            THE_PATH ='./data/MSTAR_OOD/15'
            label_list = os.listdir(THE_PATH)
        else:
            raise ValueError('Unkown setname.')

        data = []
        label = []

        folders = [osp.join(THE_PATH, label) for label in label_list if os.path.isdir(osp.join(THE_PATH, label))]

        # Get the images' paths and labels
        for idx, this_folder in enumerate(folders):
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        #Sample from the dataset by n_ways, k-shots
        if setname in ['all-train', 'train', 'all-test']:
            random.seed(random_seed)
            self.sample_label_train = []
            self.sample_label_test = []
            self.sample_data_train = []
            self.sample_data_test = []
            self.n_cls = args.n_ways
            self.n_per = args.k_shots
            self.m_ind = []  # the data index of each class
            for i in range(max(self.label) + 1):
                ind = [k for k in range(len(self.label)) if self.label[k] == i]
                self.m_ind.append(ind)
            # random shuffle
            random.shuffle(self.m_ind)
            # sample num_class indexs,e.g. 5
            classes = self.m_ind[:self.n_cls]
            for c in classes:
                random.shuffle(c)
                pos_train = c[:self.n_per]
                pos_test = c[self.n_per:]
                for m in pos_train:
                    self.sample_label_train.append(self.label[m])
                    self.sample_data_train.append(self.data[m])
                if setname == 'all-test':
                    for m in pos_test:
                        self.sample_label_test.append(self.label[m])
                        self.sample_data_test.append(self.data[m])
        # Transformation
        self.setname = setname
        if setname == 'train' or setname == 'all-train':
            image_size = 90
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize((0.1245, 0.1442, 0.1820), (0.1373, 0.1375, 0.1537))])  # transform2

        else:
            image_size = 90
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1245, 0.1442, 0.1820), (0.1373, 0.1375, 0.1537))])  # transform2 mstar_transform

    def __len__(self):
        if self.setname in ['train', 'all-train']:
            return len(self.sample_data_train)
        elif self.setname == 'all-test':
            return len(self.sample_data_test)
        else:
            return len(self.data)

    def __getitem__(self, i):
        if self.setname in ['train', 'all-train']:
            path, label = self.sample_data_train[i], self.sample_label_train[i]
            image = self.transform(Image.open(path).convert('RGB'))
        elif self.setname == 'all-test':
            path, label = self.sample_data_test[i], self.sample_label_test[i]
            image = self.transform(Image.open(path).convert('RGB'))
        else:
            path, label = self.data[i], self.label[i]
            image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path
