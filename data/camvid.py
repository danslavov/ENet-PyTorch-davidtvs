import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils
from args import get_arguments

# Get the arguments
args = get_arguments()


def get_color_encoding():
    return CamVid.color_encoding


class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """
    # Training dataset root folders
    train_folder = 'train'
    train_lbl_folder = 'train_labels'

    # Validation dataset root folders
    val_folder = 'val'
    val_lbl_folder = 'val_labels'

    # Test dataset root folders
    test_folder = 'test'
    test_lbl_folder = 'test_labels'

    # Images extension
    img_extension = args.input_image_extension

    # Default encoding for pixel value, class name, and class color
    # INFO: Dict ordering is exactly the same as channels order in the output heatmaps,
    # i.e. channel 0 represents sky, channel 1 represents building, etc.
    # Some classes in the dataset don't have the same encoding as here,
    # (e.g. pavement is 0, 0, 192) and therefore they return mean IoU = 0
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),       # 0
        ('building', (128, 0, 0)),      # 1
        ('pole', (192, 192, 128)),      # 2
        ('road_marking', (255, 69, 0)), # gets deleted during loading
        ('road', (128, 64, 128)),       # 3
        ('pavement', (60, 40, 222)),    # 4
        ('tree', (128, 128, 0)),        # 5
        ('sign_symbol', (192, 128, 128)),# 6
        ('fence', (64, 64, 128)),       # 7
        ('car', (64, 0, 128)),          # 8
        ('pedestrian', (64, 64, 0)),    # 9
        ('bicyclist', (0, 128, 192)),   # 10
        ('unlabeled', (0, 0, 0))        # 11
    ])

    def __init__(self,
                 root_dir,
                 mode='train',
                 transform=None,
                 label_transform=None,
                 loader=utils.pil_loader):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.train_data = utils.get_files(
                os.path.join(root_dir, self.train_folder),
                extension_filter=self.img_extension)

            self.train_labels = utils.get_files(
                os.path.join(root_dir, self.train_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.val_data = utils.get_files(
                os.path.join(root_dir, self.val_folder),
                extension_filter=self.img_extension)

            self.val_labels = utils.get_files(
                os.path.join(root_dir, self.val_lbl_folder),
                extension_filter=self.img_extension)
        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.test_data = utils.get_files(
                os.path.join(root_dir, self.test_folder),
                extension_filter=self.img_extension)

            self.test_labels = utils.get_files(
                os.path.join(root_dir, self.test_lbl_folder),
                extension_filter=self.img_extension)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[
                index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[
                index]
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[
                index]
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transform is not None:
            img = self.transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == 'train':
            return len(self.train_data)
        elif self.mode.lower() == 'val':
            return len(self.val_data)
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")
