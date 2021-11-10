from collections import OrderedDict
import skimage.io as io
import torch
from torchvision import transforms
from PIL import Image

BATCH_SIZE = 1  # number of images in the batch
IMAGE_NUMBER = 0  # successive number of the image from the batch
CLASS_COUNT = 12  # total number of classes
CLASS_NUMBER = 0
HIGH_VALUE = 20
LOW_VALUE = -20
MASK_CHANNELS = 3  # 3 for RGB
CLASS_NAME = 'sky'
#
COLOR_ENCODING = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])
#
mask_dir = 'C:/Users/User/PycharmProjects/ENet-PyTorch-davidtvs/data/CamVid/train_labels'
mask_name = 'Seq05VD_f05100_L.png'
tmp_dir = 'C:/Users/User/Desktop/tmp/input_label_output'
file_ext = 'png'

# a = torch.full((1, 3, 5, 10), -17)
# b = a.size()
#
# a[0][0][2][:-2] = 2
# condition = (a[0][0][2] == 2)
# a[0][0][2][condition] = -1

# channel_red = torch.tensor([[
#     [128, 128, 0, 0, 0],
#     [128, 128, 128, 128, 128],
#     [0, 0, 0, 192, 192],
#     [255, 0, 0, 192, 192]
# ]])
#
# channel_green = torch.tensor([[
#     [128, 128, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 192, 192],
#     [69, 0, 0, 192, 192]
# ]])
#
# channel_blue = torch.tensor([[
#     [128, 128, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 128, 128],
#     [0, 0, 0, 128, 128]
# ]])
#
# mask = torch.cat((channel_red, channel_green, channel_blue))
