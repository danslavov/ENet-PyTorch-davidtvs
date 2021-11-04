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


def rgb_to_class_channels(
        color_encoding,
        batch_size, image_number,
        class_count,
        mask_channels=3, high_value=20):
    """ Converts RGB ground truth mask to multichannel tensor in order to compare it with the model output. """

    mask = Image.open(mask_dir + '/' + mask_name)
    convert_tensor = transforms.Compose([transforms.ToTensor(), lambda value: value*255])
    mask = convert_tensor(mask)
    x = mask.size()[1]
    y = mask.size()[2]

    multichannel_tensor = torch.full((batch_size, class_count, x, y), LOW_VALUE)  # the same size as the model output

    for class_number in range(4):
        color = list(color_encoding.items())[class_number][1]
        red_value = color[0]
        green_value = color[1]
        blue_value = color[2]

        cond_red = mask[0] == red_value
        cond_green = mask[1] == green_value
        cond_blue = mask[2] == blue_value

        cond_image = torch.zeros(mask_channels, x, y, dtype=torch.bool)
        cond_image[0] = cond_red
        cond_image[1] = cond_green
        cond_image[2] = cond_blue

        cond_image_1_channel = torch.logical_and(torch.logical_and(cond_image[0], cond_image[1]), cond_image[2])

        multichannel_tensor[image_number][class_number][cond_image_1_channel] = high_value

    return multichannel_tensor


tensor = rgb_to_class_channels(COLOR_ENCODING, BATCH_SIZE, IMAGE_NUMBER, CLASS_COUNT)

# Save each channel as separate image
with torch.no_grad():
    tensor.cpu().numpy()
    for i in range(4):
        print(i)
        single_map = tensor[0][i]
        min_value = torch.min(single_map)
        max_value = torch.max(single_map)
        print('min {}'.format(min_value))
        print('max {}'.format(max_value))
        print('---------------')
        io.imsave('{}/outp_{}.{}'.format(tmp_dir, i, file_ext), single_map)
