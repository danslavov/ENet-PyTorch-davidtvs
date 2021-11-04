import torch

batch_count = 1
red_value = 60
green_value = 40
blue_value = 222

# a = torch.full((1, 3, 5, 10), -17)
# b = a.size()
#
# a[0][0][2][:-2] = 2
# condition = (a[0][0][2] == 2)
# a[0][0][2][condition] = -1

channel_red = torch.tensor([[
    [0, 0, 0, 0, 0],
    [0, 60, 0, 0, 0],
    [0, 60, 60, 0, 0],
    [0, 60, 0, 0, 0]
]])

channel_green = torch.tensor([[
    [0, 0, 0, 0, 0],
    [0, 40, 0, 0, 0],
    [0, 40, 40, 0, 0],
    [0, 40, 0, 0, 0]
]])

channel_blue = torch.tensor([[
    [0, 0, 0, 0, 0],
    [0, 222, 0, 0, 0],
    [0, 222, 222, 0, 0],
    [0, 222, 0, 0, 0]
]])

image = torch.cat((channel_red, channel_green, channel_blue))
# rnd = torch.randint(256, image.size())

tensor = torch.zeros(batch_count, image.size()[0], image.size()[1], image.size()[2])
tensor[0] = image

cond_red = image[0] == red_value
cond_green = image[1] == green_value
cond_blue = image[2] == blue_value

cond_image = torch.zeros(3, 4, 5, dtype=torch.bool)
cond_image[0] = cond_red
cond_image[1] = cond_green
cond_image[2] = cond_blue

tensor[0][cond_image] = -1

print(tensor)