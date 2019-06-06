import numpy as np
import torch.optim.lr_scheduler
from skimage import io
from torch.autograd import Variable
from torch.nn.functional import softmax
from models.ternaus_net import unet11
import cv2

WINDOW_SIZE = (256, 256)


def sliding_window(shape, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, shape[0], step):
        if x + window_size[0] > shape[0]:
            x = shape[0] - window_size[0]
        for y in range(0, shape[1], step):
            if y + window_size[1] > shape[1]:
                y = shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

img = io.imread('500.bmp')
img = 1 / 255 * np.asarray(img.transpose((2, 0, 1)), dtype='float32')
img = np.expand_dims(img,axis=0)


net = unet11(pretrained=False)
# net2 = unet11()

net.cuda()
# net2.cuda()

net.eval()
net.load_state_dict(torch.load('checkpoints/Google_Samara/vgg_unet11/vgg_unet11_3_74.12627725987213'))
# net2.load_state_dict(torch.load('checkpoints/Google_Samara/vgg_unet11/vgg_unet11_7_97.40455034044054'))

result = np.zeros((img.shape[2:]))

for y,x,height, witdh in sliding_window(img.shape[2:], step=WINDOW_SIZE[0], window_size=WINDOW_SIZE):
    data = img[:,:,y:y+height, x:x+witdh]
    data = torch.from_numpy(data)
    data = Variable(data.cuda())

    out = torch.sigmoid(net(data))>0.5
    # out_2 = net2(data)
    out = (out.data.cpu().numpy()[0,0])

    result[y:y+height, x:x+height] = out

cv2.imwrite('res_1.png', 255*result)
