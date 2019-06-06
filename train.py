import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

import data_generator
import loss_functions
import metrics
from models.ternaus_net import unet11

WINDOW_SIZE = (256, 256)  # Patch size
STRIDE = 32  # Stride for testing
BATCH_SIZE = 10  # Number of samples in a mini-batch

DATA_FOLDER = '/home/x/Dmitriev/dataset/'
DATASET = 'Google_Samara/'

DATA_FOLDER = DATA_FOLDER + DATASET

IMAGES_FOLDER = DATA_FOLDER + 'data/'
LABELS_FOLDER = DATA_FOLDER + 'labels/'

list_with_train_imgs_paths = [DATA_FOLDER+'train/'+'data/'+x for x in sorted(os.listdir(DATA_FOLDER+'train/'+'data/'))]
list_with_train_labels_paths =  [DATA_FOLDER+'train/'+'labels/'+x for x in sorted(os.listdir(DATA_FOLDER+'train/'+'labels/'))]
list_with_val_imgs_paths =  [DATA_FOLDER+'test/'+'data/'+x  for x in sorted(os.listdir(DATA_FOLDER+'test/'+'data/'))]
list_with_labels_paths =  [DATA_FOLDER+'test/'+'labels/'+x for x in sorted(os.listdir(DATA_FOLDER+'test/'+'labels/'))]

train_set = data_generator.DataGenerator(list_with_train_imgs_paths, list_with_train_labels_paths,
                                  cache=True, augmentation=True)
train_set = DataLoader(train_set, batch_size=BATCH_SIZE,shuffle=True)

test_set = data_generator.DataGenerator(list_with_val_imgs_paths, list_with_labels_paths,
                                  cache=True, augmentation=False)
test_set = DataLoader(test_set, batch_size=BATCH_SIZE)


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c


def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


net = unet11()

net.cuda()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)


def train(net, optimizer, epochs, scheduler=None, save_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    # weights = weights.cuda()

    iter_ = 0

    prev_test_accuracy = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()

        for batch_idx, (data, target) in enumerate(train_set):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()

            output = net(data)

            loss = loss_functions.bce_dice(output, target)

            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, epochs, batch_idx, len(train_set),
                100. * batch_idx / len(train_set), loss.item(), ))

            if iter_ % 100 == 0:
                test_loss, test_iou, acc = get_loss_and_acc_test_set(net, test_set=test_set)


                if prev_test_accuracy < test_iou:
                    torch.save(net.state_dict(),
                               'checkpoints/Google_Samara/vgg_unet11/vgg_unet11_{}_{}'.format(e, test_iou))

                    prev_test_accuracy = test_iou

                print('Test (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {} \tTest_iou:{}'.format(
                    e, epochs, batch_idx, len(train_set),
                    100. * batch_idx / len(train_set), test_loss, acc, test_iou))
            iter_ += 1

            del (data, target, loss)

def get_loss_and_acc_test_set(net, test_set):
    loss = 0
    iou = 0
    batch_idx = 0
    acc = 0

    for batch_idx, (data, target) in enumerate(test_set):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = net(data)

        loss += loss_functions.bce_dice(output, target).item()

        prediction_mask = torch.sigmoid(output)>0.5

        iou += metrics.iou(prediction_mask.data.cpu().numpy(), target.data.cpu().numpy())
        acc += metrics.accuracy(prediction_mask.data.cpu().numpy()[:,0,:,:], target.data.cpu().numpy())

        if batch_idx == 0:
            # figures = {'image' : data, 'mask':output.data.cpu().numpy()[0]}
            fig = plt.figure()
            fig.add_subplot(1, 3, 1)
            plt.imshow(data.data.cpu().numpy()[0].transpose((1,2,0)))
            fig.add_subplot(1, 3, 2)
            plt.imshow(prediction_mask.data.cpu().numpy()[0,0])
            fig.add_subplot(1, 3, 3)
            plt.imshow(target.data.cpu().numpy()[0])
            plt.show(block=False)
            plt.savefig('images/result.png')
            plt.close(fig)

        del (data, target)

    return loss/batch_idx, iou/batch_idx, acc/batch_idx

train(net, optimizer, 1000)