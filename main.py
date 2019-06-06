import itertools
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn.init
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data as data
from skimage import io
from torch.autograd import Variable
from torchsummary import summary
from tqdm import tqdm_notebook as tqdm
from models.ternaus_net import unet11

import data_generator
import loss_functions
import metrics
from models.segnet import SegNet

# Parameters
WINDOW_SIZE = (256, 256)  # Patch size
STRIDE = 32  # Stride for testing
IN_CHANNELS = 3  # Number of input channels (e.g. RGB)
FOLDER = '/home/x/Dmitriev-semseg/dataset/'  # Replace with your "/path/to/the/ISPRS/dataset/folder/"
BATCH_SIZE = 10  # Number of samples in a mini-batch

LABELS = ["Buildings", "Background"]
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)  # Weights for class balancing
CACHE = True  # Store the dataset in-memory

# DATASET = 'Vaihingen'
# DATASET = 'Potsdam'
# DATASET = 'Tatarstan'
# DATASET = 'Zubchaninovka'
# DATASET = 'Tatarstan'
DATASET = 'Tatarstan'

MAIN_FOLDER = FOLDER + DATASET+'/'
DATA_FOLDER = MAIN_FOLDER + 'data/'
LABEL_FOLDER = MAIN_FOLDER + 'labels/'
ERODED_FOLDER = LABEL_FOLDER

TEST_FOLDER = 'test/'
IN_TEST_FOLDER = TEST_FOLDER + 'in_test_images_tatarstan/'
# IN_TEST_FOLDER = TEST_FOLDER +'in_test_images_samara/'
OUT_TEST_FOLDER = TEST_FOLDER + 'out/'

palette = {0:(255, 255, 255), #  Buildings
           1:(0, 0, 0)
           } #  Rest


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

def save_list(list, filename):
    with open(filename, "wb") as fp:
        pickle.dump(list, fp)

def load_list(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)

# net = ResNetSeg(N_CLASSES)
# net = FCDenseNet67(N_CLASSES)
net = unet11(IN_CHANNELS)
# net = RefineNet4Cascade((3, WINDOW_SIZE[0]), num_classes=N_CLASSES)

net.cuda()

summary(net, (3, 224, 224))


train_set = DataGenerator(lis, LABEL_FOLDER, WINDOW_SIZE, palette, cache=CACHE)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

optimizer = optim.Adam(net.parameters(), lr=1e-5)


def notest(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER+id), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER+id), dtype='uint8') for id in test_ids)
    eroded_labels = (data_generator.convert_from_color(io.imread(ERODED_FOLDER+id), palette) for id in test_ids)

    all_preds = []
    all_gts = []
    number_result = 0

    # Switch the network to inference mode
    net.eval()

    for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total,
                     leave=False)):
            # Display in progress results
            if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                _pred = np.argmax(pred, axis=-1)
                fig = plt.figure()
                fig.add_subplot(1, 3, 1)
                plt.imshow(np.asarray(255 * img, dtype='uint8'))
                fig.add_subplot(1, 3, 2)
                plt.imshow(data_generator.convert_to_color(_pred, palette))
                fig.add_subplot(1, 3, 3)
                plt.imshow(gt)
                clear_output()

                plt.show(block=False)
                plt.savefig('/home/x/PycharmProjects/segm/results/progress_results_{}.png'.format(i))
                plt.close()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        clear_output()
        fig = plt.figure()
        fig.add_subplot(1, 3, 1)
        plt.imshow(np.asarray(255 * img, dtype='uint8'))
        fig.add_subplot(1, 3, 2)
        plt.imshow(data_generator.convert_to_color(pred, palette))
        fig.add_subplot(1, 3, 3)
        plt.imshow(gt)
        plt.show(block=False)
        plt.savefig('/home/x/PycharmProjects/segm/results/result_{}.png'.format(number_result))
        plt.close()

        number_result += 1

        all_preds.append(pred)
        all_gts.append(gt_e)

        clear_output()
        metrics.get_values_of_all_metrics(pred.ravel(), gt_e.ravel(), LABELS)
        acc = metrics.get_values_of_all_metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel(), LABELS)
    if all:
        return acc, all_preds, all_gts
    else:
        return acc


def tst_new_image(net, net_2, test_ids, all=False, stride=WINDOW_SIZE[0]//2, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (1 / 255 * np.asarray(io.imread(IN_TEST_FOLDER + id), dtype='float32') for id in test_ids)
    # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER + id)[:,:,0:3], dtype='float32') for id in test_ids)

    all_preds = []

    net.eval()
    net_2.eval()

    for img, id in tqdm(zip(test_images, test_ids), total=len(test_ids), leave=False):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

        if not all:
            for i, coords in enumerate(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
                    # Display in progress results
                    if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
                        print(IN_TEST_FOLDER.format(id))
                        print('{:.2%}'.format(i / int(total)))

                    # Build the tensor
                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                    outs = net(image_patches)
                    outs = outs.data.cpu().numpy()

                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out

                    del (outs)
        else:
            img = img.transpose((2, 0, 1))
            img = img.reshape((1,img.shape[0], img.shape[1], img.shape[2]))
            img = Variable(torch.from_numpy(img).cuda(), volatile=True)

            pred = net(img).data.cpu().numpy()[0]
            pred = pred.transpose((1, 2, 0))

        pred = np.argmax(pred, axis=-1)

        clear_output()

        all_preds.append(pred)

        clear_output()

    return all_preds


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=5):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = loss_functions.CrossEntropy2d(output, target, weight=weights)
            loss.backward()
            optimizer.step()

            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data.item(), metrics.accuracy(pred, gt)))
                plt.plot(mean_losses[:iter_]) and plt.show(block=False)
                plt.savefig('results/mean_losses_epoch_{}_iter_{}.png'.format(e,iter_))
                plt.close()
                fig = plt.figure()
                fig.add_subplot(131)
                plt.imshow(rgb)
                plt.title('RGB')
                fig.add_subplot(132)
                plt.imshow(data_generator.convert_to_color(gt, palette))
                plt.title('Ground truth')
                fig.add_subplot(133)
                plt.title('Prediction')
                plt.imshow(data_generator.convert_to_color(pred, palette))
                plt.show(block=False)
                plt.savefig('results/image_epoch_{}_iter_{}.png'.format(e, iter_))
                plt.close()
            iter_ += 1

            del (data, target, loss)

        if e % save_epoch == 0:
            acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
            torch.save(net.state_dict(), 'checkpoints/Moscow/resnet50/resnet50_{}_{}'.format(e, acc))
    torch.save(net.state_dict(), 'checkpoints/Moscow/resnet50/resnet50')

net.cuda()

def seg_test(test_ids):
    net.load_state_dict(torch.load('checkpoints/Tatarstan/segnet/segnet_final'))

    test_ids = test_ids[0:20]

    all_preds = test_new_image(net, net, test_ids, all=False, stride=32)
    test_labels = (1 / 255 * np.asarray(io.imread(LABEL_FOLDER + id), dtype='float32') for id in test_ids)

    global_accuracy = 0

    data_images = (np.asarray(io.imread(DATA_FOLDER + id)) for id in test_ids)

    for pred, id, label, data_image in zip(all_preds, test_ids, test_labels, data_images):
        # pred = CRF.get_prediction(pred, data_image)
        img = data_generator.convert_to_color(pred, palette)

        accuracy = metrics.accuracy(pred, data_generator.convert_from_color(label, palette))
        print('---Accuracy for '+id+':', accuracy)
        global_accuracy += accuracy

        data = np.array(img)
        mask = np.all(data == [255,255,0], axis = -1)
        data[mask] = [255,255,255]
        mask = np.all(data == [255,0,0], axis = -1)
        data[mask] = [255,255,255]

        io.imsave(OUT_TEST_FOLDER +id, data)

    global_accuracy /= len(test_ids)
    print('Global accuracy=', global_accuracy)

seg_test(test_ids)
