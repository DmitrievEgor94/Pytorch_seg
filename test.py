def notst(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER + id), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER + id), dtype='uint8') for id in test_ids)
    eroded_labels = (data_generator.convert_from_color(io.imread(ERODED_FOLDER + id), palette) for id in test_ids)

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


def tet_new_image(net, net_2, test_ids, all=False, stride=WINDOW_SIZE[0] // 2, batch_size=BATCH_SIZE,
                   window_size=WINDOW_SIZE):
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
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
            img = Variable(torch.from_numpy(img).cuda(), volatile=True)

            pred = net(img).data.cpu().numpy()[0]
            pred = pred.transpose((1, 2, 0))

        pred = np.argmax(pred, axis=-1)

        clear_output()

        all_preds.append(pred)

        clear_output()

    return all_preds

def seg_test(test_ids):
    net.load_state_dict(torch.load('checkpoints/Tatarstan/segnet/segnet_final'))

    test_ids = test_ids[0:20]

    all_preds = test_new_image(net, net, test_ids, all=False, stride=32)
    test_labels = (1 / 255 * np.asarray(io.imread(LABEL_FOLDER + id), dtype='float32') for id in test_ids)

    global_accuracy = 0

    data_images = (np.asarray(io.imread(DATA_FOLDER + id)) for id in test_ids)

    i = 0
    for pred, id, label, data_image in zip(all_preds, test_ids, test_labels, data_images):
        # pred = CRF.get_prediction(pred, data_image)
        img = data_generator.convert_to_color(pred, palette)

        accuracy = metrics.accuracy(pred, data_generator.convert_from_color(label, palette))
        print('---Accuracy for ' + id + ':', accuracy)
        global_accuracy += accuracy

        data = np.array(img)
        mask = np.all(data == [255, 255, 0], axis=-1)
        data[mask] = [255, 255, 255]
        mask = np.all(data == [255, 0, 0], axis=-1)
        data[mask] = [255, 255, 255]

        io.imsave(OUT_TEST_FOLDER + id, data)

    global_accuracy /= len(test_ids)
    print('Global accuracy=', global_accuracy)


seg_test(test_ids)
