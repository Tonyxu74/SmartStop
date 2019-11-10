# first make everything like 2-3 times smaller, then:
# this needs to look at the JSON and then build an image with the same dimensions with the bounding box outlined
# i guess do people second because we want people to show up in front of cars
# do what Ozan did like 0 for NULL 1 for car 2 for person
# do ML stuff

# make sure that we use the JSON to pick images, like imagelist = "ass/{}.".format(json[0]['name']) for image and label
# because i think for some images i don't ahve labels so this won't throw errors
import os
from PIL import Image
import json
import numpy as np
import tqdm
import torchvision.transforms.functional as func
import torch

UNREGULARLIZED_IMAGE_DIMS = (1280, 720)


def find_file(root_dir, endswith):
    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(endswith):
                all_files.append(os.path.join(path, file))

    return all_files


def rescale(resize=(398, 224)):
    input_train_list = find_file('../bdd100k/images/100k/train', '.jpg')
    input_val_list = find_file('../bdd100k/images/100k/val', '.jpg')
    input_test_list = find_file('../bdd100k/images/100k/test', '.jpg')

    total_list = input_train_list + input_test_list + input_val_list

    for image in tqdm.tqdm(total_list):
        img = Image.open(image)
        img = img.resize(size=resize, resample=Image.NEAREST)
        image = image.replace('train', 'train_resize').replace('val', 'val_resize').replace('test', 'test_resize')
        img.save(image)


def generate_mask(resize=(398, 224), show_image=False):
    image_dims = resize
    for type in ['train', 'val']:
        with open('../bdd100k/labels/bdd100k_labels_images_{}.json'.format(type), 'r') as jsonfil:
            img_data = jsonfil.read()

        img_data = json.loads(img_data)
        for imginfo in tqdm.tqdm(img_data):
            mask = np.zeros(shape=(image_dims[1], image_dims[0]), dtype=np.uint8)
            img_name = imginfo['name']
            carboxlist = []
            peopleboxlist = []

            for objects in imginfo['labels']:
                if objects['category'] == 'car' or objects['category'] == 'truck':
                    carboxlist.append(objects['box2d'])
                elif objects['category'] == 'person':
                    peopleboxlist.append(objects['box2d'])

            for car in carboxlist:
                x1 = int(car['x1'] / 1280 * resize[0])
                x2 = int(car['x2'] / 1280 * resize[0])
                y1 = int(car['y1'] / 720 * resize[1])
                y2 = int(car['y2'] / 720 * resize[1])

                # x refers to column number, y refers to row number
                mask[y1:y2, x1:x2] = 1

            for people in peopleboxlist:
                x1 = int(people['x1'] / 1280 * resize[0])
                x2 = int(people['x2'] / 1280 * resize[0])
                y1 = int(people['y1'] / 720 * resize[1])
                y2 = int(people['y2'] / 720 * resize[1])

                # x refers to column number, y refers to row number
                # we deal with people second to overlap car bounding boxes
                mask[y1:y2, x1:x2] = 2

            mask_img = Image.fromarray(mask).convert('L')
            mask_img.save('../bdd100k/labels/{}/{}'.format(type, img_name).replace('.jpg', '.png'))
            # np.save('../bdd100k/labels/{}/{}'.format(type, img_name).replace('.jpg', '.npy'), mask)
            if show_image:
                mask_vis = np.zeros((image_dims[1], image_dims[0], 3), dtype=np.uint8)
                for dim in range(3):
                    mask_vis[:, :, dim] = (mask == dim)*255
                mask_img = Image.fromarray(mask_vis).convert('RGB')
                mask_img.save('../bdd100k/labels/{}/{}'.format(type, img_name).replace('.jpg', '.png'))


if __name__ == '__main__':
    rescale()
    generate_mask(show_image=False)
    # list = find_file('../bdd100k/labels/{}/'.format('train'), '.png')
    # for pth in list:
    #     mask_img = Image.open(pth)
    #     test = func.crop(mask_img, 0, 0, 224, 224)
    #     test = np.asarray(test, dtype=np.uint8)
    #     test = torch.from_numpy(test).long()
    #     if test.max() > 2:
    #         print('OOP')