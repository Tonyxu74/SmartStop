from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from utils.dataset import GenerateIterator_test
from myargs import args
from torchvision import transforms
import segmentation_models_pytorch as smp
import time

DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)


def visualize():

    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.encoder_name,
        encoder_weights='imagenet',
        classes=3,
        activation=activation,
    )

    iterator_test = GenerateIterator_test('../bdd100k/images/100k/test_resize')

    if torch.cuda.is_available():
        model = model.cuda()

    pretrained_dict = torch.load('../bdd100k/model/model_Unet_1.pt')['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    with torch.no_grad():
        model.eval()

        for image, paths in tqdm(iterator_test):
            if torch.cuda.is_available():
                image = image.cuda()

            pred = model(image)

            sftmax_pred = torch.softmax(pred, dim=1)
            pred_class = torch.argmax(sftmax_pred, dim=1).cpu().numpy()

            for image, path in zip(pred_class, paths):
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                for dim in range(3):
                    img[:, :, dim] = (image == dim) * 255
                img = Image.fromarray(img).convert('RGB')
                img.save(path)


def visualize_one(path):

    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.encoder_name,
        encoder_weights='imagenet',
        classes=3,
        activation=activation,
    )

    image = Image.open(path)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
    ])

    image = transform(image).unsqueeze(0)

    pretrained_dict = torch.load('../bdd100k/model/model_Unet_1.pt')['state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    with torch.no_grad():
        model.eval()
        start = time.time()
        pred = model(image)

        sftmax_pred = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(sftmax_pred, dim=1).cpu().numpy()[0]
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        for dim in range(3):
            img[:, :, dim] = (pred_class == dim) * 255
        img = Image.fromarray(img).convert('RGB')
        img.save(path.replace('.png', '_mask.png'))

    print('time is {}'.format(time.time() - start))


if __name__ == '__main__':
    visualize_one('../goals.png')
