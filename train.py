from torch import nn
from torch import optim
import torch
from tqdm import tqdm
import numpy as np
from utils.dataset import GenerateIterator
from myargs import args
import segmentation_models_pytorch as smp
import time


def train():

    continue_train = False

    def activation(x):
        x
    model = eval('smp.'+args.model_name)(
        args.encoder_name,
        encoder_weights='imagenet',
        classes=3,
        activation=activation,
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )

    lossfn = nn.CrossEntropyLoss(reduction='mean')

    iterator_train = GenerateIterator('./bdd100k/images/100k/train_resize/', './bdd100k/labels/train/', eval=False)
    iterator_val = GenerateIterator('./bdd100k/images/100k/val_resize/', './bdd100k/labels/val/', eval=True)

    start_epoch = 1

    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    if continue_train:
        pretrained_dict = torch.load('./bdd100k/model/blah.pt')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    for epoch in range(start_epoch, args.num_epoch + 1):

        losses_sum, n_total = 0, 0
        pbar = tqdm(iterator_train[:1000], disable=False)
        start = time.time()

        for images, label in pbar:
            images, gt,  = images.cuda(), label.cuda()

            pred = model(images)

            loss_cls = lossfn(pred, gt)

            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()

            losses_sum += loss_cls.item()
            n_total += 1
            pbar.set_description('Loss: {:.5f} '.format(losses_sum / n_total))

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()

                # calculate accuracy on validation set
                preds, gts = [], []
                n_total_val = 0
                val_loss = 0
                for images, label in tqdm(iterator_val[:300]):
                    images, gt = images.cuda(), label.cuda()

                    # cross entropy based classification
                    pred = model(images)

                    val_loss += lossfn(pred, gt)

                    pred = torch.softmax(pred, 1).cpu().data.numpy()
                    pred = np.argmax(pred, 1)

                    preds.extend(pred)
                    gts.extend(label.cpu().data.numpy())

                    n_total_val += 1

                preds = np.asarray(preds)
                gts = np.asarray(gts)

                score_cls_val = (np.mean(preds == gts)).astype(np.float)

            print('|| Ep {} || Secs {:.1f} || Loss {} || Val Score {} || Val Loss {} || \n'.format(
                    epoch,
                    time.time() - start,
                    losses_sum,
                    score_cls_val,
                    val_loss,
                ))

            model.train()

        if epoch % 1 == 0:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, './bdd100k/model/model_{}_{}.pt'.format(args.model_name, epoch))


if __name__ == "__main__":
    train()