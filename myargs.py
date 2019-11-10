import argparse

parser = argparse.ArgumentParser()

######################## Model parameters ########################

parser.add_argument('--model_name', default='Unet',
                    help='pretrained model name')
parser.add_argument('--encoder_name', default='resnet18',
                    help='encoder name')
parser.add_argument('--classes', default=3, type=int,
                    help='# of classes')

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--num_epoch', default=50, type=int,
                    help='epochs to train for')

parser.add_argument('--batch_size', default=8, type=int,
                    help='input batch size')

######################## Image properties (size) ########################

parser.add_argument('--patch_width', default=398, type=int,
                    help='patch size width')
parser.add_argument('--patch_height', default=224, type=int,
                    help='patch size height')


args = parser.parse_args()