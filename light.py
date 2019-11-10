from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
import segmentation_models_pytorch as smp
from myargs import args
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import time
import socket  # Import socket module

# cd /d E:\ML Projects\Cars\SmartStop\Server
# python ServerScriptPCRecieve.py
# RECALL CHANGE PUBLIC IP IN PI SCRIPTS
# we can run both laptop scripts first
# after laptop ready
# run ClientScriptPYSend.py will stop automatically after 2 minutes
# run ClientScriptPYrecieve.py will run

DATASET_MEAN = (0.485, 0.456, 0.406)
DATASET_STD = (0.229, 0.224, 0.225)


class MyHandler(FileSystemEventHandler):
    def __init__(self):
          
        def activation(x):
            x
        model = eval('smp.'+args.model_name)(
            args.encoder_name,
            encoder_weights='imagenet',
            classes=3,
            activation=activation,
        )

        pretrained_dict = torch.load('./bdd100k/model/model_Unet_1.pt')['state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        self.model = model
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
        ])

        port = 60000  # Reserve a port for your service.
        s = socket.socket()  # Create a socket object
        host = socket.gethostname()  # Get local machine name
        s.bind((host, port))  # Bind to the port
        s.listen(5)

        self.socket = s

        print('ready!')

    def on_modified(self, event):
        if '.png' in event.src_path and 'mask' not in event.src_path:
            print(f'event type: {event.event_type}  path : {event.src_path}')

            conn, addr = self.socket.accept()

            time.sleep(0.1)
            image = Image.open(event.src_path).convert('RGB')
            image = self.transforms(image).unsqueeze(0)
            pred = self.model(image)[0]
            # pred = torch.softmax(pred, dim=0)
            pred = torch.argmax(pred, dim=0).numpy()

            if np.mean(pred) > 0.2:
                print('past halfway!')
                conn.send(b'1')

            else:
                print('not quite!')
                conn.send(b'0')

            conn.close()

            # pred_img = np.zeros((224, 224, 3), dtype=np.uint8)
            # for dim in range(3):
            #     pred_img[:, :, dim] = (pred == dim) * 255
            # img = Image.fromarray(pred_img).convert('RGB')
            # img.save(event.src_path.replace('.png', '_mask.png'))


if __name__ == "__main__":
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path='./SmartStop/Server/', recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()