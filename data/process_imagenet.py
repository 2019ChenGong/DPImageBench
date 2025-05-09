from PIL import Image
from torchvision import transforms
import torchvision
import os
import argparse

class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return torchvision.transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


def main(args):
    data_dir = args.data_dir
    image_size = args.image_size
    new_dir = args.new_dir

    print(new_dir)

    transform=transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize(size=(image_size, image_size))])
    
    #if not os.path.exists(new_dir):
    #    os.mkdir(new_dir)
    target_classes = os.listdir(data_dir)

    directory = os.path.expanduser(data_dir)
    new_directory = os.path.expanduser(new_dir)

    if args.num_workers != 1:
        sp = len(target_classes) // args.num_workers + int(len(target_classes) % args.num_workers != 0)
        target_classes = target_classes[args.worker_id*sp:(args.worker_id+1)*sp]

    c = 0
    for target_class in target_classes:
        target_dir = os.path.join(directory, target_class)
        new_target_dir = os.path.join(new_directory, target_class)
        if not os.path.exists(new_target_dir):
            os.mkdir(new_target_dir)
        
        images_name = os.listdir(target_dir)
        
        for image_name in images_name:
            image_path = os.path.join(target_dir, image_name)

            image = Image.open(image_path)
            image = image.convert('RGB')
            image = transform(image)
            new_image_name = image_name.split('.')[0]+'.png'
            image.save(os.path.join(new_target_dir, new_image_name))
            
        c += 1
        print(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="/src/data/ImageNet_ILSVRC2012/train")
    parser.add_argument('--new_dir', type=str, default='/src/data/ImageNet32_ILSVRC2012/train')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--worker_id', type=int, default=1)
    args = parser.parse_args()
    main(args)