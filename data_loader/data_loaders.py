from torchvision import transforms
from base import BaseDataLoader
import argparse
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
from utils.data.datasets import DegCIFAR10Dataset, DegCIFAR100Dataset, DegTinyImagenetDataset

class DegCIFAR10DataLoader(BaseDataLoader):
    """
    CIFAR10 data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, train=True, 
                 deg_type = 'jpeg', deg_range = None, is_to_tensor = True, is_normalized = True, 
                 transform = None, teacher_transform = None, student_transform = None, train_init_transform = None, 
                 cutout_method = None, cutout_length = None, cutout_apply_clean = True, cutout_apply_deg = True, 
                 cutout_independent = False):
        self.data_dir = data_dir
        self.cutout_method = cutout_method
        if train:
            train_init_transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        if is_to_tensor:
            if is_normalized:
                normalize = transforms.Normalize(mean = (125.3/255.0, 123.0/255.0, 113.9/255.0), 
                                                 std = (63.0/255.0, 62.1/255.0, 66.7/255.0))
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor(), normalize])
            else:
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor()])

        self.dataset = DegCIFAR10Dataset(data_dir, train, train_init_transform, teacher_transform, student_transform, 
                                         deg_type = deg_type, deg_range = deg_range, is_to_tensor = is_to_tensor, 
                                         deg_to_tensor = self.deg_to_tensor, cutout_method = cutout_method, 
                                         cutout_length = cutout_length, cutout_apply_clean = cutout_apply_clean, 
                                         cutout_apply_deg = cutout_apply_deg, cutout_independent = cutout_independent)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class DegCIFAR100DataLoader(BaseDataLoader):
    """
    CIFAR100 data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, train=True, 
                 deg_type = 'jpeg', deg_range = None, is_to_tensor = True, is_normalized = True, 
                 transform = None, teacher_transform = None, student_transform = None, train_init_transform = None, 
                 cutout_method = None, cutout_length = None, cutout_apply_clean = True, cutout_apply_deg = True, 
                 cutout_independent = False):
        self.data_dir = data_dir
        self.cutout_method = cutout_method
        if train:
            train_init_transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        if is_to_tensor:
            if is_normalized:
                normalize = transforms.Normalize(mean = [129.3/255.0, 124.1/255.0, 112.4/255.0], 
                                                 std = [68.2/255.0, 65.4/255.0, 70.4/255.0])
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor(), normalize])
            else:
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor()])

        self.dataset = DegCIFAR100Dataset(data_dir, train, train_init_transform, teacher_transform, student_transform, 
                                         deg_type = deg_type, deg_range = deg_range, is_to_tensor = is_to_tensor, 
                                         deg_to_tensor = self.deg_to_tensor, cutout_method = cutout_method, 
                                         cutout_length = cutout_length, cutout_apply_clean = cutout_apply_clean, 
                                         cutout_apply_deg = cutout_apply_deg, cutout_independent = cutout_independent)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class DegTinyImagenetDataLoader(BaseDataLoader):
    """
    Tiny ImageNet data loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, train=True, 
                 deg_type = 'jpeg', deg_range = None, is_to_tensor = True, is_normalized = True, 
                 transform = None, teacher_transform = None, student_transform = None, train_init_transform = None, 
                 cutout_method = None, cutout_length = None, cutout_apply_clean = True, cutout_apply_deg = True, 
                 cutout_independent = False):
        self.data_dir = data_dir
        self.cutout_method = cutout_method
        if train:
            train_init_transform = transforms.Compose([transforms.RandomHorizontalFlip()])
        if is_to_tensor:
            if is_normalized:
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225])
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor(), normalize])
            else:
                self.deg_to_tensor = transforms.Compose([transforms.ToTensor()])

        self.dataset = DegTinyImagenetDataset(data_dir, train, train_init_transform, teacher_transform, student_transform, 
                                         deg_type = deg_type, deg_range = deg_range, is_to_tensor = is_to_tensor, 
                                         deg_to_tensor = self.deg_to_tensor, cutout_method = cutout_method, 
                                         cutout_length = cutout_length, cutout_apply_clean = cutout_apply_clean, 
                                         cutout_apply_deg = cutout_apply_deg, cutout_independent = cutout_independent)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Testing KD data loaders')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    # (image_clean, image_deg), targets = data_loader.dataset.__getitem__(index=5)
    # print("Save images into test_imgs from:")
    # print(image_clean.cpu().detach().numpy().shape)
    # img = image_clean.cpu().detach().numpy().transpose(1,2,0)
    # img = (img - np.min(img)) / (np.max(img) - np.min(img))
    # plt.imsave('./test_img.png', img)
    # print('targets: ', targets)

    for batch_idx, (images, targets) in enumerate(data_loader):
        (image_clean, image_deg) = images
        (labels, levels) = targets
        print(labels)
        exit()
