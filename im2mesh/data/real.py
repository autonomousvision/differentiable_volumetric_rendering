import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms


class ImageDataset(data.Dataset):
    r""" Cars Dataset.

    Args:
        dataset_folder (str): path to the dataset dataset
        img_size (int): size of the cropped images
        transform (list): list of transformations applied to the data points
    """

    def __init__(self, dataset_folder, img_size=224, transform=None,
                 return_idx=False,
                 img_extensions=['.jpg', '.jpeg', '.JPG', '.JPEG', '.png',
                                 '.PNG'],):
        """

        Arguments:
            dataset_folder (path): path to the KITTI dataset
            img_size (int): required size of the cropped images
            return_idx (bool): wether to return index
        """
        self.img_path = dataset_folder
        self.file_list = os.listdir(self.img_path)

        self.file_list = [
            f for f in self.file_list
            if os.path.splitext(f)[1] in img_extensions
        ]
        self.len = len(self.file_list)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.return_idx = return_idx

    def get_model(self, idx):
        ''' Returns the model.

        Args:
            idx (int): ID of data point
        '''
        f_name = os.path.basename(self.file_list[idx])
        f_name = os.path.splitext(f_name)[0]
        return f_name

    def get_model_dict(self, idx):
        f_name = os.path.basename(self.file_list[idx])
        model_dict = {
            'model': f_name
        }
        return model_dict

    def __len__(self):
        ''' Returns the length of the dataset.'''
        return self.len

    def __getitem__(self, idx):
        ''' Returns the data point.

        Args:
            idx (int): ID of data point
        '''
        f = os.path.join(self.img_path, self.file_list[idx])
        img_in = Image.open(f)
        img = Image.new("RGB", img_in.size)
        img.paste(img_in)
        if self.transform:
            img = self.transform(img)

        idx = torch.tensor(idx)

        data = {
            'inputs': img,
        }

        if self.return_idx:
            data['idx'] = idx

        return data
