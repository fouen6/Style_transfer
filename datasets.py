import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        简单图片数据集
        :param root: 存储图片的文件夹路径
        :param transform: 应用于图像的转换
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname)
                            for fname in os.listdir(root)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    def __getitem__(self, index):
        """
        获取一个图片
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """
        返回数据集中的图片数量
        """
        return len(self.image_paths)
