import io
import sys

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import zipfile
import cv2
import numpy as np
from PIL import Image

class ZipDataset(Dataset):
    def __init__(self, root_path, cache_into_memory=True):
        if cache_into_memory:
            f = open(root_path, 'rb')
            self.zip_content = f.read()
            f.close()
            self.zip_file = zipfile.ZipFile(io.BytesIO(self.zip_content), 'r')
        else:
            self.zip_file = zipfile.ZipFile(root_path, 'r')
        self.name_list = list(filter(lambda x: x[-4:] == '.jpg', self.zip_file.namelist()))
        self.to_tensor = ToTensor()

    def __getitem__(self, key):
        buf = self.zip_file.read(name=self.name_list[key])
        img = self.to_tensor(cv2.imdecode(np.fromstring(buf, dtype=np.uint8), cv2.IMREAD_COLOR))

        # pil_img = Image.fromarray(arr[:, :, ::-1]) # because the current mode is BGR
        # pil_img.save('savedimage.jpg')
        return img

    def __len__(self):
        return len(self.name_list)


if __name__ == '__main__':
    root_path = '/Users/pratik/data/celeba/img_align_celeba.zip'
    f = open(root_path, 'rb')
    zip_content = f.read()
    f.close()
    zip_file = zipfile.ZipFile(io.BytesIO(zip_content), 'r')
    name_list = list(filter(lambda x: x[-4:] == '.jpg', zip_file.namelist()))

    buf = zip_file.read(name=name_list[int(sys.argv[1])])
    arr1 = np.frombuffer(buf, dtype=np.uint8)
    arr = cv2.imdecode(arr1, cv2.IMREAD_COLOR)

    pil_img = Image.fromarray(arr[:, :, ::-1]) # because the current mode is BGR
    pil_img.save('savedimage.jpg')

