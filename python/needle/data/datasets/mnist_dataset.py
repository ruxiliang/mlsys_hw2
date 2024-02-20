import gzip
import struct
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

def parse_mnist(image_filename, label_filename) -> tuple[np.ndarray[np.float32], np.ndarray[np.uint8]]:
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    
    # parse image file
    with gzip.open(image_filename) as image_file:
        image_file_content = image_file.read()
        pixels = struct.unpack(f">iiii{''.join(['B' for _ in range(len(image_file_content) - 16)])}", image_file_content)
        pixels_ndarray = np.asarray(pixels[4:], dtype=np.float32).reshape(pixels[1],784)
        pixels_ndarray = (pixels_ndarray - pixels_ndarray.min()) / pixels_ndarray.max()
    with gzip.open(label_filename) as label_file:
        label_file_content = label_file.read()
        labels = struct.unpack(f">ii{''.join(['B' for _ in range(len(label_file_content) - 8)])}", label_file_content)
        labels = np.asarray(labels[2:], dtype=np.uint8)
    return pixels_ndarray, labels


# class MNISTDataset(Dataset):
#     def __init__(
#         self,
#         image_filename: str,
#         label_filename: str,
#         transforms: Optional[List] = None,
#     ):
#         ### BEGIN YOUR SOLUTION
#         self.pixel_ndarrays, self.labels = parse_mnist(image_filename, label_filename)
#         self.transforms = transforms
#         ### END YOUR SOLUTION

#     def __getitem__(self, index) -> object:
#         ### BEGIN YOUR SOLUTION
#         imgs = self.pixel_ndarrays[index]
#         labels = self.labels[index]
#         if len(imgs.shape) > 1:
#             batch_size = len([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
#             imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs]).reshape(batch_size, 28, 28, 1)
#         else:
#             imgs = self.apply_transforms(imgs.reshape(28, 28, 1)).reshape((1, 28, 28 , 1))
#         return (imgs, labels)
#         ### END YOUR SOLUTION

#     def __len__(self) -> int:
#         ### BEGIN YOUR SOLUTION
#         return self.pixel_ndarrays.shape[0]
        
#         ### END YOUR SOLUTION
#         ### END YOUR SOLUTION
class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, "rb") as img_file:
            magic_num, img_num, row, col = struct.unpack(">4i", img_file.read(16))
            assert(magic_num == 2051)
            tot_pixels = row * col
            imgs = [np.array(struct.unpack(f"{tot_pixels}B",
                                           img_file.read(tot_pixels)),
                                           dtype=np.float32)
                    for _ in range(img_num)]
            X = np.vstack(imgs)
            X -= np.min(X)
            X /= np.max(X)
            self.X = X

        with gzip.open(label_filename, "rb") as label_file:
            magic_num, label_num = struct.unpack(">2i", label_file.read(8))
            assert(magic_num == 2049)
            self.y = np.array(struct.unpack(f"{label_num}B", label_file.read()), dtype=np.uint8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        labels = self.y[index]
        if len(imgs.shape) > 1:
            imgs = np.stack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
        else:
            imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
        return (imgs, labels)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION