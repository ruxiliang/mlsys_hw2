import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        return img if not flip_img else np.flip(img, 1)
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        H, W, C = img.shape
        padded_img = np.zeros((H + self.padding * 2, W + self.padding * 2, C))
        padded_img[self.padding:self.padding + H, self.padding:self.padding + W, :] = img
        return padded_img[self.padding + shift_x:self.padding + H + shift_x, self.padding + shift_y:self.padding + W + shift_y, :]
        ### END YOUR SOLUTION   

class FlattenMnist(Transform):
    def __call__(self, x: np.ndarray):
        shape = x.shape
        after_dim = 1
        for dim in shape:
            after_dim *= dim
        return x.reshape(after_dim)