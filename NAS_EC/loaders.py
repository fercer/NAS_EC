from __future__ import print_function, division
import os
import torch
from functools import reduce
from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


VALID_FILE_EXTENSIONS = ['gif', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'ppm', 'pgm',
                         'GIF', 'PNG', 'JPG', 'JPEG', 'TIF', 'TIFF', 'PPM', 'PGM']


class Rescale(object):
    """Rescale the image and its respective target to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        try:
            image, target = sample

        except ValueError:
            image = sample
            target = None

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        res_image = transform.resize(image, (new_h, new_w), order=3)

        if target is None:
            return res_image, None

        if isinstance(target, tuple): # Positions are stored as tuples
            res_factor_h = float(new_h)/h
            res_factor_w = float(new_w)/w
            if len(target) < 3:
                disc_size = []
                pos = target 
            else: 
                disc_size = [target[-3]*res_factor_w, target[-2]*res_factor_h, target[-1]]
                pos = target[:-3]

            pos = np.array(pos).reshape(-1, 2)
            res_target = tuple(reduce(lambda l1, l2: l1 + l2, map(lambda p: [int(p[0]*res_factor_w), int(p[1]*res_factor_h)], pos)) + disc_size)

        elif isinstance(target, np.ndarray) and target.ndim > 1 and target.shape[-2] > 1 and target.shape[-1] > 1:
            res_target = transform.resize(target, (new_h, new_w), order=0, anti_aliasing=False)

        else:
            res_target = target

        return res_image, res_target


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        try:
            image, target = sample

        except ValueError:
            image = sample
            target = None

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        crop_image = image[top:(top + new_h), left:(left + new_w)]

        if target is None:
            return crop_image, None

        if isinstance(target, tuple): # positions are stored as tuples
            if len(target) < 3:
                disc_size = []
                pos = target 
            else: 
                disc_size = list(target[-3:])
                pos = target[:-3]

            pos = np.array(pos).reshape(-1, 2)
            crop_target = tuple(reduce(lambda l1, l2: l1 + l2, map(lambda p: [p[0] - left, p[1] - top], pos)) + disc_size)

        elif isinstance(target, np.ndarray) and target.ndim > 1 and target.shape[-2] > 1 and target.shape[-1] > 1:
            crop_target = target[top:(top + new_h), left:(left + new_w)]

        else:
            crop_target = target

        return crop_image, crop_target


class RandomFlip(object):
    """Flip the image and its respective target randomly.

    """
    def __init__(self, v_flip_prob=0.5, h_flip_prob=0.5):
        self.v_flip_prob = v_flip_prob
        self.h_flip_prob = h_flip_prob

    def __call__(self, sample):
        try:
            image, target = sample

        except ValueError:
            image = sample
            target = None

        v_flip = np.random.rand() <= self.v_flip_prob
        h_flip = np.random.rand() <= self.h_flip_prob

        flip_image = image[::-1, :, :] if v_flip else image
        flip_image = flip_image[:, ::-1, :] if h_flip else flip_image

        if target is None:
            return flip_image, None

        if isinstance(target, tuple): # positions are stored as tuples
            h, w = image.shape[:2]
            if len(target) < 3:
                disc_size = []
                pos = target 

            else:
                flip_angle = target[-1]
                flip_angle = np.arcsin(-np.sin(flip_angle)) if v_flip else flip_angle
                flip_angle = np.arccos(-np.cos(flip_angle)) if h_flip else flip_angle
                disc_size = list(target[-3:-1]) + [target[-1] + flip_angle]
                pos = target[:-3]

            pos = np.array(pos).reshape(-1, 2)
            flip_target = tuple(reduce(lambda l1, l2: l1 + l2, map(lambda p: [(w - p[0] - 1) if h_flip else p[0], (h - p[1] - 1) if v_flip else p[1]], pos)) + disc_size)

        elif isinstance(target, np.ndarray) and target.ndim > 1 and target.shape[-2] > 1 and target.shape[-1] > 1:
            if target.ndim > 2:
                flip_target = target[::-1, :, :] if v_flip else target
                flip_target = flip_target[:, ::-1, :] if h_flip else flip_target

            else:
                flip_target = target[::-1, :] if v_flip else target
                flip_target = flip_target[:, ::-1] if h_flip else flip_target

        else:
            flip_target = target

        return flip_image, flip_target


class RandomRotation(object):
    """Rotate the image and its respective target by a random angle.

    Args:
        center_offset: Allow to move the center of reference on the image a random offset.
    """
    def __init__(self, center_offset=False):
        self.center_offset = center_offset

    def __call__(self, sample):
        try:
            image, target = sample

        except ValueError:
            image = sample
            target = None

        h, w = image.shape[:2]

        center_h = int(np.random.rand() * h) if self.center_offset else h//2
        center_w = int(np.random.rand() * w) if self.center_offset else w//2
        angle = np.random.rand() * 360.0

        rot_image = transform.rotate(image, angle, False, [center_w, center_h], order=3)

        if target is None:
            return rot_image, None
        
        if isinstance(target, tuple): # positions are stored as tuples
            a_cos = np.cos(angle/180*np.pi)
            a_sin = np.sin(angle/180*np.pi)
            rot_mat = np.array([[a_cos, -a_sin], [a_sin, a_cos]], dtype=np.float32)
            if len(target) < 3:
                disc_size = []
                pos = target 
            else: 
                disc_size = list(target[-3:-1]) + [np.mod(target[-1] + angle/180.0 * np.pi, 2*np.pi)]
                pos = target[:-3]

            pos = np.array(pos).reshape(-1, 2) - np.array([[center_w, center_h]])
            pos = np.matmul(pos, rot_mat)
            pos += np.array([[center_w, center_h]])
            rot_target = tuple(list(pos.flatten()) + disc_size)

        elif isinstance(target, np.ndarray) and target.ndim > 1 and target.shape[-2] > 1 and target.shape[-1] > 1:
            rot_target = transform.rotate(target, angle, False, [center_w, center_h], order=0)

        else:
            rot_target = target

        return rot_image, rot_target


class ColorConvert(object):
    """Convert the colors space of the input image and target

    Args:
        image_weights (tuple or numpy array): Weights assigned to each channel of the input image.
        target_weights (tuple or numpy array): Weights assigned to each channel of the target image.
    """
    def __init__(self, image_weigths, target_weights=None):
        self.image_weights = image_weigths
        self.target_weights = target_weights

    def __call__(self, sample):
        try:
            image, target = sample

        except ValueError:
            image = sample
            target = None

        if self.image_weights is not None:
            image = image[..., 0] * self.image_weights[0] + image[..., 1] * self.image_weights[1] + image[..., 2] * self.image_weights[2]

        if target is None:
            return image, None

        if self.target_weight is not None and isinstance(target, np.ndarray) and target.ndim > 1 and target.shape[-2] > 1 and target.shape[-1] > 1:
            target = target[..., 0] * self.target_weights[0] + target[..., 1] * self.target_weights[1] + target[..., 2] * self.target_weights[2]

        return image, target


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, input_datatype=np.float32, target_datatype=np.int64):
        self.input_datatype = input_datatype
        self.target_datatype = target_datatype

    def __call__(self, sample):
        try:
            image, target = sample

        except ValueError:
            image = sample
            target = None

        if image.ndim > 2:
            image = image.transpose((2, 0, 1))

        else:
            image = image[:, np.newaxis, ...]

        if target is None:
            return torch.from_numpy(image.copy().astype(self.input_datatype)), None

        if isinstance(target, np.ndarray) and target.ndim > 1 and target.shape[-2] > 1 and target.shape[-1] > 1:
            target = target.transpose((2, 0, 1)).astype(self.target_datatype)

        else:
            target = np.array(target, dtype=self.target_datatype)


        return torch.from_numpy(image.copy().astype(self.input_datatype)), torch.from_numpy(target.copy())


class ImageDataset(Dataset):
    def __init__(self, root_dir, available_indices=None, transform=None, dataset_size=-1):
        self.transform = transform
        self.root_dir = root_dir

        im_file_list = list(map(lambda fn: fn, sorted(filter(lambda fn: fn.split('.')[-1] in VALID_FILE_EXTENSIONS, os.listdir(os.path.join(root_dir, 'Data'))))))
        self.im_fn_dict = {}
        if available_indices is not None:
            self.im_file_list = im_file_list[available_indices[0]:(available_indices[1]+1)]
            for i, fn_i in enumerate(range(available_indices[0], (available_indices[1]+1))):
                fn = im_file_list[fn_i]
                self.im_fn_dict[fn.split('.')[0]] = (fn_i, i)

        else:
            self.im_file_list = im_file_list
            for i, fn in enumerate(im_file_list):                
                self.im_fn_dict[fn.split('.')[0]] = (i, i)

        self.dataset_size = len(self.im_file_list) if dataset_size < 0 else dataset_size

    def _getTarget(self, idx):
        return None

    def _getImage(self, idx):
        image = io.imread(os.path.join(self.root_dir, 'Data', self.im_file_list[idx]))
        return image

    def _transform(self, image, target):
        if self.transform:
            image, tr_target = self.transform((image, target))

            if tr_target is not None:
                target = tr_target

        return image, target

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if idx >= len(self.im_file_list):
            idx = np.random.randint(len(self.im_file_list))

        image = self._getImage(idx)
        target = self._getTarget(idx)
        image, target = self._transform(image, target)

        return image, target


class Image2ImageDataset(ImageDataset):
    def __init__(self, root_dir, task_folder, available_indices=None, transform=None, **kwargs):
        super(Image2ImageDataset, self).__init__(root_dir, available_indices, transform, **kwargs)
        self.task_folder = task_folder
        self._loadGroundtruth(available_indices)

    def _loadGroundtruth(self, available_indices=None):
        gt_file_list = list(map(lambda fn: fn, sorted(filter(lambda fn: fn.split('.')[-1] in VALID_FILE_EXTENSIONS, os.listdir(os.path.join(self.root_dir, self.task_folder))))))
        if available_indices is not None:
            self.gt_file_list = gt_file_list[available_indices[0]:(available_indices[1]+1)]

        else:
            self.gt_file_list = gt_file_list

    def _getTarget(self, idx):
        target = io.imread(os.path.join(self.root_dir, self.task_folder, self.gt_file_list[idx]))
        return target


class Image2LabelDataset(ImageDataset):
    def __init__(self, root_dir, task_folder, available_indices=None, transform=None, n_labels=1, **kwargs):
        super(Image2LabelDataset, self).__init__(root_dir, available_indices, transform, **kwargs)
        self.task_folder = task_folder
        self.n_labels = n_labels
        self._loadGroundtruth(available_indices)

    def _loadGroundtruth(self, available_indices=None):
        self.gt_labels = [[0]*self.n_labels for _ in range(len(self.im_file_list))]

    def _getTarget(self, idx):
        return self.gt_labels[idx]


class Image2PositionsDataset(Image2LabelDataset):
    def __init__(self, root_dir, task_folder, subtask_folder, available_indices=None, transform=None, **kwargs):
        super(Image2PositionsDataset, self).__init__(root_dir, task_folder, subtask_folder, available_indices, transform, **kwargs)

    def _loadGroundtruth(self, available_indices=None):
        self.gt_labels = []*len(self.im_file_list)

    def _getTarget(self, idx):
        target = self.gt_labels[idx]
        return target


class StareDiagnoses(Image2LabelDataset):
    def __init__(self, root_dir, available_indices=None, transform=None, **kwargs):
        super(StareDiagnoses, self).__init__(root_dir, 'Classification/Goldstandards', available_indices, transform, n_labels=15, **kwargs)

    def _loadGroundtruth(self, available_indices=None):
        self.gt_labels = np.zeros([len(self.im_file_list), self.n_labels], dtype=np.int64)
        fp = open(os.path.join(self.root_dir, 'Classification', 'all-mg-codes.csv'), 'r', encoding='utf-8-sig')
        for curr_line in fp.readlines():
            curr_line = curr_line.split(';')
            im_fn = curr_line[0]
            labels = curr_line[1].split(' ')
            fn_i, i = self.im_fn_dict.get(im_fn, (None, None))
            if fn_i is not None and (available_indices is None or fn_i in available_indices):
                labels = list(map(lambda l: int(l), labels))
                for l in labels:
                    self.gt_labels[i][l] = 1

        fp.close()


class StareOpticNerve(Image2PositionsDataset):
    def __init__(self, root_dir, available_indices=None, transform=None, **kwargs):
        super(StareOpticNerve, self).__init__(root_dir, 'OpticNerve', available_indices, transform, **kwargs)
        
    def _loadGroundtruth(self, available_indices=None):
        self.gt_labels = []

        fp = open(os.path.join(self.root_dir, 'OpticNerve', 'GT_NERVES.txt'), 'r')
        for curr_line in fp.readlines():
            curr_line = curr_line.split(' ')
            im_fn = curr_line[0].split('.')[0]
            position = curr_line[1:]
            fn_i, i = self.im_fn_dict.get(im_fn, (None, None))
            if fn_i is not None and (available_indices is None or fn_i in available_indices):
                position = list(map(lambda p: int(p), position))
                self.gt_labels.append(tuple(position))

        fp.close()


class DriveSegmentation(Image2ImageDataset):
    def __init__(self, root_dir, transform=None, train=False, **kwargs):
        if train:
            available_indices = (20, 39)

        else:
            available_indices = (0, 19)

        super(DriveSegmentation, self).__init__(root_dir, 'Segmentation', available_indices, transform, **kwargs)


class DriveFOV(Image2ImageDataset):
    def __init__(self, root_dir, transform=None, train=False, **kwargs):
        if train:
            available_indices = (20, 39)

        else:
            available_indices = (0, 19)

        super(DriveFOV, self).__init__(root_dir, 'FOV', available_indices, transform, **kwargs)


class CHASEDB1Segmentation(Image2ImageDataset):
    def __init__(self, root_dir, available_indices=None, transform=None, **kwargs):
        super(CHASEDB1Segmentation, self).__init__(root_dir, 'Segmentation', available_indices, transform, **kwargs)

    def _loadGroundtruth(self, available_indices=None):
        gt_file_list = list(map(lambda fn: fn, sorted(filter(lambda fn: fn.split('.')[-1] in VALID_FILE_EXTENSIONS, os.listdir(os.path.join(self.root_dir, self.task_folder))))))
        if available_indices is not None:
            self.gt_file_list = gt_file_list[available_indices[0]:2*(available_indices[1]+1):2]

        else:
            self.gt_file_list = gt_file_list

    def _getTarget(self, idx):
        target = np.concatenate((io.imread(os.path.join(self.root_dir, self.task_folder, self.gt_file_list[2*idx]))[..., np.newaxis], io.imread(os.path.join(self.root_dir, self.task_folder, self.gt_file_list[2*idx+1]))[..., np.newaxis]), axis=2)
        return target


class HRFSegmentation(Image2ImageDataset):
    def __init__(self, root_dir, available_indices=None, transform=None, **kwargs):
        super(HRFSegmentation, self).__init__(root_dir, 'Segmentation', available_indices, transform, **kwargs)
        

class HRFFOV(Image2ImageDataset):
    def __init__(self, root_dir, available_indices=None, transform=None, **kwargs):
        super(HRFFOV, self).__init__(root_dir, 'FOV', available_indices, transform, **kwargs)
        

class HRFOpticNerve(Image2PositionsDataset):
    def __init__(self, root_dir, available_indices=None, transform=None, **kwargs):
        super(HRFOpticNerve, self).__init__(root_dir, 'OpticNerve', available_indices, transform, **kwargs)

    def _loadGroundtruth(self, available_indices=None):
        self.gt_labels = []

        fp = open(os.path.join(self.root_dir, 'OpticNerve', 'optic_disk_centers.csv'), 'r')
        fp.readline() # Headers
        for curr_line in fp.readlines():
            curr_line = curr_line.split(',')
            im_fn = curr_line[0]
            position = curr_line[1:] # Add agin the disc size, therefore it can be used to form an ellipse when a resize is applied. The last value is the angle, in order to encode rotations
            fn_i, i = self.im_fn_dict.get(im_fn, (None, None))
            if fn_i is not None and (available_indices is None or fn_i in available_indices):
                position = list(map(lambda p: int(p), position))
                position += [position[-1], 0]
                self.gt_labels.append(tuple(position))

        fp.close()


if __name__ == '__main__':
    from matplotlib.patches import Circle
    print('Testing datasets and dataloaders')

    # composed = transforms.Compose([Rescale((565, 565)), RandomRotation(center_offset=False), RandomFlip(), ToTensor()])
    composed = transforms.Compose([Rescale((224, 224)), RandomFlip(), RandomRotation(center_offset=False), ToTensor()])
    stare_ds = StareDiagnoses(root_dir=r'D:\Test_data\Retinal_images\STARE', transform=composed)
    dataloader = DataLoader(stare_ds, batch_size=8, shuffle=True, num_workers=0)

    for i_batch, (image, target) in enumerate(dataloader):
        print(i_batch, image.shape, target)