import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from .copy_paste import copy_paste

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img1 /= 255.0
        img1 -= self.mean
        img1 /= self.std
        img2 /= 255.0
        img2 -= self.mean
        img2 /= self.std


        return {'image': (img1, img2),
                'label': mask}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.float32).transpose((2, 0, 1))
        img2 = np.array(img2).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img1 = torch.from_numpy(img1).float()
        img2 = torch.from_numpy(img2).float()
        mask = torch.from_numpy(mask).float()/255

        return {'image': (img1, img2),
                'label': mask}

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': (img1, img2),
                'label': mask}

class RandomVerticalFlip(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        return {'image': (img1, img2),
                'label': mask}

class RandomFixRotate(object):
    def __init__(self):
        self.degree = [Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            rotate_degree = random.choice(self.degree)
            img1 = img1.transpose(rotate_degree)
            img2 = img2.transpose(rotate_degree)
            mask = mask.transpose(rotate_degree)

        return {'image': (img1, img2),
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img1 = img1.rotate(rotate_degree, Image.BILINEAR, expand=False)
            img2 = img2.rotate(rotate_degree, Image.BILINEAR, expand=False)
            mask = mask.rotate(rotate_degree, Image.NEAREST, expand=False)

        return {'image': (img1, img2),
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': (img1, img2),
                'label': mask}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']

        # print(img1.size)
        # print(img2.size)
        # print(mask.size)
        assert img1.size == mask.size and img2.size == mask.size

        img1 = img1.resize(self.size, Image.BILINEAR)
        img2 = img2.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': (img1, img2),
                'label': mask}

class AdjustBrightness(object):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        ran_val = random.random()
        if ran_val < self.prob:
            brightness_factor = 0.5 + random.random()
            img1 = F.adjust_brightness(img1, brightness_factor)  # torchvision.transforms mainly focus on PIL image
            # brightness_factor = 0.5 + random.random()
            img2 = F.adjust_brightness(img2, brightness_factor)

        return {'image': (img1,img2),
                'label': mask}

class AdjustContrast(object):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < self.prob:
            # contrast_factor [0.5, 1.5)
            contrast_factor = 0.5 + random.random()
            img1 = F.adjust_contrast(img1, contrast_factor)
            # contrast_factor = 0.5 + random.random()
            img2 = F.adjust_contrast(img2, contrast_factor)

        return {'image': (img1,img2),
                'label': mask}

class AdjustHue(object):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < self.prob:
            # hue_factor must in [-0.5, 0.5]
            hue_factor = -0.5 + random.random()
            # print("hue_factor:{}".format(hue_factor))
            img1 = F.adjust_hue(img1, hue_factor)
            # hue_factor = -0.5 + random.random()
            img2 = F.adjust_hue(img2, hue_factor)

        return {'image': (img1,img2),
                'label': mask}

class AdjustGamma(object):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < self.prob:
            # gamma [0.25, 4]
            gamma = 0.25 + random.random() * (4.0 - 0.25)
            img1 = F.adjust_gamma(img1, gamma)
            # gamma = 0.25 + random.random() * (4.0 - 0.25)
            img2 = F.adjust_gamma(img2, gamma)

        return {'image': (img1,img2),
                'label': mask}

class AdjustSaturation(object):
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < self.prob:
            # saturation_factor [0.5, 1.5)
            saturation_factor = 0.5 + random.random()
            # print("saturation_factor:{}".format(saturation_factor))
            img1 = F.adjust_saturation(img1, saturation_factor)
            # saturation_factor = 0.5 + random.random()
            img2 = F.adjust_saturation(img2, saturation_factor)

        return {'image': (img1,img2),
                'label': mask}

class RandomGaussianBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < self.prob:
            radius = random.random()
            img1 = img1.filter(ImageFilter.GaussianBlur(
                radius=radius))
            img2 = img2.filter(ImageFilter.GaussianBlur(
                radius=radius))

        return {'image': (img1,img2),
                'label': mask}

def style_transfer_FTT(source_image, target_image):
    # source_image = np.array(source_image)
    # target_image = np.array(target_image)
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:, :, i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:, :, i])
        target_image_fshift = np.fft.fftshift(target_image_f)

        change_length = 1
        source_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
        int(h / 2) - change_length:int(h / 2) + change_length] = \
            target_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length]

        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)

        source_image_if[source_image_if > 255] = np.max(source_image[:, :, i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)
    out = out.astype(np.uint8)
    # out = Image.fromarray(out).convert('RGB')
    return out

class StyleTransform(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        mask = np.array(mask).astype(np.uint8)

#        if random.random() < 0.5:
        img2 = style_transfer_FTT(img2, img1)
#        else:
#        img1 = style_transfer_FTT(img1, img2)

        img1 = np.array(img1)
        img1 = img1.astype(np.uint8)
        img1 = Image.fromarray(img1).convert('RGB')
        img2 = np.array(img2)
        img2 = img2.astype(np.uint8)
        img2 = Image.fromarray(img2).convert('RGB')
        mask = np.array(mask)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        return {'image': (img1,img2),
                'label': mask,
                'image_copy':sample['image_copy'],
                'label_copy':sample['label_copy']}

class CopyPaste(object):
    def __call__(self, sample):
        img1 = sample['image'][0]
        img2 = sample['image'][1]
        mask = sample['label']
        if random.random() < 0.5:
            img1 = np.array(img1)
            img2 = np.array(img2)
            mask = np.array(mask)
            # mask[mask == 1] = 255
            random_value = random.random()
            image_copy_A = sample['image_copy'][0]
            image_B_copy = sample['image_copy'][1]
            label_copy = sample['label_copy']
            image_copy_A = np.array(image_copy_A).astype(np.uint8)
            image_B_copy = np.array(image_B_copy).astype(np.uint8)
            label_copy = np.array(label_copy).astype(np.uint8)
            # label_copy[label_copy == 1] = 255
            # label_copy = np.array(label_copy)
            # label_copy[label_copy==1] = 255
            _, img1 = copy_paste(img_copy=image_copy_A, mask_copy=label_copy, img_target=img1, mask_target=mask, random=random_value)
            mask, img2 = copy_paste(img_copy=image_B_copy, mask_copy=label_copy, img_target=img2, mask_target=mask, random=random_value)
            # mask[mask == 255] = 1

        img1 = np.array(img1)
        img1 = img1.astype(np.uint8)
        img1 = Image.fromarray(img1).convert('RGB')
        img2 = np.array(img2)
        img2 = img2.astype(np.uint8)
        img2 = Image.fromarray(img2).convert('RGB')
        mask = np.array(mask)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
        return {'image':(img1,img2),
                'label':mask}

'''
We don't use Normalize here, because it will bring negative effects.
the mask of ground truth is converted to [0,1] in ToTensor() function.
'''
train_transforms = transforms.Compose([
            StyleTransform(),
            CopyPaste(),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomFixRotate(),
            RandomRotate(30),
            FixedResize(256),
            AdjustBrightness(prob=0.5),
            AdjustContrast(prob=0.5),
            AdjustHue(prob=0.5),
            AdjustGamma(prob=0.5),
            AdjustSaturation(prob=0.5),
            RandomGaussianBlur(prob=0.5),
            # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])

test_transforms = transforms.Compose([
            StyleTransform(),
            FixedResize(256),
            # RandomHorizontalFlip(),
            # RandomVerticalFlip(),
            # RandomFixRotate(),
            # RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            # RandomGaussianBlur(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()])