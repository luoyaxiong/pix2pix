"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import pandas as pd


def tensor2im(input_image, type, name, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    background = 0
    facade = 0
    opening = 0
    cover = 0
    green = 0
    orange = 0
    rest = 0
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy() # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        if type == 'fake_B':
            for i in range(image_numpy.shape[0]):
                for j in range(image_numpy.shape[1]):
                    if (  # Background
                        image_numpy[i][j][0] >= 0) & (
                        image_numpy[i][j][0] <= 5) & (
                        image_numpy[i][j][1] >= 0) & (
                        image_numpy[i][j][1] <= 5) & (
                        image_numpy[i][j][2] >= 215) & (
                            image_numpy[i][j][2] <= 230):
                        background += 1
                        #image_numpy[i][j] = [0, 0, 220]
                    elif (  # Facade
                        image_numpy[i][j][0] >= 0) & (
                        image_numpy[i][j][0] <= 10) & (
                        image_numpy[i][j][1] >= 35) & (
                        image_numpy[i][j][1] <= 60) & (
                        image_numpy[i][j][2] >= 240) & (
                            image_numpy[i][j][2] <= 255):
                        facade += 1
                        #image_numpy[i][j] = [0, 50, 250]
                    elif (  # Opening
                        image_numpy[i][j][0] >= 0) & (
                        image_numpy[i][j][0] <= 50) & (
                        image_numpy[i][j][1] >= 80) & (
                        image_numpy[i][j][1] <= 190) & (
                        image_numpy[i][j][2] >= 235) & (
                            image_numpy[i][j][2] <= 255):
                        opening += 1
                        #image_numpy[i][j] = [3, 130, 255]
                    elif (  # Cover
                        image_numpy[i][j][0] >= 200) & (
                        image_numpy[i][j][0] <= 255) & (
                        image_numpy[i][j][1] >= 200) & (
                        image_numpy[i][j][1] <= 255) & (
                        image_numpy[i][j][2] >= 5) & (
                            image_numpy[i][j][2] <= 60):
                        cover += 1
                        #image_numpy[i][j] = [250, 250, 0]
                    elif (  # Orage
                        image_numpy[i][j][0] >= 200) & (
                        image_numpy[i][j][0] <= 255) & (
                        image_numpy[i][j][1] >= 60) & (
                        image_numpy[i][j][1] <= 170) & (
                        image_numpy[i][j][2] >= 0) & (
                            image_numpy[i][j][2] <= 5):
                        orange += 1
                        #image_numpy[i][j] = [0, 50, 250]
                    elif (  # Green
                        image_numpy[i][j][0] >= 80) & (
                        image_numpy[i][j][0] <= 150) & (
                        image_numpy[i][j][1] >= 220) & (
                        image_numpy[i][j][1] <= 255) & (
                        image_numpy[i][j][2] >= 160) & (
                            image_numpy[i][j][2] <= 200):
                        green += 1
                        #image_numpy[i][j] = [0, 50, 250]
                    else:
                        rest += 1
                        #image_numpy[i][j] = [0, 0, 220]
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    if type == 'fake_B':
        data = pd.DataFrame([background, facade, opening,
                             cover, green, orange, rest])
        path = r'C:/Users/Xiaorang/Desktop/Project/model of proj2/pytorch-CycleGAN-and-pix2pix/results/facades_pix2pix/test_latest/gsv_ratio/'
        data.to_csv(path + name + 'data.csv')
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print(
            'mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' %
            (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
