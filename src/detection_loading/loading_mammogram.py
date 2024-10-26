# Copyright (C) 2020 Yanqi Xu, Yiqiu Shen, Laura Heacock, Carlos Fernandez-Granda, Krzysztof J. Geras

# This file is part of Mammo-DETR.
#
# Mammo-DETR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# Mammo-DETR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with Mammo-DETR.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np
import cv2, h5py
from numpy.random import RandomState
import imageio

"""
TODO:
1. remove the augmentation code in this file
2. lots of code can be deleted
"""

def read_h5(file_name):
    data = h5py.File(file_name, 'r')
    image = np.array(data['image']).T
    data.close()
    return image


def simple_rotation(image_to_rotate, angle):
    '''
    rotates image with a given angle using a given library

    Returns:
    rotated image
    '''

    def _rotate_bound(image, angle, mode):
        # Note, this method also works for 3D [H, W, C] images

        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH), flags=mode)

    image_rotated = _rotate_bound(image_to_rotate, angle, cv2.INTER_CUBIC)
    return image_rotated


def simple_resize(image_to_resize, size):
    '''
    resizes image_to_resie to size using given library

    Returns:
    resized image
    '''
    image_resized = cv2.resize(
        image_to_resize, (size[1], size[0]), interpolation=cv2.INTER_CUBIC
    )
    if len(image_to_resize.shape) == 3 and len(image_resized.shape) == 2 and image_to_resize.shape[2] == 1:
        image_resized = np.expand_dims(image_resized, 2)
    return image_resized


def _sample_rotation_angle(random_number_generator, max_rotation_noise):
    '''
    randomly samples rotation angle

    Returns:
    rotation angle
    '''

    rotation_noise_multiplier = random_number_generator.uniform(
        low=-1.0, high=1.0
    )
    rotation_noise = max_rotation_noise * rotation_noise_multiplier

    return rotation_noise


def _shift_window_inside_image(start, end, image_axis_size, input_axis_size):
    '''
    if the window is drawn to be outside of the image, shift it to be inside

    Returns:
    new start and end indices
    '''
    if start < 0:
        start = 0
        end = start + input_axis_size
    elif end > image_axis_size:
        end = image_axis_size
        start = end - input_axis_size

    return start, end


def _zero_pad_and_align_window(image_axis_size, input_axis_size,
                               max_crop_and_size_noise, bidirectional):
    '''
    if the image is small, calculate padding and align window accordingly

    We made sure pad_width is positive, and after padding,
    there will be room for window to move as much as max_crop_noise

    Returns:
    new start and end indices, padding amount for front and back of this axis
    '''
    pad_width = input_axis_size - image_axis_size \
                + max_crop_and_size_noise * (2 if bidirectional else 1)
    assert (pad_width >= 0)

    if bidirectional:
        pad_front = int(pad_width / 2)
        start = max_crop_and_size_noise
    else:
        start, pad_front = 0, 0

    pad_back = pad_width - pad_front
    end = start + input_axis_size
    return start, end, pad_front, pad_back


def _window_location_at_center_point(input_size, center_y, center_x):
    '''
    calculate window location w.r.t. center point (can be outside of image)

    Returns:
    border (4 integers)
    '''
    half_height = input_size[0] // 2
    half_width = input_size[1] // 2
    top = center_y - half_height
    bottom = center_y + input_size[0] - half_height
    left = center_x - half_width
    right = center_x + input_size[1] - half_width
    return top, bottom, left, right


def calculate_cropping_border_and_pad(image, input_size,
                                      random_number_generator, max_crop_noise,
                                      max_crop_size_noise, best_center, view):
    '''
    Calculate crop using the best center point and ideal window size
    Pad small images to have enough room for crop noise and size noise

    Returns:
    padded_image, border (numpy array of 4 ints)
    '''

    max_crop_noise = np.array(max_crop_noise)
    crop_noise_multiplier = np.zeros(2, dtype=np.float32)

    is_image_2_dimensional = len(image.shape) == 2
    if is_image_2_dimensional:
        image = np.expand_dims(image, 2)

    if max_crop_noise.any():
        # there is no point in sampling crop_noise_multiplier
        # if it's going to be multiplied by (0, 0)
        crop_noise_multiplier = random_number_generator.uniform(
            low=-1.0, high=1.0, size=2
        )
    # the breast is shifted to the left,
    # there is no point in expanding the image on the left

    if best_center is None:
        # Fallback method.
        # If center point is not given for this particular example,
        # use the previous method.
        center_y, center_x = image.shape[0] // 2, input_size[1] // 2
        # window starts from 0 on x dimension,
        # but center of image in y direction
    else:
        center_y, center_x = best_center

    # get the window around the center point.
    # The window might be outside of the image.
    top, bottom, left, right = _window_location_at_center_point(
        input_size, center_y, center_x
    )

    pad_y_top, pad_y_bottom, pad_x_right = 0, 0, 0

    view_without_side = view.split('-')[1]
    if view_without_side == "MLO":
        if image.shape[0] < \
                                input_size[0] + max_crop_noise[0] + max_crop_size_noise:
            # Image is smaller than window size + noise margin in y direction.
            # MLO view: only pad at the bottom
            top, bottom, _, pad_y_bottom = _zero_pad_and_align_window(
                image.shape[0],
                input_size[0],
                max_crop_noise[0] + max_crop_size_noise,
                False
            )
    else:
        if image.shape[0] < \
                        input_size[0] + (max_crop_noise[0] + max_crop_size_noise) * 2:
            # Image is smaller than window size + noise margin in y direction.
            # CC view: pad at both top and bottom
            top, bottom, pad_y_top, pad_y_bottom = _zero_pad_and_align_window(
                image.shape[0],
                input_size[0],
                max_crop_noise[0] + max_crop_size_noise,
                True
            )

    if image.shape[1] < \
                            input_size[1] + max_crop_noise[1] + max_crop_size_noise:
        # Image is smaller than window size + noise margin in x direction.
        left, right, _, pad_x_right = _zero_pad_and_align_window(
            image.shape[1],
            input_size[1],
            max_crop_noise[1] + max_crop_size_noise,
            False
        )

    # Pad image if necessary
    # Allocating memory once and copying contents over is more efficient
    # than np.pad or concatenate
    if pad_y_top > 0 or pad_y_bottom > 0 or pad_x_right > 0:
        new_zero_array = np.zeros((
            image.shape[0] + pad_y_top + pad_y_bottom,
            image.shape[1] + pad_x_right, image.shape[2]), dtype=image.dtype)
        new_zero_array[pad_y_top: image.shape[0] + pad_y_top, \
        0: image.shape[1]] = image
        image = new_zero_array

    # if window is drawn outside of image, shift it to be inside the image.
    top, bottom = _shift_window_inside_image(
        top, bottom, image.shape[0], input_size[0]
    )
    left, right = _shift_window_inside_image(
        left, right, image.shape[1], input_size[1]
    )

    if top == 0:
        # there is nowhere to shift upwards, we only apply noise downwards
        crop_noise_multiplier[0] = np.abs(crop_noise_multiplier[0])
    elif bottom == image.shape[0]:
        # there is nowhere to shift down, we only apply noise upwards
        crop_noise_multiplier[0] = -np.abs(crop_noise_multiplier[0])
    # else: we do nothing to the noise multiplier

    if left == 0:
        # there is nowhere to shift left, we only apply noise to move right
        crop_noise_multiplier[1] = np.abs(crop_noise_multiplier[1])
    elif right == image.shape[1]:
        # there is nowhere to shift right, we only apply noise to move left
        crop_noise_multiplier[1] = -np.abs(crop_noise_multiplier[1])
    # else: we do nothing to the noise multiplier

    borders = np.array((top, bottom, left, right), dtype=np.int32)

    # if the maximum noise is too large and might put the crop outside of image
    # it has to be made smaller for this image
    top_margin = top
    bottom_margin = image.shape[0] - bottom
    left_margin = left
    right_margin = image.shape[1] - right

    if crop_noise_multiplier[0] >= 0:
        vertical_margin = bottom_margin
    else:
        vertical_margin = top_margin

    if crop_noise_multiplier[1] >= 0:
        horizontal_margin = right_margin
    else:
        horizontal_margin = left_margin

    if vertical_margin < max_crop_noise[0]:
        max_crop_noise[0] = vertical_margin

    if horizontal_margin < max_crop_noise[1]:
        max_crop_noise[1] = horizontal_margin

    crop_noise = np.round(max_crop_noise * crop_noise_multiplier)
    crop_noise = np.array(
        (crop_noise[0], crop_noise[0], crop_noise[1], crop_noise[1]),
        dtype=np.int32
    )
    borders = borders + crop_noise

    # this is to make sure that the crop isn't outside of the image
    assert ((borders[0] >= 0) and (borders[1] <= image.shape[0])
            and (borders[2] >= 0) and (borders[3] <= image.shape[1])), \
        "Centre of the crop area is sampled such that" \
        " the borders are outside of the image. Borders: " \
        + str(borders) + ', image shape: ' + str(image.shape)

    if is_image_2_dimensional:
        return image[:, :, 0], borders
    else:
        return image, borders


def _resize_randomly_border(image, input_size, borders,
                            random_number_generator, max_crop_size_noise):
    '''
    Crop using the best center point and ideal window size
    Pad small images to have enough room for crop noise and size noise

    Returns:
    padded_image, border (numpy array of 4 ints)
    '''

    size_noise_multiplier = random_number_generator.uniform(
        low=-1.0, high=1.0, size=4
    )

    top_margin = borders[0]
    bottom_margin = image.shape[0] - borders[1]
    left_margin = borders[2]
    right_margin = image.shape[1] - borders[3]

    max_crop_size_noise = min(
        max_crop_size_noise,
        top_margin, bottom_margin, left_margin, right_margin
    )

    if input_size[0] >= input_size[1]:
        max_crop_size_vertical_noise = max_crop_size_noise
        max_crop_size_horizontal_noise = np.round(
            max_crop_size_noise * (input_size[1] / input_size[0])
        )
    elif input_size[0] < input_size[1]:
        max_crop_size_vertical_noise = np.round(
            max_crop_size_noise * (input_size[0] / input_size[1])
        )
        max_crop_size_horizontal_noise = max_crop_size_noise
    else:
        raise RuntimeError()

    max_crop_size_noise = np.array(
        (max_crop_size_vertical_noise, max_crop_size_vertical_noise,
         max_crop_size_horizontal_noise, max_crop_size_horizontal_noise),
        dtype=np.int32
    )
    size_noise = np.round(max_crop_size_noise * size_noise_multiplier)
    size_noise = np.array(size_noise, dtype=np.int32)
    borders = borders + size_noise

    # this is to make sure that the crop isn't outside of the image
    assert ((borders[0] >= 0) and (borders[1] <= image.shape[0]) \
            and (borders[2] >= 0) and (borders[3] <= image.shape[1])), \
        "Centre of the crop area is sampled such that the borders are" \
        " outside of the image. Borders: " \
        + str(borders) + ', image shape: ' + str(image.shape)

    # this is to make sure that the top is above the bottom
    assert borders[1] > borders[0], \
        "Bottom above the top. Top: " + str(borders[0]) \
        + ', bottom: ' + str(borders[1])

    # this is to make sure that the left is left to the right
    assert borders[3] > borders[2], \
        "Left on the right. Left: " + str(borders[2]) \
        + ', right: ' + str(borders[3])

    return borders


def _crop_image(image, input_size, borders):
    '''
    Crop image using borders, and resize it to be input_size

	Returns:
	cropped_image
    '''
    cropped_image = image[borders[0]: borders[1], borders[2]: borders[3]]
    if ((borders[1] - borders[0]) != input_size[0]) or ((borders[3] - borders[2]) != input_size[1]):
        cropped_image = simple_resize(cropped_image, input_size)
    return cropped_image


def flip_image(image, view, horizontal_flip, mode='training'):
    side, view_without_side = view.split('-')[:2]
    if mode == 'training':
        flip_condition = ((horizontal_flip == 'NO') and (side.lower() == 'r')) or ((horizontal_flip == 'YES') and (side.lower() == 'l'))
        if view_without_side == 'LM':
            flip_condition = not flip_condition
        if flip_condition:
            image = np.fliplr(image)
    elif mode == 'medical':
        if horizontal_flip == 'YES':
            image = np.fliplr(image)
    else:
        raise KeyError(mode)

    return image


def random_augmentation_best_center(image, input_size, random_number_generator,
                                    max_crop_noise=(0, 0),
                                    max_crop_size_noise=0,
                                    max_rotation_noise=0,
                                    auxiliary_image=None,
                                    best_center=None, view=""):
    '''
    Place border according to best_center, apply random noise in location and
    size of border, pad image if necessary, crop image within the border.

    auxiliary_image is used when there is another image, e.g. cancer heatmap,
    that needs to be augmented exactly in the same way like the original

    Returns:
    Augmented image and jointly augmented auxiliary image
    '''

    if max_rotation_noise > 0:
        angle = _sample_rotation_angle(
            random_number_generator, max_rotation_noise
        )
        image = simple_rotation(image, angle)
        if auxiliary_image is not None:
            auxiliary_image = simple_rotation(auxiliary_image, angle)

    joint_image = np.expand_dims(image, 2)
    if auxiliary_image is not None:
        is_auxiliary_2_dimensional = len(auxiliary_image.shape) == 2
        if is_auxiliary_2_dimensional:
            auxiliary_image = np.expand_dims(auxiliary_image, 2)
        joint_image = np.concatenate([joint_image, auxiliary_image], axis=2)

    padded_joint_image, borders = calculate_cropping_border_and_pad(
        joint_image,
        input_size,
        random_number_generator,
        max_crop_noise,
        max_crop_size_noise,
        best_center,
        view
    )
    borders = _resize_randomly_border(
        padded_joint_image,
        input_size,
        borders,
        random_number_generator,
        max_crop_size_noise
    )
    sampled_joint_image = _crop_image(padded_joint_image, input_size, borders)

    if auxiliary_image is None:
        return sampled_joint_image[:, :, 0], None
    elif is_auxiliary_2_dimensional:
        return sampled_joint_image[:, :, 0], sampled_joint_image[:, :, 1]
    else:
        return sampled_joint_image[:, :, 0], sampled_joint_image[:, :, 1:]


def read_image_png(file_name):
    image = np.array(imageio.imread(file_name))
    image = image.astype(np.float32)
    return image


def load_mammogram_img(img_path, input_size, view, best_center, horizontal_flip="NO"):
    """
    Top-level function that loads a mammographic image
    :param data_prefix:
    :param metadata:
    :param view:
    :param img_idx:
    :param augmentation_center:
    :return:
    """
    # step 1: load image
    img = read_image_png(img_path)

    # step 2: apply horizontal flip if required
    image = flip_image(img, view, horizontal_flip)

    # step 3: crop and pad image to the required size
    # somehow, wmlce_env_1.6.1 doesn't have default_rng
    #rng = np.random.default_rng(seed=666)
    rng = RandomState(666)
    cropped_image, _ = random_augmentation_best_center(
        image=image,
        input_size=input_size,
        random_number_generator=rng,
        best_center=best_center,
        view=view,
    )
    return cropped_image

