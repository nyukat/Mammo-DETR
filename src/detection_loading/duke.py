import os
import random
from collections import defaultdict
from typing import Sequence

import cv2
import h5py as h5
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy.ndimage import binary_closing
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.measure import label
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

#import dbt_comp.utilities.pickling as pickling
#import dbt_comp.utilities.reading_images as reading_images


class NegativeImageSampler(Sampler):
    r"""Samples all positive images, and x% of negative images at every epoch
    Args:
        positive_indices (sequence)   : a sequence of indices for positive images
        negative_indices (sequence)   : a sequence of indices for negative images
        sample_negatives (float): percentage of samples to draw
                                  or it could be 'match', in which case we match positive and negatives
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, positive_indices: Sequence[int], negative_indices: Sequence[int],
                 sample_negatives: float, seed=0, local_rank=0, world_size=1, sampler_relational_fraction=False) -> None:

        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.sample_negatives = sample_negatives
        self.local_rank = local_rank
        self.world_size = world_size
        self.epoch = 0
        self.seed = seed
        self.sampler_relational_fraction = sampler_relational_fraction

        if isinstance(sample_negatives, str):
            # if 'match', sample the same number of negatives as positives
            if sample_negatives.lower() == 'match':
                if len(positive_indices) <= len(self.negative_indices):
                    self.number_of_negatives_to_sample = len(positive_indices)
                else:
                    # if the number of negative images is smaller than the number of positive images,
                    # just sample all negative images. Do NOT sample with replacement.
                    # this case should not happen anyways.
                    self.number_of_negatives_to_sample = len(negative_indices)
            else:
                raise KeyError(sample_negatives)
        else:
            if self.sampler_relational_fraction:
                # percentage value wrt to size of positive set
                self.number_of_negatives_to_sample = int(len(self.positive_indices) * self.sample_negatives)
            else:
                # percentage value
                assert 0 <= sample_negatives <= 1, "sampling fraction should be between 0 and 1"
                self.number_of_negatives_to_sample = int(len(self.negative_indices) * self.sample_negatives)

        self.total_size_before_dropping = len(self.positive_indices) + self.number_of_negatives_to_sample
        self.num_samples = self.total_size_before_dropping // world_size
        self.total_size = self.num_samples * world_size

        stats = {'positive dbt': len(self.positive_indices),
                 'negative dbt': self.number_of_negatives_to_sample}
        # pprint(stats)

    def __iter__(self):
        if self.world_size == 1:
            # for reproducibility
            indices_to_load_this_epoch = self.positive_indices + random.sample(
                self.negative_indices,
                self.number_of_negatives_to_sample
            )
            random.shuffle(indices_to_load_this_epoch)
        else:
            # Using DDP
            rng = np.random.RandomState(self.seed + self.epoch)
            indices_to_load_this_epoch = self.positive_indices + rng.choice(
                self.negative_indices,
                self.number_of_negatives_to_sample,
                replace=False
            ).tolist()
            rng.shuffle(indices_to_load_this_epoch)
            indices_to_load_this_epoch = indices_to_load_this_epoch[self.local_rank:self.total_size:self.world_size]

        return iter(indices_to_load_this_epoch)

    def __len__(self):
        if self.world_size == 1:
            return self.total_size_before_dropping
        return self.num_samples

    def set_epoch(self, epoch):
        #         print(f'\nsampler at {self.local_rank}: setting epoch to {epoch}\n')
        self.epoch = epoch


class NegativeImageAndFFDMSampler(Sampler):
    r"""Samples all positive images, and x% of negative images at every epoch
    Args:
        positive_indices (sequence)   : a sequence of indices for positive images
        negative_indices (sequence)   : a sequence of indices for negative images
        sample_negatives (float): percentage of samples to draw
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, positive_dbt_indices: Sequence[int], negative_dbt_indices: Sequence[int],
                 positive_ffdm_indices: Sequence[int], negative_ffdm_indices: Sequence[int],
                 sample_negatives: float, sample_ffdm: float, seed=0, local_rank=0, world_size=1, yolox=False, sampler_relational_fraction=False) -> None:

        self.positive_dbt_indices = positive_dbt_indices
        self.negative_dbt_indices = negative_dbt_indices
        self.positive_ffdm_indices = positive_ffdm_indices
        self.negative_ffdm_indices = negative_ffdm_indices
        self.sample_negatives = sample_negatives
        self.sample_ffdm = sample_ffdm
        self.local_rank = local_rank
        self.world_size = world_size
        self.epoch = 0
        self.seed = seed
        self.yolox = yolox
        if self.yolox:
            # also, the seed between processors are set differently for YOLOX
            # without ensuring the same seed between processes, this will not work
            assert self.world_size == 1, "must handle local_rank first to handle more than 1 GPU with yolox mode"
        self.sampler_relational_fraction = sampler_relational_fraction

        # print(1, len(positive_dbt_indices), len(negative_dbt_indices), self.sample_negatives)
        # determine number_of_negative_dbt_to_sample
        if isinstance(sample_negatives, str):
            # if 'match', sample the same number of negatives as positives
            if sample_negatives.lower() == 'match':
                if len(positive_dbt_indices) <= len(negative_dbt_indices):
                    self.number_of_negative_dbt_to_sample = len(positive_dbt_indices)
                else:
                    # if the number of positive images is smaller than the number of negative images,
                    # just sample all negative images. Do NOT sample with replacement.
                    # this case should not happen anyways.
                    self.number_of_negative_dbt_to_sample = len(negative_dbt_indices)
            else:
                raise KeyError(sample_negatives)
        else:
            if self.sampler_relational_fraction:
                # percentage value wrt to size of positive set
                self.number_of_negative_dbt_to_sample = int(len(positive_dbt_indices) * self.sample_negatives)
            else:
                # percentage value
                assert 0 <= sample_negatives <= 1, "sampling fraction should be between 0 and 1"
                self.number_of_negative_dbt_to_sample = int(len(self.negative_dbt_indices) * self.sample_negatives)

        # determine number_of_positive_ffdm_to_sample and number_of_negative_ffdm_to_sample
        if isinstance(sample_ffdm, str):
            # if 'match', sample the same number of ffdm positive and negative as dbt positive
            if sample_ffdm.lower() == 'match':
                if len(positive_dbt_indices) <= len(positive_ffdm_indices):
                    self.number_of_positive_ffdm_to_sample = len(positive_dbt_indices)
                else:
                    # if the number of ffdm positive images is smaller than the number of dbt positive images,
                    # just sample all ffdm positive images. Do NOT sample with replacement.
                    # this case should not happen anyways.
                    self.number_of_positive_ffdm_to_sample = len(positive_ffdm_indices)
            # if 'match', sample the same number of ffdm negative and negative as dbt positive
            if sample_ffdm.lower() == 'match':
                if len(positive_dbt_indices) <= len(negative_ffdm_indices):
                    self.number_of_negative_ffdm_to_sample = len(positive_dbt_indices)
                else:
                    # if the number of ffdm positive images is smaller than the number of dbt positive images,
                    # just sample all ffdm positive images. Do NOT sample with replacement.
                    # this case should not happen anyways.
                    self.number_of_negative_ffdm_to_sample = len(negative_ffdm_indices)
            else:
                raise KeyError(sample_ffdm)
        else:
            if self.sampler_relational_fraction:
                # percentage value wrt to size of positive set
                self.number_of_positive_ffdm_to_sample = int(len(positive_dbt_indices) * self.sample_ffdm)
                if isinstance(sample_negatives, str):
                    self.number_of_negative_ffdm_to_sample = self.number_of_positive_ffdm_to_sample
                else:
                    self.number_of_negative_ffdm_to_sample = int(self.number_of_positive_ffdm_to_sample * self.sample_negatives)
            else:
                # percentage value
                assert 0 <= sample_ffdm <= 1, "sampling fraction should be between 0 and 1"
                assert 0 <= sample_negatives <= 1, "if sample_ffdm is float, sample_negatives should also be float"
                self.number_of_positive_ffdm_to_sample = int(len(positive_ffdm_indices) * self.sample_ffdm)
                self.number_of_negative_ffdm_to_sample = int(len(negative_ffdm_indices) * self.sample_ffdm * self.sample_negatives)

        self.total_size_before_dropping = len(
            self.positive_dbt_indices) + self.number_of_negative_dbt_to_sample + self.number_of_positive_ffdm_to_sample + self.number_of_negative_ffdm_to_sample
        self.num_samples = self.total_size_before_dropping // world_size
        self.total_size = self.num_samples * world_size
        stats = {'positive dbt': len(positive_dbt_indices),
                 'negative dbt': self.number_of_negative_dbt_to_sample,
                 'positive ffdm': self.number_of_positive_ffdm_to_sample,
                 'negative ffdm': self.number_of_negative_ffdm_to_sample}
        # pprint(stats)

    def __iter__(self):
        if self.yolox:
            while True:
                yield from self.iterable()

        yield from self.iterable()

    def iterable(self):
        if self.world_size == 1:
            # for reproducibility
            indices_to_load_this_epoch = self.positive_dbt_indices + random.sample(
                self.negative_dbt_indices,
                self.number_of_negative_dbt_to_sample
            ) + random.sample(
                self.positive_ffdm_indices,
                self.number_of_positive_ffdm_to_sample
            ) + random.sample(
                self.negative_ffdm_indices,
                self.number_of_negative_ffdm_to_sample
            )
            random.shuffle(indices_to_load_this_epoch)
        else:
            # Using DDP
            rng = np.random.RandomState(self.seed + self.epoch)
            indices_to_load_this_epoch = self.positive_dbt_indices + rng.choice(
                self.negative_dbt_indices,
                self.number_of_negative_dbt_to_sample,
                replace=False
            ).tolist() + rng.choice(
                self.positive_ffdm_indices,
                self.number_of_positive_ffdm_to_sample,
                replace=False
            ).tolist() + rng.choice(
                self.negative_ffdm_indices,
                self.number_of_negative_ffdm_to_sample,
                replace=False
            ).tolist()
            rng.shuffle(indices_to_load_this_epoch)
            indices_to_load_this_epoch = indices_to_load_this_epoch[self.local_rank:self.total_size:self.world_size]

        return iter(indices_to_load_this_epoch)

    def __len__(self):
        if self.world_size == 1:
            return self.total_size_before_dropping
        return self.num_samples

    def set_epoch(self, epoch):
        #         print(f'\nsampler at {self.local_rank}: setting epoch to {epoch}\n')
        self.epoch = epoch


def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels + 1):
        this_mask = (mask == i)
        if len(img_mask[this_mask]) > 0 and img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)

    return mask, mask_pixels_dict


def get_edge_values(lesion, axis):
    """
    Finds the bounding box for the largest connected component
    """
    has_value = np.any(lesion, axis=int(axis == "y"))
    edge_start = np.arange(lesion.shape[int(axis == "x")])[has_value][0]
    edge_end = np.arange(lesion.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end


def hflip_image_annot_seg(image, annots, seg_dict):
    # channel order at this point: height, width, depth
    rows, cols, channels = image.shape

    # flip right-view images to start from left
    image = image[:, ::-1, :]

    if seg_dict is not None:
        # TODO: depending on whether or not I decide to transpose the segmentatoins,
        # the channel to flip can be different
        for k in seg_dict.keys():
            seg_dict[k] = seg_dict[k][:, ::-1, :]

    if annots is not None:
        x1 = annots[:, 0].copy()
        x2 = annots[:, 2].copy()

        x_tmp = x1.copy()

        annots[:, 0] = cols - x2
        annots[:, 2] = cols - x_tmp
    return image, annots, seg_dict


def hflip_image_annot_seg_v4(image, annots_list, seg_dict):
    # channel order at this point: height, width, depth
    rows, cols, channels = image.shape

    # flip right-view images to start from left
    image = image[:, ::-1, :]

    if seg_dict is not None:
        # TODO: depending on whether or not I decide to transpose the segmentatoins,
        # the channel to flip can be different
        for k in seg_dict.keys():
            seg_dict[k] = seg_dict[k][:, ::-1, :]

    if annots_list is not None:
        for annots in annots_list:
            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp
    return image, annots_list, seg_dict


def vflip_image_annot_seg(image, annots, seg_dict):
    # channel order at this point: height, width, depth
    rows, cols, channels = image.shape

    # flip right-view images to start from left
    image = image[::-1, :, :]

    if seg_dict is not None:
        # TODO: depending on whether or not I decide to transpose the segmentatoins,
        # the channel to flip can be different
        for k in seg_dict.keys():
            seg_dict[k] = seg_dict[k][::-1, :, :]

    if annots is not None:
        y1 = annots[:, 1].copy()
        y2 = annots[:, 3].copy()

        y_tmp = y1.copy()

        annots[:, 1] = rows - y2
        annots[:, 3] = rows - y_tmp
    return image, annots, seg_dict


def vflip_image_annot_seg_v4(image, annots_list, seg_dict):
    # channel order at this point: height, width, depth
    rows, cols, channels = image.shape

    # flip right-view images to start from left
    image = image[::-1, :, :]

    if seg_dict is not None:
        # TODO: depending on whether or not I decide to transpose the segmentatoins,
        # the channel to flip can be different
        for k in seg_dict.keys():
            seg_dict[k] = seg_dict[k][::-1, :, :]

    if annots_list is not None:
        for annots in annots_list:
            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()

            y_tmp = y1.copy()

            annots[:, 1] = rows - y2
            annots[:, 3] = rows - y_tmp
    return image, annots, seg_dict


def get_label_idx(obj_label_dict, lesion_class, ad):
    if len(obj_label_dict) == 1:
        # use one label for all
        return 0
    else:
        # set key according to label
        key = get_label_key(lesion_class, ad)

        # retrieve label index from obj_label_dict
        if len(obj_label_dict) == 2:
            # 'malignant', 'benign'
            # or
            # 'distortion', 'rest'
            return partial_match(obj_label_dict, key)

        if len(obj_label_dict) == 4:
            # 'malignant_distortion', 'malignant_rest', 'benign_distortion', 'benign_rest'
            return obj_label_dict[key]

        raise KeyError(f'{obj_label_dict}, {lesion_class}, {ad}')


def partial_match(mydict, key):
    k0, k1 = key.split('_')
    if k0 in mydict:
        return mydict[k0]
    if k1 in mydict:
        return mydict[k1]
    raise KeyError(f'{mydict}, {key}')


def get_label_key(lesion_class, ad):
    if lesion_class == 'cancer':
        if ad:
            key = 'malignant_distortion'
        else:
            key = 'malignant_rest'
    else:
        if ad:
            key = 'benign_distortion'
        else:
            key = 'benign_rest'

    return key


def revert_label_key(key):
    if key == 'malignant_distortion':
        return 'cancer', 1
    if key == 'malignant_rest':
        return 'cancer', 0
    if key == 'benign_distortion':
        return 'benign', 1
    if key == 'benign_rest':
        return 'benign', 0
    raise KeyError(key)


def get_annotations_from_lesions(lesions, obj_label_dict):
    annotations = np.zeros((0, 5))
    for lesion in lesions:
        x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
        annotation = np.zeros((1, 5))
        annotation[0, :4] = [x, y, w, h]
        annotation[0, 4] = get_label_idx(obj_label_dict, lesion['Class'], lesion['AD'])
        annotations = np.append(annotations, annotation, axis=0)
    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
    return annotations


def get_annotations_from_lesions_V2(lesions):
    annotations = np.zeros((0, 5))
    for lesion in lesions:
        x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
        annotation = np.zeros((1, 5))
        annotation[0, :4] = [x, y, w, h]
        annotation[0, 4] = lesion['combined_label']
        annotations = np.append(annotations, annotation, axis=0)
    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
    return annotations


def concat_pad_annots(annots):
    if not torch.is_tensor(annots[0]):
        if not annots[0]:
            return None
        if annots[0] is None:
            return None

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return annot_padded


def concat_pad_annots_numpy(annots):
    if annots[0] is None:
        return None

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = np.ones((len(annots), max_num_annots, 5), dtype=np.float32) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 5), dtype=np.float32) * -1

    return annot_padded


def build_lesion_dict_wrt_slice(lesions):
    lesions_slices_dict = defaultdict(list)
    for lesion in lesions:
        lesions_slices_dict[lesion['Slice']].append(lesion)
    lesions_slices_dict.default_factory = None
    return lesions_slices_dict


def get_annotations_from_lesions_V4(lesions, num_slices):
    annotations_all_slices = []
    lesions_slices_dict = build_lesion_dict_wrt_slice(lesions)

    for i in range(num_slices):

        annotations = np.zeros((0, 5))
        for lesion in lesions_slices_dict.get(i, []):
            x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
            annotation = np.zeros((1, 5))
            annotation[0, :4] = [x, y, w, h]
            annotation[0, 4] = lesion['combined_label']
            annotations = np.append(annotations, annotation, axis=0)
        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        # gather annotations for all slices
        annotations_all_slices.append(annotations)

    #     return concat_pad_annots(annotations_all_slices)
    return annotations_all_slices


class DukeDatasetV1(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None, filter_negatives=False, apply_window=False, slice_neighbor_sampling=0,
                 random_patch_selection=False, obj_list=['malignant_distortion', 'malignant_rest', 'benign_distortion', 'benign_rest'], fraction=1,
                 load_segmentation=False, seg_dir=None, augment_boxes_without_segmentation=False, sample_wrt_class=False, **kwargs):

        self.root_dir = root_dir
        self.seg_dir = seg_dir
        self.data_list = pickling.unpickle_from_file(set)
        self.filter_negatives = filter_negatives
        self.transform = transform
        self.apply_window = apply_window
        self.slice_neighbor_sampling = slice_neighbor_sampling
        self.random_patch_selection = random_patch_selection
        self.obj_list = obj_list
        self.obj_label_dict = {k: i for i, k in enumerate(self.obj_list)}
        self.fraction = fraction
        self.load_segmentation = load_segmentation
        self.augment_boxes_without_segmentation = augment_boxes_without_segmentation
        self.sample_wrt_class = sample_wrt_class
        self.load_center_slice_negative = kwargs.get('load_center_slice_negative', False)
        self.neighboring_slices_training = kwargs.get('neighboring_slices_training', 0)
        self.phase = kwargs.get('phase', None)
        self.mixup = kwargs.get('mixup', False)

        if self.apply_window:
            raise NotImplementedError()

        self.filter_images()  # This should now be called, even if self.filter_negatives is False

    def filter_images(self):
        """
        modify the self.data_list only if self.filter_negatives > 0
        if (self.filter_negatives is False or 0) and (fraction == 1),
            then retain the original list orders for inference.
        """
        if not isinstance(self.filter_negatives, str):
            assert 0 <= float(self.filter_negatives) <= 1

        positive_list, negative_list = [], []
        positive_indices, negative_indices = [], []

        # first filter the length using self.fraction
        # and then split between images with/without lesions
        # and then further select some fraction of negative images
        if not self.sample_wrt_class:
            for i, image in enumerate(self.data_list):
                any_lesion_valid = False
                if ('lesions' in image) and (len(image['lesions']) > 0):
                    any_lesion_valid = False
                    for lesion in image['lesions']:
                        if (lesion['Width'] > 1) and (lesion['Height'] > 1):
                            any_lesion_valid = True
                if any_lesion_valid:
                    positive_list.append(image)
                    positive_indices.append(i)
                else:
                    negative_list.append(image)
                    negative_indices.append(i)
        else:
            stats = defaultdict(int)
            for i, image in enumerate(self.data_list):
                if image['classification_labels']['mal'] == 1 or image['classification_labels']['ben'] == 1:
                    if self.sample_wrt_class == 'clf':  # todo
                        positive_list.append(image)
                        positive_indices.append(i)
                    elif image['lesions']:
                        for lesion in image['lesions']:
                            if not ((lesion['Width'] > 1) and (lesion['Height'] > 1)):
                                break
                        else:
                            positive_list.append(image)
                            positive_indices.append(i)
                else:
                    negative_list.append(image)
                    negative_indices.append(i)

                if 'classification_labels' in image:
                    if image['classification_labels']:
                        stats['clf_label_full'] += 1
                    else:
                        stats['clf_label_empty'] += 1
                else:
                    stats['no_clf_label'] += 1

                if 'lesions' in image:
                    if image['lesions']:
                        stats['lesions_full'] += 1
                    else:
                        stats['lesions_empty'] += 1
                else:
                    stats['no_lesions'] += 1

            # pprint(stats)
            # print(99, len(positive_list), len(negative_list))
            # print(100, self.filter_negatives)
        if self.filter_negatives == 0:
            # No need to modify the data_list.
            # Basically, this is a special case for inference
            # where the data_list order from the pickle file is preserved
            self.positive_indices = positive_indices
            self.negative_indices = negative_indices
        else:
            # now we need to rearrange the list anyways
            # since the indices will all change anyways when we remove negative items from the original data_list
            # simply rebuild data_list by combining positive_list and negative_list
            if isinstance(self.filter_negatives, str):
                if self.filter_negatives.upper() == 'NYU':
                    # filter all NYU images, retain Duke images, without using which_dataset metadata
                    result = []
                    for image in negative_list:
                        if image['PatientID'].startswith('DBT'):
                            # Duke image
                            result.append(image)
                    negative_subset = result
                else:
                    raise KeyError(self.filter_negatives)
            else:
                negative_subset = negative_list[:int(len(negative_list) * (1 - float(self.filter_negatives)))]

            self.data_list = positive_list + negative_subset
            self.positive_indices = list(range(len(positive_list)))
            self.negative_indices = list(range(len(positive_list), len(self.data_list)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # sample a slice with lesion, and all lesions within the slice
        annot, slice_idx, lesions = self.load_annotations(idx)

        # load sampled slice
        img = self.load_image(idx, slice_idx)

        # load segmentation, if required to
        seg_dict = self.load_segmentation_func(idx, slice_idx, lesions) if self.load_segmentation else None

        # make all breasts face left direction
        img, annot, seg_dict = self.lr_flip_image_annot(img, annot, self.data_list[idx]['View'], seg_dict)

        sample = {'img': img, 'annot': annot, 'seg_dict': seg_dict}

        if self.transform:
            sample = self.transform(sample)

        return {**sample, 'index': idx}

    def lr_flip_image_annot(self, image, annots, view, seg_dict=None):
        if view[0] == 'r':
            image, annots, seg_dict = hflip_image_annot_seg(image, annots, seg_dict)

        return image, annots, seg_dict

    @staticmethod
    def get_label_class_dict(lesions):
        result = defaultdict(list)
        for lesion in lesions:
            key = get_label_key(lesion['Class'], lesion['AD'])
            result[key].append(lesion)
        # turn off automatic key generation
        result.default_factory = None
        return result

    @staticmethod
    def create_segmentation_from_box(class_lesions, image_shape):
        h, w = image_shape[1:]
        result = np.zeros((1, h, w), dtype=np.uint8)
        for lesion in class_lesions:
            x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
            x2 = x + w
            y2 = y + h
            result[:, y:y2, x:x2] = 1  # currently loading each segmentation file separately
        return result

    def load_segmentation_func(self, image_index, slice_index, lesions):
        image_metadata = self.data_list[image_index]

        # find out all lesion labels for the current slide
        lesion_classes_dict = self.get_label_class_dict(lesions)

        seg_dict = {}

        for lesion_class, class_lesions in lesion_classes_dict.items():

            seg_filename = os.path.join(
                self.seg_dir,
                f"{image_metadata.get('short_file_path', '')}.{lesion_class}.hdf5"
            )
            if os.path.exists(seg_filename):
                # load one slice of segmentation with a lesion
                seg_dict[lesion_class] = reading_images.read_image_3d_duke(
                    file_name=seg_filename,
                    start=slice_index,
                    end=slice_index + 1,
                    flip=False,  # flip is False for segmentations
                ).astype(np.uint8)  # should already be uint8, but check
            elif self.augment_boxes_without_segmentation:
                seg_dict[lesion_class] = self.create_segmentation_from_box(class_lesions, image_metadata['shape'])

            # nothing to transpose if I chose not to load the segmentation
            if lesion_class in seg_dict:
                seg_dict[lesion_class] = np.transpose(seg_dict[lesion_class], (1, 2, 0))  # channel dimension at the end
        return seg_dict

    def load_image(self, image_index, slice_index):

        # load one slice with a lesion
        image_metadata = self.data_list[image_index]
        image = reading_images.read_image_3d_duke(
            file_name=os.path.join(
                self.root_dir,
                image_metadata['hdf5_path']
            ),
            start=slice_index,
            end=slice_index + 1,
            flip=image_metadata['flip'],
        )

        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        if image.max() != 0:
            image = image.astype(np.float32) / image.max()
        return image

    def load_annotations(self, image_index):
        # get ground truth annotations

        # no lesion for this image
        if len(self.data_list[image_index]['lesions']) == 0:
            if self.load_center_slice_negative:
                return np.zeros((0, 5)), self.data_list[image_index]['shape'][0] // 2, []
            return np.zeros((0, 5)), random.randint(0, self.data_list[image_index]['shape'][0] - 1), []

        # sample a slice with a lesion
        if self.random_patch_selection:
            sampled_lesion = random.choice(self.data_list[image_index]['lesions'])
        else:
            # if no sampling, turn off randomization as well.
            sampled_lesion = self.data_list[image_index]['lesions'][0]
        slice_idx_sampled_lesion = sampled_lesion['Slice']

        # parse annotations
        # Select all lesions which appear at the current slice
        lesions_in_current_slice = []
        for idx, lesion in enumerate(self.data_list[image_index]['lesions']):

            x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
            # some annotations have basically no width / height, skip them
            if w <= 1 or h <= 1:
                continue

            # if the center of patch appears within +- self.slice_neighbor_sampling slices, extract them.
            if (slice_idx_sampled_lesion - self.slice_neighbor_sampling > lesion['Slice']) or \
                    (lesion['Slice'] > slice_idx_sampled_lesion + self.slice_neighbor_sampling):
                continue

            lesions_in_current_slice.append(lesion)

        annotations = get_annotations_from_lesions(lesions_in_current_slice, self.obj_label_dict)

        # sample slice_idx from neighbors
        sampled_slice_idx = random.randint(
            slice_idx_sampled_lesion - self.slice_neighbor_sampling,
            slice_idx_sampled_lesion + self.slice_neighbor_sampling
        )

        # return all found annotations, and the sampled slice index
        return annotations, min(max(0, sampled_slice_idx), self.data_list[image_index]['shape'][0] - 1), lesions_in_current_slice


class DukeDatasetV1Eval(DukeDatasetV1):
    # TODO: it shouldn't matter much, but I should take care of the newer init parameters from the loader V1
    def __init__(self, root_dir, set='train2017', transform=None, apply_window=False, filter_negatives=False):
        super(DukeDatasetV1Eval, self).__init__(root_dir, set=set, transform=transform, apply_window=apply_window, filter_negatives=filter_negatives,
                                                slice_neighbor_sampling=0)

    def __getitem__(self, idx):
        img = self.load_3d_image(idx)  # use channel dim as depth, convert this later
        img, _, _ = self.lr_flip_image_annot(img, None, self.data_list[idx]['View'])
        sample = {'img': img, 'annot': None}
        if self.transform:
            sample = self.transform(sample)
        return {**sample, 'index': idx}

    def load_3d_image(self, image_index):

        # load one slice with a lesion
        image_metadata = self.data_list[image_index]
        image = reading_images.read_image_3d_duke(
            file_name=os.path.join(
                self.root_dir,
                image_metadata['hdf5_path']
            ),
            start=0,
            end=image_metadata['shape'][0],
            flip=image_metadata['flip'],
        )
        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        if image.max() != 0:
            image = image.astype(np.float32) / image.max()
        return image


class DukeDatasetV2(DukeDatasetV1):
    def __init__(self, root_dir, set='train2017', transform=None, filter_negatives=False, apply_window=False, random_patch_selection=False, fraction=1,
                 load_segmentation=False, seg_dir=None, augment_boxes_without_segmentation=False, sample_wrt_class=False, **kwargs):
        # slice_neighbor_sampling and obj_list will be completely ignored
        # they are there for syntax-compatibility of constructing dataset object
        super().__init__(
            root_dir=root_dir, set=set, transform=transform, filter_negatives=filter_negatives,
            apply_window=apply_window, slice_neighbor_sampling=0, random_patch_selection=random_patch_selection,
            obj_list=[], fraction=fraction, load_segmentation=load_segmentation, seg_dir=seg_dir,
            augment_boxes_without_segmentation=augment_boxes_without_segmentation,
            load_center_slice_negative=kwargs.get('load_center_slice_negative', False),
            neighboring_slices_training=kwargs.get('neighboring_slices_training', 0),
            phase=kwargs.get('phase', None),
            mixup=kwargs.get('mixup', False),
            sample_wrt_class=sample_wrt_class,
        )
        self.check_data_version()
        self.lesion_class_to_seg_suffix_dict = {
            0: 'malignant_rest',
            1: 'benign_rest',
            2: 'malignant_distortion',
            3: 'benign_distortion',
            4: 'malignant_rest',
            5: 'benign_rest',
            6: 'malignant_distortion',
            7: 'benign_distortion',
            8: 'malignant',
            9: 'benign',
        }
        self.keep_track_of_dataset_indices()

        # currently, kwargs of loaders V3, V4, V5 don't automatically fall through to DatasetV2
        # this only works if I call DatasetV2 directly
        # TODO: pass additional arguments
        self.yolox = kwargs.get('yolox', False)
        if isinstance(self.yolox, str):
            # 'uint16' means original pixel values, probably not good since we use multiple modalities
            # 'uint8' means 0~255 adjusted pixel values, will be best since handles both modalities & matches YOLOX training
            # 'float' means standardized pixel values
            assert self.yolox in {'uint16', 'uint8', 'float'}, 'please give proper value for pixel ranges'

    def keep_track_of_dataset_indices(self):
        """
        Keep track of from which dataset different images come from.
        This will be useful to pick appropriate clf head and calculate loss.
        """
        self.positive_dbt_indices = []
        self.negative_dbt_indices = []
        self.positive_ffdm_indices = []
        self.negative_ffdm_indices = []
        self.positive_cview_indices = []
        self.negative_cview_indices = []

        for pos_idx in self.positive_indices:
            if self.data_list[pos_idx]['which_dataset'] < 2:
                self.positive_dbt_indices.append(pos_idx)
            elif self.data_list[pos_idx]['which_dataset'] == 2:
                self.positive_ffdm_indices.append(pos_idx)
            elif self.data_list[pos_idx]['which_dataset'] == 3:
                self.positive_cview_indices.append(pos_idx)
            else:
                raise AttributeError(self.data_list[pos_idx]['which_dataset'])

        for neg_idx in self.negative_indices:
            if self.data_list[neg_idx]['which_dataset'] < 2:
                self.negative_dbt_indices.append(neg_idx)
            elif self.data_list[neg_idx]['which_dataset'] == 2:
                self.negative_ffdm_indices.append(neg_idx)
            elif self.data_list[neg_idx]['which_dataset'] == 3:
                self.negative_cview_indices.append(neg_idx)
            else:
                raise AttributeError(self.data_list[neg_idx]['which_dataset'])

        self.positive_dbt_indices = self.positive_dbt_indices[:int(self.fraction * len(self.positive_dbt_indices))]
        self.negative_dbt_indices = self.negative_dbt_indices[:int(self.fraction * len(self.negative_dbt_indices))]
        self.positive_ffdm_indices = self.positive_ffdm_indices[:int(self.fraction * len(self.positive_ffdm_indices))]
        self.negative_ffdm_indices = self.negative_ffdm_indices[:int(self.fraction * len(self.negative_ffdm_indices))]

    def check_data_version(self):
        assert 'flip' not in self.data_list[0], "existence of flip bit means older (v1) version of data"
        assert 'horizontal_flip' in self.data_list[0], "to use a newer version of data, provide horizontal_flip"

    @staticmethod
    def get_label_class_dict(lesions):
        result = defaultdict(list)
        for lesion in lesions:
            key = lesion['combined_label']
            result[key].append(lesion)
        # turn off automatic key generation
        result.default_factory = None
        return result

    def lr_flip_image_annot(self, idx, image, annots, view, seg_dict=None):
        """
        check for horizontal_flip bit to determine whether or not to flip the images and annotations
        """
        horizontal_flip = self.data_list[idx]['horizontal_flip']

        if horizontal_flip == 'NO':
            if view[0] == 'r':
                image, annots, seg_dict = hflip_image_annot_seg(image, annots, seg_dict)
        elif horizontal_flip == 'YES':
            if view[0] == 'l':
                image, annots, seg_dict = hflip_image_annot_seg(image, annots, seg_dict)
        else:
            raise KeyError(horizontal_flip)

        return image, annots, seg_dict

    def __getitem__(self, idx):
        # sample a slice with lesion, and all lesions within the slice
        annot, slice_idx, lesions, clf_label = self.load_annotations(idx)

        # load sampled slice
        img = self.load_image(idx, slice_idx)

        # load segmentation, if required to
        seg_dict = self.load_segmentation_func(idx, slice_idx, lesions) if self.load_segmentation else None

        # make all breasts face left direction
        img, annot_t, seg_dict_t = self.lr_flip_image_annot(idx, img, annot, self.data_list[idx]['View'], seg_dict)

        if self.mixup:
            mixup_idx = random.choice(range(0, len(self.data_list) - 1))
            # print("using mixup", idx, mixup_idx)
            mixup_annot, mixup_slice_idx, mixup_lesions = self.load_annotations(mixup_idx)
            mixup_img = self.load_image(mixup_idx, mixup_slice_idx)
            mixup_seg_dict = self.load_segmentation_func(mixup_idx, mixup_slice_idx, mixup_lesions) if self.load_segmentation else None
            mixup_img, mixup_annot_t, mixup_seg_dict = self.lr_flip_image_annot(mixup_idx, mixup_img, mixup_annot, self.data_list[mixup_idx]['View'],
                                                                                mixup_seg_dict)

        # load neighboring slices
        if self.neighboring_slices_training > 0:
            adjacent_offset = self.neighboring_slices_training
            volume_slices = self.data_list[idx]['shape'][0]
            before_idx = np.maximum(slice_idx - adjacent_offset, 0)
            after_idx = np.minimum(slice_idx + adjacent_offset, volume_slices - 1)
            img_before = self.load_image(idx, before_idx)
            img_before, _, _ = self.lr_flip_image_annot(idx, img_before, annot, self.data_list[idx]['View'], seg_dict)
            img_after = self.load_image(idx, after_idx)
            img_after, _, _ = self.lr_flip_image_annot(idx, img_after, annot, self.data_list[idx]['View'], seg_dict)

            img_new = np.zeros((img.shape[0], img.shape[1], 3))
            img_new[..., 0] = np.squeeze(img_before)
            img_new[..., 1] = np.squeeze(img)
            img_new[..., 2] = np.squeeze(img_after)
            is_equal1 = (img_before == img_after).all()
            img = img_new

        sample = {
            'img': img, 'annot': annot_t, 'seg_dict': seg_dict_t,
            'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
            'best_center': self.data_list[idx]['best_center'],
            'view': self.data_list[idx]['View'],
            'clf_label': clf_label,
            'original_image_size': self.data_list[idx]['original_image_size'],
        }

        if self.mixup:
            # potential TODO: Using clf label and original_image_size are undefined when using mixup
            sample['mixup_sample'] = {
                'img': mixup_img,
                'annot': mixup_annot_t,
                'seg_dict': mixup_seg_dict,
                'which_dataset': self.data_list[mixup_idx]['which_dataset'],
                'best_center': self.data_list[mixup_idx]['best_center'],
                'view': self.data_list[mixup_idx]['View']
            }

        if self.transform:
            sample = self.transform(sample)

        return {**sample, 'index': idx}

    def load_segmentation_func(self, image_index, slice_index, lesions):
        # Override the segmentation loading method
        # now keys are the combined_label of the lesion
        # the segmentation suffixes are different between datasets
        image_metadata = self.data_list[image_index]

        # find out all lesion labels for the current slide
        lesion_classes_dict = self.get_label_class_dict(lesions)

        seg_dict = {}

        for lesion_class, class_lesions in lesion_classes_dict.items():

            seg_filename = os.path.join(
                self.seg_dir,
                f"{image_metadata.get('short_file_path', '')}.{self.lesion_class_to_seg_suffix_dict[lesion_class]}.hdf5"
            )
            if os.path.exists(seg_filename):
                # load one slice of segmentation with a lesion
                if image_metadata['which_dataset'] < 2:
                    seg_dict[lesion_class] = reading_images.read_image_3d_duke(
                        file_name=seg_filename,
                        start=slice_index,
                        end=slice_index + 1,
                        flip=False,  # flip is False for segmentations
                    ).astype(np.uint8)  # should already be uint8, but check
                else:
                    # TODO: wrap this additional logic into a helper function to decrease repeating the same code
                    seg_dict[lesion_class] = reading_images.read_image_mat(
                        file_name=seg_filename,
                    ).astype(np.uint8)  # should already be uint8, but check

                    # add an additional channel
                    seg_dict[lesion_class] = seg_dict[lesion_class].reshape(image_metadata['shape'])

            elif self.augment_boxes_without_segmentation:
                seg_dict[lesion_class] = self.create_segmentation_from_box(class_lesions, image_metadata['shape'])

            # nothing to transpose if I chose not to load the segmentation
            if lesion_class in seg_dict:
                seg_dict[lesion_class] = np.transpose(seg_dict[lesion_class], (1, 2, 0))  # channel dimension at the end
        return seg_dict

    def load_image(self, image_index, slice_index):
        # Override the segmentation loading method
        # flip bit is deprecated, but new horizontal_flip is introduced
        #

        # load one slice with a lesion
        image_metadata = self.data_list[image_index]

        if image_metadata['which_dataset'] < 2:
            image = reading_images.read_image_3d_duke(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                ),
                start=slice_index,
                end=slice_index + 1,
                flip=False,  # flip is always False for loader V2
            )
        else:
            # TODO: wrap this additional logic into a helper function to decrease repeating the same code
            image = reading_images.read_image_mat(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                )
            )

            # add an additional channel
            image = image.reshape(image_metadata['shape'])

        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        image = image.astype(np.float32)
        if self.yolox == 'uint16':
            # do nothing. keep the original pixel values
            pass
        else:
            # if not 'uint16', then divide the pixel values by its max
            # the pixel ranges are now 0~1, which is the default behavior when self.yolox is False or 'float'
            image_max = image.max()
            if (image_max != 0):
                image /= image_max
            if self.yolox == 'uint8':
                image *= 255
        return image

    def load_annotations(self, image_index):
        """
        Takes an image from the data_list by image_index, and loads it depending on
        whether there is a lesion, etc.
        :param image_index: index of the image in self.data_list
        :return: annot, slice_idx, lesions
        """

        # if there is no lesion for this image
        if len(self.data_list[image_index]['lesions']) == 0:
            annotations = np.zeros((0, 5))
            lesions = []
            if self.load_center_slice_negative:
                if 'important_slices' in self.data_list[image_index]:
                    slice_idx = self.data_list[image_index]['important_slices'][0]
                else:
                    slice_idx = self.data_list[image_index]['shape'][0] // 2
            else:
                if 'important_slices' in self.data_list[image_index]:
                    slice_idx = random.choice(self.data_list[image_index]['important_slices'])
                else:
                    slice_idx = random.randint(0, self.data_list[image_index]['shape'][0] - 1)

        else:
            if self.random_patch_selection:
                sampled_lesion = random.choice(self.data_list[image_index]['lesions'])
            else:
                sampled_lesion = self.data_list[image_index]['lesions'][0]
            slice_idx_sampled_lesion = sampled_lesion['Slice']

            # parse annotations
            # Select all lesions which appear at the current slice
            lesions = []
            for idx, lesion in enumerate(self.data_list[image_index]['lesions']):

                x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
                # some annotations have basically no width / height, skip them
                if w <= 1 or h <= 1:
                    continue

                # if the center of patch appears within +- self.slice_neighbor_sampling slices, extract them.
                if (slice_idx_sampled_lesion > lesion['Slice']) or \
                        (lesion['Slice'] > slice_idx_sampled_lesion):
                    continue

                lesions.append(lesion)

            # use V2 function to load combined_label as lesion label
            annotations = get_annotations_from_lesions_V2(lesions)

            # disable sampling slice_idx from neighbors
            slice_idx = min(max(0, slice_idx_sampled_lesion), self.data_list[image_index]['shape'][0] - 1)

        if 'classification_labels' in self.data_list[image_index]:
            clf_label = self.data_list[image_index]['classification_labels']
        else:
            clf_label = None

        return annotations, slice_idx, lesions, clf_label


class DukeDatasetV2Eval(DukeDatasetV2):
    def __init__(self, root_dir, set='train2017', transform=None, filter_negatives=False, apply_window=False, fraction=1, **kwargs):
        # slice_neighbor_sampling and obj_list will be completely ignored
        # they are there for syntax-compatibility of constructing dataset object
        super().__init__(
            root_dir=root_dir, set=set, transform=transform, filter_negatives=filter_negatives,
            apply_window=apply_window, random_patch_selection=False, fraction=fraction,
            load_segmentation=False, seg_dir=None, augment_boxes_without_segmentation=False,
            **kwargs
        )

    def load_3d_image(self, image_index):
        # Override the segmentation loading method
        # flip bit is deprecated, but new horizontal_flip is introduced

        # load all slices
        image_metadata = self.data_list[image_index]

        if image_metadata['which_dataset'] < 2:
            image = reading_images.read_image_3d_duke(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                ),
                start=0,
                end=image_metadata['shape'][0],
                flip=False,  # flip is always False for loader V2
            )
        else:
            raise KeyError(image_metadata['which_dataset'])

        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        if image.max() != 0:
            image = image.astype(np.float32) / image.max()
        return image

    def __getitem__(self, idx):

        # load the entire 3D image
        img = self.load_3d_image(idx)

        # make all breasts face left direction
        img, _, _ = self.lr_flip_image_annot(idx, img, None, self.data_list[idx]['View'], None)

        sample = {
            'img': img, 'annot': None,
            'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
            'best_center': self.data_list[idx]['best_center'],
            'view': self.data_list[idx]['View'],
        }

        if self.transform:
            sample = self.transform(sample)

        return {**sample, 'index': idx}


class DukeDatasetV3(DukeDatasetV2):
    """
    This dataset version supports loading volumes as 3d to enable det+clf training.
    """

    def __init__(self, root_dir, set='train2017', transform=None, filter_negatives=False, apply_window=False, random_patch_selection=False, fraction=1,
                 load_segmentation=False, seg_dir=None, augment_boxes_without_segmentation=False, mode='det', k_extra_slices=0, sample_wrt_class=False,
                 **kwargs):
        super().__init__(
            root_dir=root_dir, set=set, transform=transform, filter_negatives=filter_negatives,
            apply_window=apply_window, random_patch_selection=random_patch_selection,
            fraction=fraction, load_segmentation=load_segmentation, seg_dir=seg_dir,
            augment_boxes_without_segmentation=augment_boxes_without_segmentation,
            load_center_slice_negative=kwargs.get('load_center_slice_negative', False),
            neighboring_slices_training=kwargs.get('neighboring_slices_training', 0),
            phase=kwargs.get('phase', None),
            mixup=kwargs.get('mixup', False),
            sample_wrt_class=sample_wrt_class,
        )

        self.mode = mode
        self.k_extra_slices = k_extra_slices
        if k_extra_slices:
            assert self.mode in ['detclf', 'detawa'], "Loading more than 1 slice only supported in the detclf/detawa mode."

    def lr_flip_image_annot(self, idx, image, annots, view, seg_dict=None):
        """
        check for horizontal_flip bit to determine whether or not to flip the images and annotations
        """
        horizontal_flip = self.data_list[idx]['horizontal_flip']

        if horizontal_flip == 'NO':
            if view[0] == 'r':
                image, annots, seg_dict = hflip_image_annot_seg(image, annots, seg_dict)
        elif horizontal_flip == 'YES':
            if view[0] == 'l':
                image, annots, seg_dict = hflip_image_annot_seg(image, annots, seg_dict)
        else:
            raise KeyError(horizontal_flip)

        return image, annots, seg_dict

    def __getitem__(self, idx):
        # sample a slice with lesion, and all lesions within the slice
        annot, slice_idx, lesions, clf_label = self.load_annotations(idx)

        image_metadata = self.data_list[idx]
        img_shape = image_metadata['shape']

        if self.mode in ['detclf', 'detawa']:
            assert img_shape[0] > 1, "To use this mode, disable 2D FFDM images."
            # assert not self.load_segmentation, "detclf mode does not support segmentations so far."
            assert not self.mixup, "detclf mode does not support mixup so far."
            assert not self.neighboring_slices_training, "detclf mode does not support neighboring slices training so far."

            sample_list = []

            idx_list = [k for k in range(slice_idx - self.k_extra_slices, slice_idx + self.k_extra_slices + 1) if 0 <= k < img_shape[0]]

            for slice_idx_i in idx_list:
                img = self.load_image(idx, slice_idx_i)

                # load segmentation, if required to
                seg_dict = self.load_segmentation_func(idx, slice_idx_i, lesions) if self.load_segmentation else None

                # If the chosen slice, use annotations, otherwise, just pass 0's
                if slice_idx_i == slice_idx:
                    img, annot_t, _ = self.lr_flip_image_annot(idx, img, annot, self.data_list[idx]['View'])
                else:
                    img, annot_t, _ = self.lr_flip_image_annot(idx, img, np.zeros((0, 5)), self.data_list[idx]['View'])

                sample_idx = {
                    'img': img, 'annot': annot_t, 'seg_dict': seg_dict, 'det_slice_idx': slice_idx,
                    'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
                    'best_center': self.data_list[idx]['best_center'],
                    'view': self.data_list[idx]['View'],
                }
                if self.transform:
                    sample_idx = self.transform(sample_idx)
                sample_list.append(sample_idx)

            sample = {
                'img': [s['img'] for s in sample_list], 'annot': sample_list[idx_list.index(slice_idx)]['annot'],
                'clf_label': clf_label, 'idx_list': idx_list, 'seg_dict': None, 'det_slice_idx': slice_idx,
                'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
                'best_center': self.data_list[idx]['best_center'],
                'view': self.data_list[idx]['View'],
            }

            return {**sample, 'index': idx}
        else:
            # load sampled slice
            img = self.load_image(idx, slice_idx)

            # load segmentation, if required to
            seg_dict = self.load_segmentation_func(idx, slice_idx, lesions) if self.load_segmentation else None

            # make all breasts face left direction
            img, annot_t, seg_dict_t = self.lr_flip_image_annot(idx, img, annot, self.data_list[idx]['View'], seg_dict)

            if self.mixup:
                mixup_idx = random.choice(range(0, len(self.data_list) - 1))
                # print("using mixup", idx, mixup_idx)
                mixup_annot, mixup_slice_idx, mixup_lesions, _ = self.load_annotations(mixup_idx)
                mixup_img = self.load_image(mixup_idx, mixup_slice_idx)
                mixup_seg_dict = self.load_segmentation_func(mixup_idx, mixup_slice_idx, mixup_lesions) if self.load_segmentation else None
                mixup_img, mixup_annot_t, mixup_seg_dict = self.lr_flip_image_annot(mixup_idx, mixup_img, mixup_annot, self.data_list[mixup_idx]['View'],
                                                                                    mixup_seg_dict)

            # load neighboring slices
            if self.neighboring_slices_training > 0:
                adjacent_offset = self.neighboring_slices_training
                volume_slices = self.data_list[idx]['shape'][0]
                before_idx = np.maximum(slice_idx - adjacent_offset, 0)
                after_idx = np.minimum(slice_idx + adjacent_offset, volume_slices - 1)
                img_before = self.load_image(idx, before_idx)
                img_before, _, _ = self.lr_flip_image_annot(idx, img_before, annot, self.data_list[idx]['View'], seg_dict)
                img_after = self.load_image(idx, after_idx)
                img_after, _, _ = self.lr_flip_image_annot(idx, img_after, annot, self.data_list[idx]['View'], seg_dict)

                img_new = np.zeros((img.shape[0], img.shape[1], 3))
                img_new[..., 0] = np.squeeze(img_before)
                img_new[..., 1] = np.squeeze(img)
                img_new[..., 2] = np.squeeze(img_after)
                is_equal1 = (img_before == img_after).all()
                img = img_new

            sample = {
                'img': img, 'annot': annot_t, 'seg_dict': seg_dict_t, 'clf_label': clf_label,
                'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
                'best_center': self.data_list[idx]['best_center'],
                'view': self.data_list[idx]['View'],
            }

            if self.mixup:
                sample['mixup_sample'] = {
                    'img': mixup_img,
                    'annot': mixup_annot_t,
                    'seg_dict': mixup_seg_dict,
                    'which_dataset': self.data_list[mixup_idx]['which_dataset'],
                    'best_center': self.data_list[mixup_idx]['best_center'],
                    'view': self.data_list[mixup_idx]['View']
                }

            if self.transform:
                sample = self.transform(sample)

            return {**sample, 'index': idx}

    def load_segmentation_func(self, image_index, slice_index, lesions):
        # Override the segmentation loading method
        # now keys are the combined_label of the lesion
        # the segmentation suffixes are different between datasets
        image_metadata = self.data_list[image_index]

        # find out all lesion labels for the current slide
        lesion_classes_dict = self.get_label_class_dict(lesions)

        seg_dict = {}

        for lesion_class, class_lesions in lesion_classes_dict.items():

            seg_filename = os.path.join(
                self.seg_dir,
                f"{image_metadata.get('short_file_path', '')}.{self.lesion_class_to_seg_suffix_dict[lesion_class]}.hdf5"
            )
            if os.path.exists(seg_filename):
                # load one slice of segmentation with a lesion
                if image_metadata['which_dataset'] < 2:
                    seg_dict[lesion_class] = reading_images.read_image_3d_duke(
                        file_name=seg_filename,
                        start=slice_index,
                        end=slice_index + 1,
                        flip=False,  # flip is False for segmentations
                    ).astype(np.uint8)  # should already be uint8, but check
                else:
                    # TODO: wrap this additional logic into a helper function to decrease repeating the same code
                    seg_dict[lesion_class] = reading_images.read_image_mat(
                        file_name=seg_filename,
                    ).astype(np.uint8)  # should already be uint8, but check

                    # add an additional channel
                    seg_dict[lesion_class] = seg_dict[lesion_class].reshape(image_metadata['shape'])

            elif self.augment_boxes_without_segmentation:
                seg_dict[lesion_class] = self.create_segmentation_from_box(class_lesions, image_metadata['shape'])

            # nothing to transpose if I chose not to load the segmentation
            if lesion_class in seg_dict:
                seg_dict[lesion_class] = np.transpose(seg_dict[lesion_class], (1, 2, 0))  # channel dimension at the end
        return seg_dict

    def load_image(self, image_index, slice_index):
        # Override the segmentation loading method
        # flip bit is deprecated, but new horizontal_flip is introduced
        #

        # load one slice with a lesion
        image_metadata = self.data_list[image_index]
        clf_label = [0, 0, 0, 0]
        if image_metadata['which_dataset'] < 2:
            image = reading_images.read_image_3d_duke(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                ),
                start=slice_index,
                end=slice_index + 1,
                flip=False,  # flip is always False for loader V2
            )
        else:
            # TODO: wrap this additional logic into a helper function to decrease repeating the same code
            image = reading_images.read_image_mat(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                )
            )

            # add an additional channel
            image = image.reshape(image_metadata['shape'])
        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        if image.max() != 0:
            image = image.astype(np.float32) / image.max()
        return image

    def load_annotations(self, image_index):
        """
        Takes an image from the data_list by image_index, and loads it depending on
        whether there is a lesion, etc.
        :param image_index: index of the image in self.data_list
        :return: annot, slice_idx, lesions
        """

        # if there is no lesion for this image
        if len(self.data_list[image_index]['lesions']) == 0:
            annotations = np.zeros((0, 5))
            lesions = []
            if self.load_center_slice_negative:
                slice_idx = self.data_list[image_index]['shape'][0] // 2
            else:
                slice_idx = random.randint(0, self.data_list[image_index]['shape'][0] - 1)

        else:
            if self.random_patch_selection:
                sampled_lesion = random.choice(self.data_list[image_index]['lesions'])
            else:
                sampled_lesion = self.data_list[image_index]['lesions'][0]
            slice_idx_sampled_lesion = sampled_lesion['Slice']

            # parse annotations
            # Select all lesions which appear at the current slice
            lesions = []
            for idx, lesion in enumerate(self.data_list[image_index]['lesions']):

                x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
                # some annotations have basically no width / height, skip them
                if w <= 1 or h <= 1:
                    continue

                # if the center of patch appears within +- self.slice_neighbor_sampling slices, extract them.
                if (slice_idx_sampled_lesion > lesion['Slice']) or \
                        (lesion['Slice'] > slice_idx_sampled_lesion):
                    continue

                lesions.append(lesion)

            # use V2 function to load combined_label as lesion label
            annotations = get_annotations_from_lesions_V2(lesions)

            # disable sampling slice_idx from neighbors
            slice_idx = min(max(0, slice_idx_sampled_lesion), self.data_list[image_index]['shape'][0] - 1)
        if 'classification_labels' in self.data_list[image_index]:
            clf_label = self.data_list[image_index]['classification_labels']
        else:
            clf_label = None

        return annotations, slice_idx, lesions, clf_label


class DukeDatasetV4(DukeDatasetV2):
    """
    This dataset is practically the same as V2, but loads all slices in each volume.
    They will be loaded as batches in the end, but they may be treated as channels in the intermediate processing.
    
    """

    def __init__(self, root_dir, set='train2017', transform=None, filter_negatives=False, apply_window=False, random_patch_selection=False, fraction=1,
                 load_segmentation=False, seg_dir=None, augment_boxes_without_segmentation=False, sample_wrt_class=False, **kwargs):
        """
        random_patch_selection will no longer be used, since all lesions will be loaded
        """
        super().__init__(
            root_dir=root_dir, set=set, transform=transform, filter_negatives=filter_negatives,
            apply_window=apply_window, random_patch_selection=random_patch_selection,
            fraction=fraction, load_segmentation=load_segmentation, seg_dir=seg_dir,
            augment_boxes_without_segmentation=augment_boxes_without_segmentation,
            load_center_slice_negative=kwargs.get('load_center_slice_negative', False),
            neighboring_slices_training=kwargs.get('neighboring_slices_training', 0),
            phase=kwargs.get('phase', None),
            mixup=kwargs.get('mixup', False),
            sample_wrt_class=sample_wrt_class,
        )

    def lr_flip_image_annot(self, idx, image, annots, view, seg_dict=None):
        """
        check for horizontal_flip bit to determine whether or not to flip the images and annotations
        """
        horizontal_flip = self.data_list[idx]['horizontal_flip']

        if horizontal_flip == 'NO':
            if view[0] == 'r':
                image, annots, seg_dict = hflip_image_annot_seg_v4(image, annots, seg_dict)
        elif horizontal_flip == 'YES':
            if view[0] == 'l':
                image, annots, seg_dict = hflip_image_annot_seg_v4(image, annots, seg_dict)
        else:
            raise KeyError(horizontal_flip)

        return image, annots, seg_dict

    def load_all_annotations(self, image_index):
        """
        Takes an image from the data_list by image_index, and loads it depending on
        whether there is a lesion, etc.
        :param image_index: index of the image in self.data_list
        :return: annot, slice_idx, lesions
        """
        image_metadata = self.data_list[image_index]

        # parse annotations
        # Select all lesions which appear at the current slice
        lesions = []
        for idx, lesion in enumerate(image_metadata['lesions']):
            x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
            # some annotations have basically no width / height, skip them
            if w <= 1 or h <= 1:
                continue

            lesions.append(lesion)

        # use V4 function to load combined_label as lesion label
        annotations = get_annotations_from_lesions_V4(lesions, image_metadata['shape'][0])

        return annotations, lesions

    def load_3d_image(self, image_index):
        # Override the segmentation loading method
        # flip bit is deprecated, but new horizontal_flip is introduced

        # load all slices
        image_metadata = self.data_list[image_index]

        if image_metadata['which_dataset'] < 2:
            image = reading_images.read_image_3d_duke(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                ),
                start=0,
                end=image_metadata['shape'][0],
                flip=False,  # flip is always False for loader V2
            )
        else:
            raise KeyError(image_metadata['which_dataset'])

        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        if image.max() != 0:
            image = image.astype(np.float32) / image.max()
        return image

    @staticmethod
    def create_segmentation_from_box(class_lesions, image_shape):
        """
        In V2, this function creates one slice with segmentation
        In V4, however, this creates the entire volume with full segmentation
        """
        d, h, w = image_shape
        result = np.zeros((d, h, w), dtype=np.uint8)
        for lesion in class_lesions:
            x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
            x2 = x + w
            y2 = y + h
            result[lesion['Slice'], y:y2, x:x2] = 1  # currently loading each segmentation file separately
        return result

    def load_3d_segmentation_func(self, image_index, lesions):
        """
        slice_index is no longer used
        """
        # Override the segmentation loading method
        # now keys are the combined_label of the lesion
        # the segmentation suffixes are different between datasets
        image_metadata = self.data_list[image_index]

        # find out all lesion labels for the current slide
        lesion_classes_dict = self.get_label_class_dict(lesions)

        seg_dict = {}

        for lesion_class, class_lesions in lesion_classes_dict.items():

            seg_filename = os.path.join(
                self.seg_dir,
                f"{image_metadata.get('short_file_path', '')}.{self.lesion_class_to_seg_suffix_dict[lesion_class]}.hdf5"
            )
            if os.path.exists(seg_filename):
                # load one slice of segmentation with a lesion
                if image_metadata['which_dataset'] < 2:
                    seg_dict[lesion_class] = reading_images.read_image_3d_duke(
                        file_name=seg_filename,
                        start=0,
                        end=image_metadata['shape'][0],
                        flip=False,  # flip is False for segmentations
                    ).astype(np.uint8)  # should already be uint8, but check
                else:
                    raise KeyError(image_metadata['which_dataset'])

            elif self.augment_boxes_without_segmentation:
                seg_dict[lesion_class] = self.create_segmentation_from_box(class_lesions, image_metadata['shape'])

            # nothing to transpose if I chose not to load the segmentation
            if lesion_class in seg_dict:
                seg_dict[lesion_class] = np.transpose(seg_dict[lesion_class], (1, 2, 0))  # channel dimension at the end
        return seg_dict

    # TODO: fix this function
    def __getitem__(self, idx):

        # load all annotations
        annot, lesions = self.load_all_annotations(idx)

        # load segmentation, if required to
        seg_dict = self.load_3d_segmentation_func(idx, lesions) if self.load_segmentation else None

        # load the entire 3D image
        img = self.load_3d_image(idx)

        # make all breasts face left direction
        img, annot_t, seg_dict_t = self.lr_flip_image_annot(idx, img, annot, self.data_list[idx]['View'], seg_dict)

        if self.mixup:
            raise NotImplementedError('Mixup for 3D loader not implemented')

        sample = {
            'img': img, 'annot': annot_t, 'seg_dict': seg_dict_t,
            'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
            'best_center': self.data_list[idx]['best_center'],
            'view': self.data_list[idx]['View'],
        }

        if self.transform:
            sample = self.transform(sample)

        return {**sample, 'index': idx}


class DukeDatasetV5(DukeDatasetV4):
    """
    This dataset version supports loading volumes as 3d to enable det+clf training.
    """

    def __init__(self, root_dir, set='train2017', transform=None, filter_negatives=False, apply_window=False, random_patch_selection=False, fraction=1,
                 load_segmentation=False, seg_dir=None, augment_boxes_without_segmentation=False, mode='det', k_extra_slices=0, sample_wrt_class=False,
                 **kwargs):
        super().__init__(
            root_dir=root_dir, set=set, transform=transform, filter_negatives=filter_negatives,
            apply_window=apply_window, random_patch_selection=random_patch_selection,
            fraction=fraction, load_segmentation=load_segmentation, seg_dir=seg_dir,
            augment_boxes_without_segmentation=augment_boxes_without_segmentation,
            load_center_slice_negative=kwargs.get('load_center_slice_negative', False),
            neighboring_slices_training=kwargs.get('neighboring_slices_training', 0),
            phase=kwargs.get('phase', None),
            mixup=kwargs.get('mixup', False),
            sample_wrt_class=sample_wrt_class,
        )

        self.mode = mode
        self.k_extra_slices = k_extra_slices
        if k_extra_slices:
            assert self.mode in ['detclf', 'detawa'], "Loading more than 1 slice only supported in the detclf/detawa mode."

    def __getitem__(self, idx):
        # assert self.mode in ['detclf', 'detawa'], "Only ['detclf', 'detawa'] modes are supported with DukeDatasetV5 so far."
        assert not self.neighboring_slices_training, "DukeDatasetV5 does not support neighboring slices training so far."

        # do this to get the idx of the chosen slice and label for classification
        _, slice_idx, _, clf_label = self.load_annotations(idx)

        image_metadata = self.data_list[idx]
        img_shape = image_metadata['shape']
        assert img_shape[0] > 1, "DukeDatasetV5 does not support 2D FFDM images, looks like one has been just loaded."

        # choose indices of at most k before and at most k after the slice_idx
        idx_list = [k for k in range(slice_idx - self.k_extra_slices, slice_idx + self.k_extra_slices + 1) if 0 <= k < img_shape[0]]

        # load all annotations
        if self.phase != 'inference':
            annot, lesions = self.load_multiple_annotations(idx, idx_list)
        else:
            annot, lesions = None, None
        # annot, lesions = self.load_all_annotations(idx)

        # print(0, idx_list)
        # load segmentation, if required to
        seg_dict = self.load_3d_segmentation_func(idx, lesions) if self.load_segmentation else None
        if seg_dict is not None:
            for key in seg_dict.keys():
                # print(idx_list)
                seg_dict[key] = seg_dict[key][:, :, idx_list]

        # load the entire 3D image
        img = self.load_3d_image(idx, idx_list)

        # make all breasts face left direction
        img, annot_t, seg_dict_t = self.lr_flip_image_annot(idx, img, annot, self.data_list[idx]['View'], seg_dict)

        if self.mixup:
            raise NotImplementedError('Mixup for 3D loader not implemented')
        sample = {
            'img': img, 'annot': annot_t, 'seg_dict': seg_dict_t,  # TODO is this ok for transform?
            'which_dataset': self.data_list[idx]['which_dataset'],  # additional metadata to indicate which dataset
            'best_center': self.data_list[idx]['best_center'],
            'view': self.data_list[idx]['View'],
            'idx_list': idx_list, 'det_slice_idx': slice_idx,
            'index': idx, 'clf_label': clf_label
        }
        if self.transform:
            sample = self.transform(sample)

        if self.phase != 'inference':
            sample['annot'] = sample['annot'][idx_list.index(slice_idx)]

        return sample

    def load_3d_image(self, image_index, idx_list):
        # Override the segmentation loading method
        # flip bit is deprecated, but new horizontal_flip is introduced

        # load all slices
        image_metadata = self.data_list[image_index]

        if image_metadata['which_dataset'] < 2:
            image = reading_images.read_image_3d_duke(
                file_name=os.path.join(
                    self.root_dir,
                    image_metadata['hdf5_path']
                ),
                start=idx_list[0],
                end=idx_list[-1] + 1,
                flip=False,  # flip is always False for loader V2
            )
        else:
            raise KeyError(image_metadata['which_dataset'])

        image = np.transpose(image, (1, 2, 0))  # channel dimension at the end
        if image.max() != 0:
            image = image.astype(np.float32) / image.max()
        return image

    def load_multiple_annotations(self, image_index, idx_list):
        """
        Takes an image from the data_list by image_index, and loads it depending on
        whether there is a lesion, etc.
        :param image_index: index of the image in self.data_list
        :return: annot, slice_idx, lesions
        """
        image_metadata = self.data_list[image_index]

        # parse annotations
        # Select all lesions which appear at the current slice
        lesions = []
        for idx, lesion in enumerate(image_metadata['lesions']):
            if lesion['Slice'] in idx_list:
                x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
                # some annotations have basically no width / height, skip them
                if w <= 1 or h <= 1:
                    continue

                lesions.append(lesion)

        # use V4 function to load combined_label as lesion label
        annotations = get_annotations_from_lesions_V4(lesions, image_metadata['shape'][0])
        annotations = [annotations[i] for i in idx_list]

        return annotations, lesions

    def load_annotations(self, image_index):
        """
        Takes an image from the data_list by image_index, and loads it depending on
        whether there is a lesion, etc.
        :param image_index: index of the image in self.data_list
        :return: annot, slice_idx, lesions
        """

        # if there is no lesion for this image
        if len(self.data_list[image_index]['lesions']) == 0:
            annotations = np.zeros((0, 5))
            lesions = []
            if self.load_center_slice_negative:
                slice_idx = self.data_list[image_index]['shape'][0] // 2
            else:
                slice_idx = random.randint(0, self.data_list[image_index]['shape'][0] - 1)

        else:
            if self.random_patch_selection:
                sampled_lesion = random.choice(self.data_list[image_index]['lesions'])
            else:
                sampled_lesion = self.data_list[image_index]['lesions'][0]
            slice_idx_sampled_lesion = sampled_lesion['Slice']

            # parse annotations
            # Select all lesions which appear at the current slice
            lesions = []
            for idx, lesion in enumerate(self.data_list[image_index]['lesions']):

                x, y, w, h = lesion['X'], lesion['Y'], lesion['Width'], lesion['Height']
                # some annotations have basically no width / height, skip them
                if w <= 1 or h <= 1:
                    continue

                # if the center of patch appears within +- self.slice_neighbor_sampling slices, extract them.
                if (slice_idx_sampled_lesion > lesion['Slice']) or \
                        (lesion['Slice'] > slice_idx_sampled_lesion):
                    continue

                lesions.append(lesion)

            # use V2 function to load combined_label as lesion label
            annotations = get_annotations_from_lesions_V2(lesions)

            # disable sampling slice_idx from neighbors
            slice_idx = min(max(0, slice_idx_sampled_lesion), self.data_list[image_index]['shape'][0] - 1)

        if 'classification_labels' in self.data_list[image_index]:
            clf_label = self.data_list[image_index]['classification_labels']
        else:
            clf_label = None

        return annotations, slice_idx, lesions, clf_label

YOLOX_ORDER_CLF_LABELS = 'malben' # save this as global variableto match with the updated ordering of labels

def get_yolox_annot(annot, num_labels):
    # shape of annot: [batch size, num lesions, [x1, y1, x2, y2, c]]
    # target annot: [batch size, num lesions, [class, xc, yc, w, h]]
    # target missing annot: 0 values everywhere, rather than using -1

    # step 1: set all missing values to 0
    annot[annot == -1] = 0

    # get lesion locations and classes
    x1 = annot[:, :, 0:1]
    y1 = annot[:, :, 1:2]
    x2 = annot[:, :, 2:3]
    y2 = annot[:, :, 3:4]
    lesion_class = annot[:, :, 4:]

    # get classes (this is only a temporary measure)
    if num_labels == 1:
        lesion_class[lesion_class >= 0] = 0
    elif num_labels == 2:
        # reduce labels to 0 or 1
        # benign: 0, malignant: 1
        # this is consistent with the way I load cls_labels
        # works with 10-class combined-label scheme (label_version==2)
        # 4-classes in Duke
        lesion_class[lesion_class == 0] = 0  # no longer need to swap with this ordering
        lesion_class[lesion_class == 1] = 1
        lesion_class[lesion_class == 2] = 0
        lesion_class[lesion_class == 3] = 1

        # NYU DBT
        lesion_class[lesion_class == 4] = 0
        lesion_class[lesion_class == 5] = 1
        lesion_class[lesion_class == 6] = 0
        lesion_class[lesion_class == 7] = 1

        # NYU FFDM
        lesion_class[lesion_class==8] = 0
        lesion_class[lesion_class==9] = 1
    elif num_labels == 6:
        # reduce labels to 0 or 1
        # works with 10-class combined-label scheme (label_version==2)
        # 4-classes in Duke
        lesion_class[lesion_class==0] = 0 # temporary swap value
        lesion_class[lesion_class==1] = 1
        lesion_class[lesion_class==2] = 2
        lesion_class[lesion_class==3] = 3
        
        # NYU DBT
        lesion_class[lesion_class==4] = 0
        lesion_class[lesion_class==5] = 1
        lesion_class[lesion_class==6] = 2
        lesion_class[lesion_class==7] = 3
        
        # NYU FFDM
        lesion_class[lesion_class==8] = 4
        lesion_class[lesion_class==9] = 5

    else:
        raise KeyError(num_labels)

    # calculate centers, width and height
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    return torch.cat([lesion_class, x_center, y_center, width, height], dim=2)


def get_yolox_clf_label(clf_label, num_labels):
    result = []
    for each in clf_label:
        if num_labels == 1:
            result.append([max(each['mal'], each['ben'])])
        elif num_labels in {2, 6}:
            # TODO: MULTIMODAL_DETECTION, revisit, don't take 6 here
            result.append([each['mal'], each['ben']])
        else:
            raise KeyError(num_labels)

    return torch.LongTensor(result)


def get_yolox_output(result_dict, num_labels):
    image = result_dict['img']
    annot = result_dict['annot']
    index = result_dict['index']

    clf_label = result_dict['clf_label']

    bs = image.shape[0]

    yolox_annot = get_yolox_annot(annot, num_labels)
    yolox_clf_label = get_yolox_clf_label(clf_label, num_labels)

    return image.contiguous(), yolox_annot.contiguous(), yolox_clf_label.contiguous(), [torch.LongTensor([2116] * bs),
                                                                                        torch.LongTensor([1339] * bs)], torch.LongTensor(index)


def collater_detclf(data, yolox_num_labels=False):
    batch_idx = [i for i, s in enumerate(data) for _ in s['img']]
    batch_idx_temp = []
    num = 0
    for i, idx in enumerate(batch_idx):
        if num == idx:
            batch_idx_temp.append(i)
            num += 1
    batch_idx = batch_idx_temp
    imgs = [im for s in data for im in s['img']]
    annots = [s['annot'] for s in data]

    if 'scale' in data[0]:
        scales = [s['scale'] for s in data]
    else:
        scales = None

    if 'framed_meta' in data[0]:
        framed_meta = [s['framed_meta'] for s in data]
    else:
        framed_meta = None

    annot_padded = concat_pad_annots(annots)

    imgs = torch.stack(imgs)
    if len(imgs.shape) == 3:  # if there is no channel dim, add it (later it will be repeated 3 times)
        imgs = imgs.unsqueeze(1)
    else:  # if there is, it is most likely at the end, therefore, permute to move it to 1st dim
        imgs = imgs.permute(0, 3, 1, 2)

    result_dict = {'img': imgs, 'annot': annot_padded, 'scale': scales, 'framed_meta': framed_meta}
    # handle optional arguments: index, best_center, which_dataset, seg_dict
    if 'index' in data[0]:
        result_dict['index'] = [s['index'] for s in data]
    if 'best_center' in data[0]:
        result_dict['best_center'] = [s['best_center'] for s in data]
    if 'which_dataset' in data[0]:
        result_dict['which_dataset'] = [s['which_dataset'] for s in data]
    if 'idx_list' in data[0]:
        result_dict['idx_list'] = [s['idx_list'] for s in data]
    if 'det_slice_idx' in data[0]:
        result_dict['det_slice_idx'] = [s['det_slice_idx'] for s in data]
    if 'clf_label' in data[0]:
        if data[0]['clf_label'] is not None:
            result_dict['clf_label'] = [s['clf_label'] for s in data]
        else:
            result_dict['clf_label'] = None
    if 'original_image_size' in data[0]:
        result_dict['original_image_size'] = [s['original_image_size'] for s in data]

    # mixup example
    if 'mixup_sample' in data[0]:
        mixup_sample = {
            'img': [s['mixup_sample']['img'] for s in data],
            'annot': [s['mixup_sample']['annot'] for s in data],
            'seg_dict': [s['mixup_sample']['seg_dict'] for s in data],
        }
        result_dict['mixup_sample'] = mixup_sample

    result_dict['batch_idx'] = batch_idx
    result_dict['seg_dict'] = [s.get('seg_dict', None) for s in data]

    if yolox_num_labels:
        return get_yolox_output(result_dict, num_labels=yolox_num_labels)

    return result_dict


def collater(data, is_loader_v4=False, is_loader_v5=False, return_vol_seg=False, yolox_num_labels=False):
    if is_loader_v5:
        batch_idx = [i for i, s in enumerate(data) for _ in range(s['img'].shape[2])]
        batch_idx_temp = []
        num = 0
        for i, idx in enumerate(batch_idx):
            if num == idx:
                batch_idx_temp.append(i)
                num += 1
        batch_idx = batch_idx_temp
    else:
        batch_idx = [0]
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    if 'scale' in data[0]:
        scales = [s['scale'] for s in data]
    else:
        scales = None

    if 'framed_meta' in data[0]:
        framed_meta = [s['framed_meta'] for s in data]
    else:
        framed_meta = None

    if not is_loader_v5:
        imgs = torch.stack(imgs).permute(0, 3, 1, 2)

    if is_loader_v4:
        # switch the channel and batch dimension
        imgs = imgs.permute(1, 0, 2, 3)
        # imgs = imgs.repeat(1, 3, 1, 1) # repeating in the CPU side takes up too much RAM. Do it in forward_data
        # annots are already merged and padded in ResizerV4
        annot_padded = annots[0]
    elif is_loader_v5:
        imgs = torch.cat(imgs, dim=2).permute(2, 0, 1).unsqueeze(1)
        annot_padded = concat_pad_annots(annots)
    else:
        annot_padded = concat_pad_annots(annots)

    result_dict = {'img': imgs, 'annot': annot_padded, 'scale': scales, 'framed_meta': framed_meta}

    # handle optional arguments: index, best_center, which_dataset, seg_dict
    if 'index' in data[0]:
        result_dict['index'] = [s['index'] for s in data]
    if 'best_center' in data[0]:
        result_dict['best_center'] = [s['best_center'] for s in data]
    if 'which_dataset' in data[0]:
        result_dict['which_dataset'] = [s['which_dataset'] for s in data]
    if is_loader_v4:
        if 'index' in data[0]:
            result_dict['index'] = result_dict['index'] * imgs.shape[0]
        if 'best_center' in data[0]:
            result_dict['best_center'] = result_dict['best_center'] * imgs.shape[0]
        if 'which_dataset' in data[0]:
            result_dict['which_dataset'] = result_dict['which_dataset'] * imgs.shape[0]
    if is_loader_v5:
        if 'idx_list' in data[0]:
            result_dict['idx_list'] = [s['idx_list'] for s in data]
        if 'det_slice_idx' in data[0]:
            result_dict['det_slice_idx'] = [s['det_slice_idx'] for s in data]
    if 'clf_label' in data[0]:
        if data[0]['clf_label'] is not None:
            result_dict['clf_label'] = [s['clf_label'] for s in data]
        else:
            result_dict['clf_label'] = None
    if 'original_image_size' in data[0]:
        result_dict['original_image_size'] = [s['original_image_size'] for s in data]
    result_dict['batch_idx'] = batch_idx

    # mixup example
    if 'mixup_sample' in data[0]:
        mixup_sample = {
            'img': [s['mixup_sample']['img'] for s in data],
            'annot': [s['mixup_sample']['annot'] for s in data],
            'seg_dict': [s['mixup_sample']['seg_dict'] for s in data],
        }
        result_dict['mixup_sample'] = mixup_sample

    if (is_loader_v4 or is_loader_v5) and (not return_vol_seg):
        # if volume-wise loader, explicitly turn off returning segmentations from the data loaders
        result_dict['seg_dict'] = [None for s in data]
    else:
        result_dict['seg_dict'] = [s.get('seg_dict', None) for s in data]

    if yolox_num_labels:
        # TODO: get num_labels separately
        return get_yolox_output(result_dict, num_labels=yolox_num_labels)

    return result_dict


class Resizer(object):
    """
    Resize the height of the image to the size
    corresponding to the compound_coef of current EfficientDet architecture.

    Unlike COCO dataset, we know that height is longer, and is a certain ratio of width.
    We can use this to use smaller padding to save memory -> width_dict
    """

    def __init__(self, img_size=512):
        self.img_size = img_size
        self.width_dict = {
            512: 512,  # closest multiple of 128 larger than 415.93
            640: 640,  # closest multiple of 128 larger than 519.91
            768: 768,  # closest multiple of 128 larger than 623.90, but bump it up due to augmentation
            896: 768,  # closest multiple of 128 larger than 727.88
            1024: 896,  # closest multiple of 128 larger than 831.869
            1152: 1024,  # closest multiple of 128 larger than 935
            1280: 1152,  # closest multiple of 128 larger than 1039.83
            1408: 1280,  # closest multiple of 128 larger than 1143, but bump it up due to augmentation
            1536: 1408,  # closest multiple of 128 larger than 1247.80, but bump it up due to augmentation
            1664: 1408,  # closest multiple of 128 larger than 1664*1790/2257=1319.7
            1792: 1536,  # closest multiple of 128 larger than x*1790/2257=1421.2
            1920: 1536,  # closest multiple of 128 larger than x*1790/2257=1522.7
            2048: 1664,  # closest multiple of 128 larger than x*1790/2257=1624.2
            2176: 1792,  # closest multiple of 128 larger than x*1790/2257=1725.7
            2304: 1920,  # closest multiple of 128 larger than x*1790/2257=1827.7
            2432: 2048,  # closest multiple of 128 larger than x*1790/2257=1928.7
            2560: 2048,  # closest multiple of 128 larger than x*1790/2257=2030.3
        }

    def resize(self, sample: dict):
        image, annots, segs = sample['img'], sample.get('annot', None), sample.get('seg_dict', None)

        height, width, num_slices = image.shape

        # we know that height is larger
        scale = self.img_size / height
        resized_height = self.img_size
        resized_width = int(width * scale)

        # Resize & pad image
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        new_image = np.zeros((self.img_size, self.width_dict[self.img_size], num_slices))
        if num_slices == 1:
            new_image[0:resized_height, 0:resized_width, 0] = image
        else:
            new_image[0:resized_height, 0:resized_width, :] = image

        # Resize & pad segmentations
        if segs is not None:
            for seg_class, seg in segs.items():
                if seg.ndim == 2:
                    seg = cv2.resize(seg, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
                elif seg.ndim == 3:
                    seg_intermediate = np.zeros((resized_height, resized_width, seg.shape[-1]))
                    for i in range(seg.shape[-1]):
                        seg_intermediate[..., i] = cv2.resize(seg[..., i], (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
                else:
                    raise ValueError(f"Number of dimensions in segmentation {seg.ndim} not handled")

                if seg.ndim == 2:
                    new_seg = np.zeros((self.img_size, self.width_dict[self.img_size], 1))
                    new_seg[0:resized_height, 0:resized_width, :] = seg_intermediate
                elif seg.ndim == 3:
                    new_seg = np.zeros((self.img_size, self.width_dict[self.img_size], seg.shape[-1]))
                    new_seg[0:resized_height, 0:resized_width, :] = seg_intermediate

                sample['seg_dict'][seg_class] = new_seg

        annots = self.adjust_annots(scale, annots)

        canvas_height, canvas_width = new_image.shape[:2]

        padding_h = canvas_height - resized_height
        padding_w = canvas_width - resized_width

        sample['img'] = new_image
        if annots is not None:
            sample['annot'] = annots
        sample['scale'] = scale
        sample['framed_meta'] = (resized_width, resized_height, width, height, padding_w, padding_h)
        return sample

    def adjust_annots(self, scale, annots):
        # handle test phase
        if isinstance(annots, np.ndarray):
            if annots.size == 0:
                return annots
        else:
            if not annots:
                return annots
        if annots is not None:
            annots[:, :4] *= scale
        return annots

    def __call__(self, sample: dict):
        sample = self.resize(sample)
        if 'mixup_sample' in sample:
            mixup_sample = sample['mixup_sample']
            mixup_sample = self.resize(mixup_sample)
            sample['mixup_sample'] = mixup_sample
        return sample


class ResizerV2(Resizer):
    def __init__(self, img_size=512):
        super().__init__(img_size=img_size)
        # worst ratio is (2116-200)x(1339-100)
        self.width_dict = {
            512: 384,  # closest multiple of 128 larger than 331.1
            640: 512,  # closest multiple of 128 larger than 413.1
            768: 512,  # closest multiple of 128 larger than 496.6
            896: 640,  # closest multiple of 128 larger than 579.4
            1024: 768,  # closest multiple of 128 larger than 662.1
            1152: 768,  # closest multiple of 128 larger than 744.9
            1280: 896,  # closest multiple of 128 larger than 827.7
            1408: 1024,  # closest multiple of 128 larger than 910.4
            1536: 1024,  # closest multiple of 128 larger than 993.2
            1664: 1152,  # closest multiple of 128 larger than 1076.04
            1792: 1280,  # closest multiple of 128 larger than 1158.8
            1920: 1280,  # closest multiple of 128 larger than 1241.6
            2048: 1408,  # closest multiple of 128 larger than 1324.4
            2176: 1408,  # closest multiple of 128 larger than 1407.1
        }


class ResizerV4(ResizerV2):
    def __init__(self, img_size=512):
        super().__init__(img_size=img_size)

    def adjust_annots(self, scale, annots_list):
        # handle test phase
        if annots_list is not None:
            for annots in annots_list:
                annots[:, :4] *= scale
        return concat_pad_annots_numpy(annots_list)


def _window_location_at_center_point(input_size, center_y, center_x):
    """
    calculate window location w.r.t. center point (can be outside of image)
    Returns:
    border (4 integers)
    """
    half_height = input_size[0] // 2
    half_width = input_size[1] // 2
    top = center_y - half_height
    bottom = center_y + input_size[0] - half_height
    left = center_x - half_width
    right = center_x + input_size[1] - half_width
    return top, bottom, left, right


def _zero_pad_and_align_window(image_axis_size, input_axis_size,
                               max_crop_and_size_noise, bidirectional):
    """
    if the image is small, calculate padding and align window accordingly
    We made sure pad_width is positive, and after padding,
    there will be room for window to move as much as max_crop_noise
    Returns:
    new start and end indices, padding amount for front and back of this axis
    """
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


def _shift_window_inside_image(start, end, image_axis_size, input_axis_size):
    """
    if the window is drawn to be outside of the image, shift it to be inside
    Returns:
    new start and end indices
    """
    if start < 0:
        start = 0
        end = start + input_axis_size
    elif end > image_axis_size:
        end = image_axis_size
        start = end - input_axis_size

    return start, end


def pad_zeros(image, pad_y_top, pad_y_bottom, pad_x_right):
    new_zero_array = np.zeros(
        (
            image.shape[0] + pad_y_top + pad_y_bottom,
            image.shape[1] + pad_x_right,
            image.shape[2]
        ),
        dtype=image.dtype
    )
    new_zero_array[pad_y_top: image.shape[0] + pad_y_top, \
    0: image.shape[1]] = image
    return new_zero_array


class V2InputSizeAdjuster(object):
    def __init__(self, mode, dbt_input_size=(2116, 1339), ffdm_input_size=(2866, 1814), ffdm_additional_input_size=None):
        r"""
        Currently supported mode: "train", "val"
        if the image is larger than input size,
          - "train" mode:
          - "val" mode: crop around the most important center point
              - the window drawn from this center point might lie outside of the image
              - (sometimes even the center point itself has negative indices)
              - this is expected. We simply push the input window to inside the padded image.
        """
        assert mode in ('train', 'val')
        self.mode = mode
        self.dbt_input_size = dbt_input_size
        self.ffdm_input_size = ffdm_input_size
        self.ffdm_additional_input_size = ffdm_additional_input_size

    def adjust_size(self, sample: dict):
        image, annots, segs = sample['img'], sample.get('annot', None), sample.get('seg_dict', None)

        view, best_center = sample['view'], sample['best_center']
        if sample['which_dataset'] < 2 or sample['which_dataset'] == 3:
            input_size = self.dbt_input_size
        else:
            # assume FFDM (which_dataset == 2)
            if self.ffdm_additional_input_size and (sample['original_image_size'][-2:] == (4096, 3328)):
                # if (1) additional input size is provided
                # and (2) the current image has the original resolution where the additional input size should be applied
                # use the alternative input size
                input_size = self.ffdm_additional_input_size
            else:
                input_size = self.ffdm_input_size

        height, width, num_slices = image.shape

        # use some portions of logic from BreastScreening data augmenter
        center_y, center_x = best_center

        top, bottom, left, right = _window_location_at_center_point(
            input_size, center_y, center_x
        )

        pad_y_top, pad_y_bottom, pad_x_right = 0, 0, 0

        if image.shape[0] < input_size[0]:
            if view[1].lower() == 'c':
                top, bottom, pad_y_top, pad_y_bottom = _zero_pad_and_align_window(
                    image.shape[0],
                    input_size[0],
                    0,
                    True
                )
            elif view[1].lower() == 'm':
                top, bottom, _, pad_y_bottom = _zero_pad_and_align_window(
                    image.shape[0],
                    input_size[0],
                    0,
                    False
                )
            else:
                raise KeyError(view)

        if image.shape[1] < input_size[1]:
            left, right, _, pad_x_right = _zero_pad_and_align_window(
                image.shape[1],
                input_size[1],
                0,
                False
            )

        # if the image is smaller in x dim, pad in x dim
        # if the image is smaller in y dim, pad in y dim
        if pad_y_top > 0 or pad_y_bottom > 0 or pad_x_right > 0:
            image = pad_zeros(image, pad_y_top, pad_y_bottom, pad_x_right)
            if segs is not None:
                for seg_class, seg in segs.items():
                    sample['seg_dict'][seg_class] = pad_zeros(
                        sample['seg_dict'][seg_class],
                        pad_y_top, pad_y_bottom, pad_x_right
                    )
            annots = self.pad_annots(pad_y_top, annots)

        # if window is drawn outside of image, shift it to be inside the image.
        top, bottom = _shift_window_inside_image(
            top, bottom, image.shape[0], input_size[0]
        )
        left, right = _shift_window_inside_image(
            left, right, image.shape[1], input_size[1]
        )
        top, bottom, left, right = int(top), int(bottom), int(left), int(right)

        if self.mode == "train":
            # randomize the window location ignoring the previously calculated best_center window
            top = random.randint(0, image.shape[0] - input_size[0])
            left = random.randint(0, image.shape[1] - input_size[1])
            bottom = top + input_size[0]
            right = left + input_size[1]

        # crop the input_size window from the padded image
        image = image[top:bottom, left:right]
        if segs is not None:
            for seg_class, seg in segs.items():
                sample['seg_dict'][seg_class] = sample['seg_dict'][seg_class][top:bottom, left:right]

        annots = self.adjust_annots(left, top, image, annots)

        sample['img'] = image
        if annots is not None:
            sample['annot'] = annots
        sample['adjuster_window_location'] = (top, bottom, left, right)
        sample['adjuster_padding'] = (pad_y_top, pad_y_bottom, pad_x_right)
        return sample

    def pad_annots(self, pad_y_top, annots):
        if annots is not None:
            annots[:, 1] += pad_y_top
            annots[:, 3] += pad_y_top
        return annots

    def adjust_annots(self, left, top, image, annots):
        if annots is not None:
            annots[:, 0] -= left
            annots[:, 1] -= top
            annots[:, 2] -= left
            annots[:, 3] -= top

            # TODO: if either x1 or x2 is out-of-bounds, set it to border value.
            # TODO: if both are out-of-bounds, remove from annots array
            new_annots = []
            for annot in annots:
                if (annot[0] >= image.shape[1]) or (annot[1] >= image.shape[0]) or (annot[2] < 0) or (annot[3] < 0):
                    # this lesion is completely out_of_bounds
                    continue

                # if the lesion is on the border of image, confine lesion within the image
                annot[0] = max(0, annot[0])
                annot[1] = max(0, annot[1])
                annot[2] = min(image.shape[1], annot[2])
                annot[3] = min(image.shape[0], annot[3])
                new_annots.append(annot)
            if len(new_annots) > 0:
                annots = np.array(new_annots)
            else:
                annots = np.zeros((0, 5))
        return annots

    def __call__(self, sample: dict):
        sample = self.adjust_size(sample)
        if 'mixup_sample' in sample:
            mixup_sample = sample['mixup_sample']
            mixup_sample = self.adjust_size(mixup_sample)
            sample['mixup_sample'] = mixup_sample
        return sample


class V4InputSizeAdjuster(V2InputSizeAdjuster):
    def __init__(self, mode, dbt_input_size=(2116, 1339), ffdm_input_size=(2866, 1814)):
        super().__init__(
            mode=mode,
            dbt_input_size=dbt_input_size,
            ffdm_input_size=ffdm_input_size,
        )

    def pad_annots(self, pad_y_top, annots_list):
        if annots_list is not None:
            for annots in annots_list:
                annots[:, 1] += pad_y_top
                annots[:, 3] += pad_y_top
        return annots_list

    def adjust_annots(self, left, top, image, annots_list):
        final_annots_list = []
        if annots_list is not None:
            for original_annots in annots_list:

                annots = original_annots.copy()

                annots[:, 0] -= left
                annots[:, 1] -= top
                annots[:, 2] -= left
                annots[:, 3] -= top

                # TODO: if either x1 or x2 is out-of-bounds, set it to border value.
                # TODO: if both are out-of-bounds, remove from annots array
                new_annots = []
                for annot in annots:
                    if (annot[0] >= image.shape[1]) or (annot[1] >= image.shape[0]) or (annot[2] < 0) or (annot[3] < 0):
                        # this lesion is completely out_of_bounds
                        continue

                    # if the lesion is on the border of image, confine lesion within the image
                    annot[0] = max(0, annot[0])
                    annot[1] = max(0, annot[1])
                    annot[2] = min(image.shape[1], annot[2])
                    annot[3] = min(image.shape[0], annot[3])
                    new_annots.append(annot)

                if len(new_annots) > 0:
                    annots = np.array(new_annots)
                else:
                    annots = np.zeros((0, 5))
                final_annots_list.append(annots)
        return final_annots_list


def numpy_to_pil(img=None, seg_dict=None):
    if img is not None:
        if img.ndim == 3:
            img = Image.fromarray(img[:, :, 0])
        elif img.ndim == 2:
            img = Image.fromarray(img)
        else:
            raise ValueError(f"Number of dimensions {img.ndim} not handled")

    if seg_dict is not None:
        for k in seg_dict.keys():
            seg_dict[k] = Image.fromarray(seg_dict[k][:, :, 0])
        return img, seg_dict
    return img


def pil_to_numpy(img, seg_dict=None):
    if type(img) == list:
        img_new = np.zeros((img[0].size[1], img[0].size[0], len(img)))
        for i in range(len(img)):
            img_new[..., i] = np.array(img[i])
        img = img_new
    else:
        img = np.expand_dims(np.array(img), axis=-1)

    if seg_dict is not None:
        for k in seg_dict.keys():
            seg_dict[k] = np.expand_dims(np.array(seg_dict[k]), axis=-1)
        return img, seg_dict
    return img


class DDSMAugmenter(object):
    """
    Use the augmentation Artie applied to DDSM dataset
    """

    def __init__(self,
                 obj_list,
                 augment_prob=0.5,
                 max_crop_noise=(100, 100),
                 max_crop_size_noise=100,
                 debug=False,
                 otsu_duke=False,
                 enable_vflip=False,
                 hardcore=False):
        self.debug = debug
        self.augment_prob = augment_prob
        self.obj_label_dict = {k: i for i, k in enumerate(obj_list)}
        self.simple_augmenter = SimpleAugmenter(
            max_crop_noise=max_crop_noise,
            max_crop_size_noise=max_crop_size_noise,
        )
        self.otsu_duke = otsu_duke
        self.enable_vflip = enable_vflip
        self.hardcore = hardcore

    def hflip(self, img, annot, seg_dict):
        # call this function for each image and segmentation
        if random.random() < self.augment_prob:
            return hflip_image_annot_seg(img, annot, seg_dict)
        return img, annot, seg_dict

    def vflip(self, img, annot, seg_dict):
        if random.random() < self.augment_prob:
            return vflip_image_annot_seg(img, annot, seg_dict)
        return img, annot, seg_dict

    @staticmethod
    def get_image_height_width(img):
        return img.height, img.width

    def get_random_affine_params(self, img, hardcore=False):
        H, W = self.get_image_height_width(img)
        if hardcore:
            rotation = np.random.randint(low=-25, high=26)
        else:
            rotation = np.random.randint(low=-15, high=16)

        if hardcore:
            trans_x = int(round(np.random.randint(low=-20, high=21) / 100 * W))  # fixed to (-10, 10)
            trans_y = int(round(np.random.randint(low=-20, high=21) / 100 * H))  # fixed to (-10, 10)
        else:
            trans_x = int(round(np.random.randint(low=-10, high=11) / 100 * W))  # fixed to (-10, 10)
            trans_y = int(round(np.random.randint(low=-10, high=11) / 100 * H))  # fixed to (-10, 10)
        translate = (trans_x, trans_y)

        if hardcore:
            scale = np.random.randint(low=4, high=26) / 10  # Keep it at (8, 15)
            shear = np.random.randint(low=-40, high=46)
        else:
            scale = np.random.randint(low=8, high=16) / 10  # Keep it at (8, 15)
            shear = np.random.randint(low=-25, high=26)
        return rotation, translate, scale, shear

    def random_affine(self, img, seg_dict):
        # call this function for each image and segmentation
        if type(img) == list:
            rotation, translate, scale, shear = self.get_random_affine_params(img[0], self.hardcore)
            for i in range(len(img)):
                img[i] = TF.affine(img[i], angle=rotation, translate=translate, scale=scale, shear=shear)
        else:
            rotation, translate, scale, shear = self.get_random_affine_params(img, self.hardcore)
            img = TF.affine(img, angle=rotation, translate=translate, scale=scale, shear=shear)
        for k in seg_dict.keys():
            seg_dict[k] = TF.affine(seg_dict[k], angle=rotation, translate=translate, scale=scale, shear=shear)

        return img, seg_dict

    @staticmethod
    def get_connected_components(seg, lesion_class, lesion_ad):
        """
        return x1, y1, x2, y2 of connected components
        if width and height > 3
        """
        #mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(seg[:, :, 0])
        mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(seg)
        lesions = []
        for k, v in mask_pixels_dict.items():
            current_lesion = mask == k

            y_edge_top, y_edge_bottom = get_edge_values(current_lesion, "y")
            x_edge_left, x_edge_right = get_edge_values(current_lesion, "x")

            width = x_edge_right - x_edge_left
            height = y_edge_bottom - y_edge_top

            if (width > 3) and (height > 3):
                this_lesion_info = {
                    'X': x_edge_left,
                    'Y': y_edge_top,
                    'Width': width,
                    'Height': height,
                    'Class': lesion_class,
                    'AD': lesion_ad,
                    'mask': current_lesion
                }
                lesions.append(this_lesion_info)

        return lesions

    def extract_lesions_from_seg(self, label_key, seg):
        lesion_class, ad = revert_label_key(label_key)
        return self.get_connected_components(seg, lesion_class, ad)

    def extract_annot_from_seg(self, seg_dict):
        lesions = []
        for k in seg_dict.keys():
            lesions.extend(self.extract_lesions_from_seg(k, seg_dict[k]))
        return get_annotations_from_lesions(lesions, self.obj_label_dict), seg_dict

    def otsu_segmentations(self, img, seg_dict, sample):
        # Don't use this if not Duke
        if sample['which_dataset'] != 0:
            return seg_dict

        for seg_class, seg_matrix in seg_dict.items():
            if seg_matrix.ndim == 3:
                for layer in range(seg_matrix.shape[-1]):
                    seg_matrix_layer = seg_matrix[..., layer]
                    try:
                        # Crop out lesion
                        x1, x2 = seg_matrix_layer.max(axis=0).nonzero()[0][0], seg_matrix_layer.max(axis=0).nonzero()[0][-1]
                        y1, y2 = seg_matrix_layer.max(axis=1).nonzero()[0][0], seg_matrix_layer.max(axis=1).nonzero()[0][-1]
                        lesion_only = np.squeeze(img)[y1:y2, x1:x2]

                        # Segmentation over cropped area
                        thresh = threshold_otsu(lesion_only)
                        good_reg = lesion_only > thresh
                        good_reg = binary_closing(good_reg, structure=np.ones((3, 3)), iterations=10)

                        # Get largest component -- there should be only one
                        labels = label(good_reg)
                        assert (labels.max() != 0)
                        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

                        # New segmentation mask
                        new_seg = np.zeros((seg_matrix_layer.shape))
                        new_seg[y1:y2, x1:x2] = largestCC

                        seg_matrix[..., layer] = new_seg
                    except Exception as why:
                        print("ERROR DURING OTSU SEG: ", why)
            elif seg_matrix.ndim == 2:
                try:
                    # Crop out lesion
                    x1, x2 = seg_matrix.max(axis=0).nonzero()[0][0], seg_matrix.max(axis=0).nonzero()[0][-1]
                    y1, y2 = seg_matrix.max(axis=1).nonzero()[0][0], seg_matrix.max(axis=1).nonzero()[0][-1]
                    lesion_only = np.squeeze(img)[y1:y2, x1:x2]

                    # Segmentation over cropped area
                    thresh = threshold_otsu(lesion_only)
                    good_reg = lesion_only > thresh
                    good_reg = binary_closing(good_reg, structure=np.ones((3, 3)), iterations=10)

                    # Get largest component -- there should be only one
                    labels = label(good_reg)
                    assert (labels.max() != 0)
                    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1

                    # New segmentation mask
                    new_seg = np.zeros((seg_matrix.shape))
                    new_seg[y1:y2, x1:x2] = largestCC

                    seg_matrix = new_seg
                except Exception as why:
                    print("ERROR DURING OTSU SEG: ", why)
            else:
                raise ValueError(f"Num dimensions of segmentation not supported: {seg_matrix.ndim}")
            # Update the segmentation
            seg_dict[seg_class] = seg_matrix

        return seg_dict

    def apply_affine(self, sample):
        img, annot, seg_dict = sample['img'], sample['annot'], sample['seg_dict']

        # Do probablistic hflip (with original numpy array instead of PIL image)
        img, annot, seg_dict = self.hflip(img, annot, seg_dict)

        # Do probabilistic vflip
        if self.enable_vflip:
            img, annot, seg_dict = self.vflip(img, annot, seg_dict)

        if seg_dict is None:
            # if no segmentation, apply SimpleAugmenter
            # in this case, segmentation is not adjusted
            # meaning, using self.debug with Duke data without segmentation is problematic
            sample['img'], sample['annot'] = img, annot
            sample = self.simple_augmenter(sample)
            img, annot = sample['img'], sample['annot']
        else:
            # print('segmentation found', flush=True)
            annot = np.zeros((0, 5))  # ignore previous annotations
            # draw bounding box based on segmentation

            # Threshold pseudo-segmentations on Duke dataset
            if self.otsu_duke:
                seg_dict = self.otsu_segmentations(img, seg_dict, sample)

            # 1. turn segmentations and images into PIL image
            # 2. get the parameters for affine transformation,
            #    augment both image and segmentation using the same parameter
            # 3. turn PIL image back to numpy image
            if img.shape[2] > 1:
                _, seg_dict = numpy_to_pil(None, seg_dict)
                img_list = []
                for i in range(img.shape[2]):
                    img_x = numpy_to_pil(img[..., i], None)
                    img_list.append(img_x)
                img_list, seg_dict = self.random_affine(img_list, seg_dict)
                img, seg_dict = pil_to_numpy(img_list, seg_dict)
            else:
                img, seg_dict = numpy_to_pil(img, seg_dict)
                img, seg_dict = self.random_affine(img, seg_dict)
                img, seg_dict = pil_to_numpy(img, seg_dict)
            # 4. get bounding boxes, filter out too small ones (filter < 4 in either dim)
            annot, new_seg = self.extract_annot_from_seg(seg_dict)

        # return augmented image, annotations and segmentations
        sample['img'], sample['annot'], sample['seg_dict'] = img, annot, new_seg

        return sample

    def __call__(self, sample):
        sample = self.apply_affine(sample)

        try:
            if 'mixup_sample' in sample:
                mixup_sample = sample['mixup_sample']
                mixup_sample = self.apply_affine(mixup_sample)
                sample['mixup_sample'] = mixup_sample
        except Exception as why:
            print("Exception during affine on mixup: ", why)

        return sample


class DDSMAugmenterV2(DDSMAugmenter):
    """
    V2 handles the labels (0~9 labels in V2)
    """

    def __init__(self, augment_prob=0.5, max_crop_noise=(100, 100), max_crop_size_noise=100, debug=False, **kwargs):
        super().__init__(
            obj_list=[],
            augment_prob=augment_prob,
            max_crop_noise=max_crop_noise,
            max_crop_size_noise=max_crop_size_noise,
            debug=debug,
            **kwargs
        )

    @staticmethod
    def get_connected_components(seg, label_key):
        """
        return x1, y1, x2, y2 of connected components
        if width and height > 3
        """
        mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(seg[:, :, 0])
        lesions = []
        accepted_masks = []
        for k, v in mask_pixels_dict.items():
            current_lesion = mask == k

            y_edge_top, y_edge_bottom = get_edge_values(current_lesion, "y")
            x_edge_left, x_edge_right = get_edge_values(current_lesion, "x")

            width = x_edge_right - x_edge_left
            height = y_edge_bottom - y_edge_top

            if (width > 3) and (height > 3):
                this_lesion_info = {
                    'X': x_edge_left,
                    'Y': y_edge_top,
                    'Width': width,
                    'Height': height,
                    'combined_label': label_key,
                }
                lesions.append(this_lesion_info)
                accepted_masks.append(np.where(mask == k, 1, 0))

        return lesions, accepted_masks

    def extract_lesions_from_seg(self, label_key, seg):
        lesions, accepted_masks = self.get_connected_components(seg, label_key)
        return lesions, accepted_masks

    def extract_annot_from_seg(self, seg_dict):
        lesions = []
        seg_dict_new = {}
        for k in seg_dict.keys():
            extracted_lesions, accepted_masks = self.extract_lesions_from_seg(k, seg_dict[k])
            lesions.extend(extracted_lesions)
            if len(accepted_masks) > 0:
                seg_dict_new[k] = np.stack(accepted_masks)
                seg_dict_new[k] = np.transpose(seg_dict_new[k], (1, 2, 0))  # channel dim at the end
        return get_annotations_from_lesions_V2(lesions), seg_dict_new


class DDSMAugmenterV4(DDSMAugmenterV2):
    """
    V4 handles volume-consistent augmentation
    """

    def __init__(self, augment_prob=0.5, max_crop_noise=(100, 100), max_crop_size_noise=100, debug=False, **kwargs):
        super().__init__(
            augment_prob=augment_prob,
            max_crop_noise=max_crop_noise,
            max_crop_size_noise=max_crop_size_noise,
            debug=debug,
            **kwargs
        )
        # replace simple_augmenter with V4 version
        self.simple_augmenter = SimpleAugmenterV4(
            max_crop_noise=max_crop_noise,
            max_crop_size_noise=max_crop_size_noise,
        )

    def hflip(self, img, annot, seg_dict):
        # call this function for each image and segmentation
        if random.random() < self.augment_prob:
            return hflip_image_annot_seg_v4(img, annot, seg_dict)
        return img, annot, seg_dict

    def vflip(self, img, annot, seg_dict):
        if random.random() < self.augment_prob:
            return vflip_image_annot_seg_v4(img, annot, seg_dict)
        return img, annot, seg_dict

    @staticmethod
    def get_image_height_width(img):
        return img.shape

    def extract_annot_from_seg(self, seg_dict):
        """
        get segmentations for all slices for all label_keys, even if that segmentation is empty
        this is so that I can stack the result later
        """
        lesions = []
        seg_dict_new = {}
        for k in seg_dict.keys():
            extracted_lesions, accepted_masks = self.extract_lesions_from_seg(k, seg_dict[k])
            lesions.extend(extracted_lesions)
            if len(accepted_masks) > 0:
                seg_dict_new[k] = np.stack(accepted_masks)
                seg_dict_new[k] = np.transpose(seg_dict_new[k], (1, 2, 0))  # channel dim at the end
            else:
                seg_dict_new[k] = np.zeros(seg_dict[k].shape, dtype=seg_dict[k].dtype)
        return get_annotations_from_lesions_V2(lesions), seg_dict_new

    def apply_affine(self, sample):
        img, annot, seg_dict = sample['img'], sample['annot'], sample['seg_dict']

        # Do probablistic hflip (with original numpy array instead of PIL image)
        img, annot, seg_dict = self.hflip(img, annot, seg_dict)

        # Do probabilistic vflip
        if self.enable_vflip:
            img, annot, seg_dict = self.vflip(img, annot, seg_dict)

        if seg_dict is None:
            # if no segmentation, apply SimpleAugmenter
            # in this case, segmentation is not adjusted
            # meaning, using self.debug with Duke data without segmentation is problematic
            sample['img'], sample['annot'] = img, annot
            sample = self.simple_augmenter(sample)
            img, annot = sample['img'], sample['annot']
        else:
            # Threshold pseudo-segmentations on Duke dataset
            assert not self.otsu_duke, "otsu_duke for V4 not implemented"

            # get transformation parameters
            rotation, translate, scale, shear = self.get_random_affine_params(img[:, :, 0], self.hardcore)

            # transform each slice separately
            # assume the image and segmentations have the same shape
            augmented_img_list, augmented_annot_list, augmented_seg_dict_list = [], [], []
            for slice_idx in range(img.shape[2]):

                current_slice_img = img[:, :, slice_idx:slice_idx + 1]
                current_slice_seg_dict = {k: v[:, :, slice_idx:slice_idx + 1] for k, v in seg_dict.items()}

                current_slice_img, current_slice_seg_dict = numpy_to_pil(current_slice_img, current_slice_seg_dict)
                current_slice_img = TF.affine(current_slice_img, angle=rotation, translate=translate, scale=scale, shear=shear)

                for k in current_slice_seg_dict.keys():
                    current_slice_seg_dict[k] = TF.affine(current_slice_seg_dict[k], angle=rotation, translate=translate, scale=scale, shear=shear)
                current_slice_img, current_slice_seg_dict = pil_to_numpy(current_slice_img, current_slice_seg_dict)
                current_slice_annot, current_slice_new_seg = self.extract_annot_from_seg(current_slice_seg_dict)
                augmented_img_list.append(current_slice_img)
                augmented_annot_list.append(current_slice_annot)
                augmented_seg_dict_list.append(current_slice_new_seg)

            img = np.concatenate(augmented_img_list, axis=-1)
            annot = augmented_annot_list
            new_seg = {k: np.concatenate([x[k] for x in augmented_seg_dict_list], axis=-1) for k in seg_dict.keys()}

        # return augmented image, annotations and segmentations
        sample['img'], sample['annot'], sample['seg_dict'] = img, annot, new_seg

        return sample


class SimpleAugmenter(object):
    """
    simple random crop and resize augmenter
    """

    def __init__(self, max_crop_noise=(100, 100), max_crop_size_noise=100):
        self.max_crop_noise = max_crop_noise
        self.max_crop_size_noise = max_crop_size_noise

    def random_crop(self, image, annots):
        h, w, _ = image.shape
        # get crop size
        new_h = h - self.max_crop_noise[0] - self.max_crop_size_noise
        new_w = w - self.max_crop_noise[1] - self.max_crop_size_noise
        new_h += random.randint(0, self.max_crop_size_noise)
        new_w += random.randint(0, self.max_crop_size_noise)
        # translate image
        translate_h = random.randint(0, self.max_crop_noise[0])
        translate_w = random.randint(0, self.max_crop_noise[1])

        annots = self.adjust_annots(translate_w, translate_h, new_w, new_h, annots)

        return image[translate_h:translate_h + new_h, translate_w:translate_w + new_w, :], annots

    def adjust_annots(self, translate_w, translate_h, new_w, new_h, annots):
        if annots is not None:
            annots[:, 0] -= translate_w
            annots[:, 2] -= translate_w
            annots[:, 1] -= translate_h
            annots[:, 3] -= translate_h

            # if the lesion is out-of-bounds, correct it
            annots[annots < 0] = 0
            annots[:, 0][annots[:, 0] > new_w] = new_w
            annots[:, 2][annots[:, 2] > new_w] = new_w
            annots[:, 1][annots[:, 1] > new_h] = new_h
            annots[:, 3][annots[:, 3] > new_h] = new_h

            # retain size > 1 lesions
            indices_to_remove = []
            for i in range(len(annots)):
                if (annots[i, 2] - annots[i, 0] <= 1) or (annots[i, 3] - annots[i, 1] <= 1):
                    indices_to_remove.append(i)
            annots = np.delete(annots, indices_to_remove, 0)
        return annots

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        img, annot = self.random_crop(img, annot)

        sample['img'], sample['annot'] = img, annot
        return sample


class SimpleAugmenterV4(SimpleAugmenter):
    """
    simple random crop and resize augmenter
    """

    def __init__(self, max_crop_noise=(100, 100), max_crop_size_noise=100):
        super().__init__(
            max_crop_noise=max_crop_noise,
            max_crop_size_noise=max_crop_size_noise,
        )

    def adjust_annots(self, translate_w, translate_h, new_w, new_h, annots_list):
        new_annots_list = []
        if annots_list is not None:
            for original_annots in annots_list:
                annots = original_annots.copy()
                annots[:, 0] -= translate_w
                annots[:, 2] -= translate_w
                annots[:, 1] -= translate_h
                annots[:, 3] -= translate_h

                # if the lesion is out-of-bounds, correct it
                annots[annots < 0] = 0
                annots[:, 0][annots[:, 0] > new_w] = new_w
                annots[:, 2][annots[:, 2] > new_w] = new_w
                annots[:, 1][annots[:, 1] > new_h] = new_h
                annots[:, 3][annots[:, 3] > new_h] = new_h

                # retain size > 1 lesions
                indices_to_remove = []
                for i in range(len(annots)):
                    if (annots[i, 2] - annots[i, 0] <= 1) or (annots[i, 3] - annots[i, 1] <= 1):
                        indices_to_remove.append(i)
                annots = np.delete(annots, indices_to_remove, 0)
                new_annots_list.append(annots)
        return new_annots_list


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        result = {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
        if 'scale' in sample:
            result['scale'] = sample['scale']

        return result


class Standardizer(object):

    def __init__(self, disable=False):
        self.disable = disable

    def __call__(self, sample):
        if self.disable:
            return sample

        image = sample['img']
        sample['img'] = (image - image.mean()) / np.maximum(np.std(image), 10 ** (-5))

        return sample


class Pass(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        return sample


class RepeatChannels(object):

    def __init__(self, channels):
        self.channels = channels

    def __call__(self, obj):
        """
        Args:
            tensor: Tensor to be repeated.
        Returns:
            Tensor: repeated tensor.
        """
        if obj['img'].shape[-1] != self.channels:
            obj['img'] = obj['img'].repeat(1, 1, self.channels)
        return obj

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NumPyToTorch(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        """
        Args:
            tensor: Tensor to be repeated.
        Returns:
            Tensor: repeated tensor.
        """
        sample['img'] = torch.from_numpy(sample['img']).to(torch.float32)
        # handle test phase
        if isinstance(sample.get('annot', None), (np.ndarray, np.generic)):
            sample['annot'] = torch.from_numpy(sample['annot'])
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ElasticTransformer(object):
    def __init__(self,
                 alpha: tuple = (500, 3000),
                 sigma: tuple = (20, 60),
                 mode: str = 'constant',
                 p: float = 1.0,
                 verbose_debug: bool = False):
        """
        Perform elastic deformations on images and segmentations
        :param alpha: Strength of a Gaussian filter. Tuple contains
                      a range of (minimum,maximum) alpha value
        :param sigma: Number of 'nodes' in the Gaussian filter.
                      Tuple contains a range of (minimum,maximum)
                      sigma value.
        :param mode: Behavior for points outside of boundaries
                     (refer to scipy docs for details).
        :param p: Probability of applying augmentation
        :param verbose_debug: Print out debugging info
        """
        self.alpha = alpha
        self.sigma = sigma
        self.mode = mode
        self.p = p
        self.verbose_debug = verbose_debug

        if len(alpha) != 2:
            raise ValueError("Alpha must be a tuple of (min,max) values")
        if len(sigma) != 2:
            raise ValueError("Sigma must be a tuple of (min,max) values")

    def transform(self, image, order, alpha, sigma, indices=None):
        """
        :param order: Set 0 for nearest; 1 for linear
                    use nearest for segmentations; linear for imgs
        """

        shape = image.shape

        if indices is None:
            random_state = np.random.RandomState(None)

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

        if len(image.shape) == 2:
            image = map_coordinates(image, indices, order=order, mode=self.mode, cval=image.min()).reshape(shape)
        elif len(image.shape) == 3:
            for i in range(image.shape[2]):
                image[..., i] = map_coordinates(image[..., i], indices, order=order, mode=self.mode, cval=image[..., i].min()).reshape(image[..., i].shape)

        return image, indices

    def __call__(self, sample: dict):
        # Probability of applying transformation
        if random.random() > self.p:
            if self.verbose_debug:
                print("Skipping elastic aug")
            return sample

        image = sample['img'][..., 0]
        segs = sample['seg_dict']

        # Get random alpha and sigma from the range
        # alpha = random.randint(image.shape[0]*1, image.shape[0]*2)
        alpha = random.randint(self.alpha[0], self.alpha[1])
        sigma = random.randint(self.sigma[0], self.sigma[1])

        # safety heuristic not to make too strong destructive augs
        if alpha / sigma > 80:
            sigma = alpha / 80

        if self.verbose_debug:
            print(f"Elastic aug with alpha: {alpha}, sigma: {sigma} / {alpha / sigma}")

        # Transform images
        # print(f"Transform the image with shape {image.shape} (orig {sample['img'].shape}), # seg: {len(segs)}")
        sample['img'][..., 0], indices = self.transform(image, 3, alpha, sigma)

        # Transform segmentations
        if segs is not None and len(segs) != 0:
            for seg_class, seg in segs.items():
                seg_t, _ = self.transform(np.squeeze(seg), 0, alpha, sigma, indices)
                if seg_t.ndim == 2:
                    sample['seg_dict'][seg_class][..., 0] = seg_t
                elif seg_t.ndim == 3:
                    sample['seg_dict'][seg_class] = seg_t

        return sample


class UpdateBoxes(object):
    """Update bounding boxes (annotations) after changes have been made to segmentations"""

    def __init__(self):
        pass

    @staticmethod
    def get_connected_components(seg, label_key):
        """
        return x1, y1, x2, y2 of connected components
        if width and height > 3
        """
        mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(seg[:, :, 0])
        lesions = []
        accepted_masks = []
        for k, v in mask_pixels_dict.items():
            current_lesion = mask == k

            y_edge_top, y_edge_bottom = get_edge_values(current_lesion, "y")
            x_edge_left, x_edge_right = get_edge_values(current_lesion, "x")

            width = x_edge_right - x_edge_left
            height = y_edge_bottom - y_edge_top

            if (width > 3) and (height > 3):
                this_lesion_info = {
                    'X': x_edge_left,
                    'Y': y_edge_top,
                    'Width': width,
                    'Height': height,
                    'combined_label': label_key,
                }
                lesions.append(this_lesion_info)
                accepted_masks.append(np.where(mask == k, 1, 0))

        return lesions, accepted_masks

    def extract_lesions_from_seg(self, label_key, seg):
        lesions, accepted_masks = self.get_connected_components(seg, label_key)
        return lesions, accepted_masks

    def extract_annot_from_seg(self, seg_dict):
        lesions = []
        seg_dict_new = {}
        for k in seg_dict.keys():
            extracted_lesions, accepted_masks = self.extract_lesions_from_seg(k, seg_dict[k])
            lesions.extend(extracted_lesions)
            if len(accepted_masks) > 0:
                seg_dict_new[k] = np.stack(accepted_masks)
                seg_dict_new[k] = np.transpose(seg_dict_new[k], (1, 2, 0))  # channel dim at the end
        return get_annotations_from_lesions_V2(lesions), seg_dict_new

    def __call__(self, sample):
        try:
            sample['annot'], sample['seg_dict'] = self.extract_annot_from_seg(sample['seg_dict'])
        except Exception as why:
            print("Exception during UpdateBoxes:", why)

        return sample


class Cutout(object):
    def __init__(self, n_holes=5, length_ratio=0.1, channel_dim=2, remove_seg_threshold=0.15, p=1.0):
        """
        :param n_holes: How many holes will be generated
        :param length: What % of image size is the length of the box (default: 1% => 0.01)
        :param channel_dim: Which dimension in channel dimension, default (h,w,c) => 2
        :param remove_seg_threshold: If less than this percent of original segmentation, remove it completely
        :param p: Probability of applying augmentation
        """
        self.n_holes = n_holes
        self.length_ratio = length_ratio
        self.channel_dim = channel_dim
        self.remove_seg_threshold = remove_seg_threshold
        self.p = p

    def apply_cutout(self, sample):
        img = sample['img']
        h = img.shape[0]
        w = img.shape[1]

        cutout_mask = np.ones((h, w), np.float32)

        # Create box cutout length
        length = round(h * self.length_ratio)

        # Generate holes
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            cutout_mask[y1:y2, x1:x2] = 0.

        cutout_mask = np.expand_dims(cutout_mask, self.channel_dim)  # (h,w) to (h,w,1)

        # Apply holes to the image
        img_ = img * cutout_mask

        # Apply holes to segmentation
        for label_cls, label_matrix in sample['seg_dict'].copy().items():
            positives_before = np.count_nonzero(label_matrix)

            # Get segmentation boundaries before cutout
            x1, x2 = label_matrix.max(axis=0).nonzero()[0][0], label_matrix.max(axis=0).nonzero()[0][-1]
            y1, y2 = label_matrix.max(axis=1).nonzero()[0][0], label_matrix.max(axis=1).nonzero()[0][-1]

            # Apply cutout mask
            label_matrix_ = label_matrix * cutout_mask

            # Check boundaries after cutout
            try:
                x1_, x2_ = label_matrix_.max(axis=0).nonzero()[0][0], label_matrix_.max(axis=0).nonzero()[0][-1]
                y1_, y2_ = label_matrix_.max(axis=1).nonzero()[0][0], label_matrix_.max(axis=1).nonzero()[0][-1]
            except IndexError:
                # Segmentation were cut out completely -- no positive values
                del sample['seg_dict'][label_cls]
                continue

            positives_after = np.count_nonzero(label_matrix_)

            # Cut out does cover part of the area, but not enough to change bounding box
            # In this case, we restore that part of segmentation AND the image
            if (
                    x1 == x1_ and
                    x2 == x2_ and
                    y1 == y1_ and
                    y2 == y2_ and
                    positives_before != positives_after
            ):
                label_matrix_ = label_matrix
                img_[y1:y2, x1:x2] = img[y1:y2, x1:x2]

            # If minimum of positive values not met, delete area completely
            if (positives_after / positives_before) < self.remove_seg_threshold:
                del sample['seg_dict'][label_cls]
            else:
                sample['seg_dict'][label_cls] = label_matrix_

        sample['img'] = img_
        return sample

    def __call__(self, sample):
        # Probability of applying transformation
        if random.random() > self.p:
            return sample

        try:
            sample = self.apply_cutout(sample)
        except Exception as why:
            print("Exception during Cutout: ", why)

        return sample


class RandomGamma(object):
    def __init__(self, p=1.0, gamma=(-0.3, 0.3), ignore_datasets=[2], verbose=False):
        """
        :param p: Probability of applying augmentation
        :param gamma: Power
        :param ignore_datasets: Do not apply transformation to images from this dataset (e.g. FFDM: [2])
        :param verbose: Print what gamma was chosen
        """
        self.p = p
        self.gamma = gamma
        self.ignore_datasets = ignore_datasets
        self.verbose = verbose

        if isinstance(self.gamma, tuple):
            assert len(self.gamma) == 2, "Gamma can only be a single int/float or a tuple of two float values"

    def apply_gamma(self, sample):
        img = sample['img']
        if 'which_dataset' in sample:
            if sample['which_dataset'] in self.ignore_datasets:
                if self.verbose:
                    print("Study in ignored datasets")
                return sample

        # Do not perform on negative numbers
        if img.min() < 0:
            print("Warning: can't apply random gamma when minimum value of matrix is lower than 0")
            return sample

        # Get gamma value
        gamma = self.gamma
        if isinstance(gamma, tuple):
            gamma = random.uniform(gamma[0], gamma[1])
            if self.verbose:
                print("Gamma: ", gamma)
        gamma = np.exp(gamma)

        # Apply transformation
        img = img ** gamma

        sample['img'] = img
        return sample

    def __call__(self, sample):
        # Probability of applying transformation
        if random.random() > self.p:
            return sample

        try:
            sample = self.apply_gamma(sample)
        except Exception as why:
            print("Exception during RandomGamma on main", why)

        if 'mixup_sample' in sample:
            try:
                mixup_sample = sample['mixup_sample']
                mixup_sample = self.apply_gamma(mixup_sample)
                sample['mixup_sample'] = mixup_sample
            except Exception as why:
                print("Exception during RandomGamma on mixup:", why)

        return sample


class RandomGaussianNoise(object):
    def __init__(self, p=1.0, mean=0., std=(0, 0.20), verbose=False):
        self.p = p
        self.mean = mean
        self.std = std
        self.verbose = verbose

    def apply_gaussian_noise(self, sample):
        img = sample['img']

        if isinstance(self.std, tuple):
            std = random.uniform(self.std[0], self.std[1])
            if self.verbose:
                print("Std:", std)

        img = img + np.random.randn(*img.shape) * std + self.mean

        sample['img'] = img

        return sample

    def __call__(self, sample):
        # Probability of applying transformation
        if random.random() > self.p:
            return sample

        try:
            sample = self.apply_gaussian_noise(sample)
        except Exception as why:
            print("Exception during RandomGaussianNoise:", why)

        return sample


class OneOf(object):
    def __init__(self, aug_dict: dict, p=1.0):
        self.aug_dict = aug_dict
        self.p = p

        assert sum(self.aug_dict.values()) == 1, "Probabilities must sum to one"

    def __call__(self, sample):
        # Probability of applying any of the transformations
        if random.random() > self.p:
            return sample

        selected_aug = np.random.choice(list(self.aug_dict.keys()), size=None, p=list(self.aug_dict.values()))
        sample = selected_aug(sample)

        return sample


class nnUNetIntensity(object):
    def __init__(self, p=1.0, percentiles=(0.5, 99.5), ignore_datasets=[2], verbose=False):
        self.p = p
        self.percentiles = percentiles
        self.ignore_datasets = ignore_datasets
        self.verbose = verbose

    def apply_intensity(self, sample):
        img = sample['img']

        # Apply transformation
        low, high = np.percentile(img, (self.percentiles[0], self.percentiles[1]))
        img = rescale_intensity(img, in_range=(low, high), out_range="image")

        sample['img'] = img

        return sample

    def __call__(self, sample):
        # Probability of applying transformation
        if random.random() > self.p:
            return sample

        if 'which_dataset' in sample:
            if sample['which_dataset'] in self.ignore_datasets:
                if self.verbose:
                    print("Study in ignored datasets")
                return sample

        try:
            sample = self.apply_intensity(sample)
        except Exception as why:
            print("Exception during nnUNetIntensity on main:", why)

        if 'mixup_sample' in sample:
            try:
                mixup_sample = sample['mixup_sample']
                mixup_sample = self.apply_intensity(mixup_sample)
                sample['mixup_sample'] = mixup_sample
            except Exception as why:
                print("Exception during nNUNetIntensity on mixup: ", why)

        return sample


class Mixup(object):
    def __init__(self, alpha=0.5, p=1.0):
        self.alpha = alpha
        self.p = p

    def __call__(self, sample: dict):
        # Probability of applying transformation
        if random.random() > self.p:
            return sample

        if 'mixup_sample' not in sample:
            # print("There is no mixup sample passed in the dataset")
            return sample
        first_img = sample['img']
        second_img = sample['mixup_sample']['img']

        # Mixup image
        mixup_img = self.alpha * first_img + (1 - self.alpha) * second_img

        # Concat bounding boxes
        mixup_annot = np.concatenate((sample['annot'], sample['mixup_sample']['annot']))

        # Assign mixed images back to the original sample dictionary
        sample['img'] = mixup_img
        sample['annot'] = mixup_annot

        return sample


class CopyPasteAugmentation(object):
    def __init__(self, jitter=0.5, p=1.0, n=1, localization_version=2, paste_version=2,
                 iou_threshold=0.1, scale=1.0, flip_segmentation=True, rotate_segmentation=True, augment_duke=False, paste_segmentation_weight=1.0,
                 train_file=''):
        # print("Using copy-paste augmenter.")
        self.jitter = jitter  # TODO: implement
        self.localization_version = localization_version
        self.paste_version = paste_version
        self.paste_segmentation_weight = paste_segmentation_weight
        self.iou_threshold = iou_threshold
        self.scale = scale
        basename = os.path.basename(train_file)
        dbt_path = f'/gpfs/data/geraslab/chledj01/segs/{basename[:-4]}_seg_dbt.pkl'
        ffdm_path = f'/gpfs/data/geraslab/chledj01/segs/{basename[:-4]}_seg_ffdm.pkl'
        self.dbt_seg_list = pickling.unpickle_from_file(dbt_path)
        self.dbt_seg_len = len(self.dbt_seg_list)
        self.ffdm_seg_list = pickling.unpickle_from_file(ffdm_path)
        self.ffdm_seg_len = len(self.ffdm_seg_list)
        self.p = p
        self.n = n
        self.augment_duke = augment_duke
        self.flip_segmentation = flip_segmentation
        self.rotate_segmentation = rotate_segmentation
        self.lesion_seg_suffix_to_class_dict = {
            'malignant_rest': 4,
            'benign_rest': 5,
            'malignant_distortion': 6,
            'benign_distortion': 7,
            'malignant': 8,
            'benign': 9,
        }
        # TODO - reshaping
        # TODO increase the number of negative imgs?

    def __call__(self, sample: dict):

        if not self.augment_duke:
            if sample['which_dataset'] == 0:
                return sample

        # from PIL import Image
        # random_num = random.randint(1, 100000)
        # im_to_save = Image.fromarray((sample["img"].repeat(3, -1) * 255).astype(np.uint8))
        # im_to_save.save(f"copypaste_aug_visualization/before_{random_num}.jpeg")

        h, w, d = sample["img"].shape
        h, w, d = int(h), int(w), int(d)

        n_segs = np.random.binomial(self.n, self.p)

        for i in range(n_segs):
            copied_segmentation, copied_seg_box, copied_segmentation_class = self.load_segmentation_crop(sample["which_dataset"])
            assert copied_segmentation.shape == copied_seg_box.shape

            seg_h, seg_w = copied_segmentation.shape
            seg_h, seg_w = int(seg_h), int(seg_w)
            if self.scale * seg_h < h and self.scale * seg_w < w:  # sizes of images are so much different, ie (3523,2081) with seg (1199,1494) vs (2116,1339)
                seg_x1 = random.randint(0, w - seg_w)
                seg_y1 = random.randint(0, h - seg_h)
                seg_x2 = seg_x1 + seg_w
                seg_y2 = seg_y1 + seg_h
                seg_annot = [[seg_x1, seg_y1, seg_x2, seg_y2, copied_segmentation_class]]

                if self.localization_version == 0:  # check whether overlap is not too big, then paste
                    trials = 0
                    while trials < 10 and self.calc_max_iou(sample['annot'], seg_annot) > self.iou_threshold:
                        trials += 1
                        seg_x1 = random.randint(0, w - seg_w)
                        seg_y1 = random.randint(0, h - seg_h)
                        seg_x2 = seg_x1 + seg_w
                        seg_y2 = seg_y1 + seg_h
                        seg_annot = [[seg_x1, seg_y1, seg_x2, seg_y2, copied_segmentation_class]]
                elif self.localization_version == 1:  # IOU + check whether the area is inside breast
                    trials = 0
                    likely_outside = (sample["img"][seg_y1:seg_y2, seg_x1:seg_x2] == 0).sum() > copied_segmentation.size / 20
                    while trials < 10 and (self.calc_max_iou(sample['annot'], seg_annot) > self.iou_threshold or likely_outside):
                        trials += 1
                        seg_x1 = random.randint(0, w - seg_w)
                        seg_y1 = random.randint(0, h - seg_h)
                        seg_x2 = seg_x1 + seg_w
                        seg_y2 = seg_y1 + seg_h
                        likely_outside = sample["img"][seg_y1:seg_y2, seg_x1:seg_x2].sum() < copied_segmentation.size / 20
                        seg_annot = [[seg_x1, seg_y1, seg_x2, seg_y2, copied_segmentation_class]]

                if self.paste_version == 0:  # paste only the segmentation
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box != 0] *= 1 - self.paste_segmentation_weight
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box != 0] += self.paste_segmentation_weight * copied_segmentation[:, :, np.newaxis][
                        copied_seg_box != 0]
                elif self.paste_version == 1:  # paste the whole box
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2] *= 1 - self.paste_segmentation_weight
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2] += self.paste_segmentation_weight * copied_segmentation[:, :, np.newaxis]
                elif self.paste_version == 2:  # strictly paste the segmentation, and average the roundabouts
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box != 0] = copied_segmentation[:, :, np.newaxis][copied_seg_box != 0]
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box == 0] += copied_segmentation[:, :, np.newaxis][copied_seg_box == 0]
                    sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box == 0] /= 2
                else:
                    raise ValueError(f"copy_version: {self.copy_version} not supported.")

                sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box != 0] *= 1 - self.paste_segmentation_weight
                sample["img"][seg_y1:seg_y2, seg_x1:seg_x2][copied_seg_box != 0] += self.paste_segmentation_weight * copied_segmentation[:, :, np.newaxis][
                    copied_seg_box != 0]
                sample["annot"] = np.append(sample["annot"], seg_annot, axis=0)

        # im_to_save = Image.fromarray((sample["img"].repeat(3, -1) * 255).astype(np.uint8))
        # im_to_save.save(f"copypaste_aug_visualization/after_{random_num}.jpeg")
        return sample

    def load_segmentation_crop(self, which_dataset):

        segmentation = None
        seg_box = None
        while segmentation is None or seg_box is None:
            if which_dataset in {0, 1}:
                seg_id = random.randint(0, self.dbt_seg_len - 1)
                seg_path = self.dbt_seg_list[seg_id]
            elif which_dataset == 2:
                seg_id = random.randint(0, self.ffdm_seg_len - 1)
                seg_path = self.ffdm_seg_list[seg_id]
            else:
                print(f"dataset {which_dataset} not supported in copy-paste augmentation.")

            seg_class = self.lesion_seg_suffix_to_class_dict[seg_path.split('.')[-3]]

            try:
                data = h5.File(seg_path, 'r')
                segmentation = np.array(data['img_box'])
                seg_box = np.array(data['seg_box'])
                data.close()
            except:
                print(f'Failed to load img_box or seg_box from {seg_path}. Trying to load another segmentation..')

        if self.flip_segmentation:
            flip_version = random.choice([1, 2, 3, 4])
            if flip_version == 1:
                segmentation = np.flip(segmentation)  # Vertical and horizontal flip
                seg_box = np.flip(seg_box)
            if flip_version == 2:
                segmentation = np.flip(segmentation, axis=0)  # Vertical flip
                seg_box = np.flip(seg_box, axis=0)
            if flip_version == 3:
                segmentation = np.flip(segmentation, axis=1)  # Horizontal flip
                seg_box = np.flip(seg_box, axis=1)
            if flip_version == 4:
                pass  # No flip

        if self.rotate_segmentation:
            rotate_version = random.choice([1, 2, 3, 4])
            if rotate_version == 1:
                segmentation = np.rot90(segmentation, k=1)  # Rotate 90 degrees
                seg_box = np.rot90(seg_box, k=1)
            if rotate_version == 2:
                segmentation = np.rot90(segmentation, k=2)  # Rotate 180 degrees
                seg_box = np.rot90(seg_box, k=2)
            if rotate_version == 3:
                segmentation = np.rot90(segmentation, k=3)  # Rotate 270 degrees
                seg_box = np.rot90(seg_box, k=3)
            if rotate_version == 4:
                pass  # No rotation

        return segmentation, seg_box, seg_class

    def copy_segmentation_from_image(self, cp_sample):
        for key in cp_sample['seg_dict'].keys():
            # print(1, key, cp_sample['annot'][0][4])  # todo: make sure its equal
            assert key == int(cp_sample['annot'][0][4])
            w, h, d = cp_sample['img'].shape
            x1, y1, x2, y2 = cp_sample['annot'][0][:4]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            segmentation_box = cp_sample['seg_dict'][key][y1:y2, x1:x2]
            image_box = cp_sample['img'][y1:y2, x1:x2]
            segmented_image = segmentation_box * image_box

            if self.copy_version == 0:
                return segmented_image, cp_sample['annot'][0][4]
            elif self.copy_version == 1:
                return image_box, cp_sample['annot'][0][4]
            elif self.copy_version == 2:
                return random.choice([image_box, segmented_image]), cp_sample['annot'][0][4]
            else:
                raise ValueError(f"copy_version: {self.copy_version} not supported.")
            # break  # for now, only 1 segm per image - TODO consider disabling.

    def calc_max_iou(self, annot, seg_annot, not_part_lesion=False):
        """
        Calculate largest IoU between seg_annot and any annotation.
        :param annot: Numpy array with annotations: [[x1, y1, x2, y2, cls], ...]
        :param seg_annot: Numpy array with copied segmentation annotation: [[x1, y1, x2, y2, cls]]
        :param not_part_lesion:
        :return:
        """
        max_iou = 0

        if annot.size == 0:
            return 0

        for a in annot:
            area = (seg_annot[0][2] - seg_annot[0][0]) * (seg_annot[0][3] - seg_annot[0][1])
            iw = min(a[2], seg_annot[0][2]) - max(a[0], seg_annot[0][0])
            ih = min(a[3], seg_annot[0][3]) - max(a[1], seg_annot[0][1])
            iw = max(iw, 0)
            ih = max(ih, 0)
            ua = (a[3] - a[1]) * (a[2] - a[0]) + area - iw * ih
            ua = max(ua, 1e-8)
            intersection = iw * ih
            IoU = intersection / ua
            max_iou = max(max_iou, IoU)
            # if not_part_lesion:
            #     IoU[torch.lt(torch.unsqueeze(a[0], dim=1), seg_annot[0][0])] = 0
            #     IoU[torch.gt(torch.unsqueeze(a[2], dim=1), seg_annot[0][2])] = 0
            #     IoU[torch.lt(torch.unsqueeze(a[1], dim=1), seg_annot[0][1])] = 0
            #     IoU[torch.gt(torch.unsqueeze(a[3], dim=1), seg_annot[0][3])] = 0
        return IoU

    # cp_sample[cp_idx] = {
    #     'img': cp_img,
    #     'annot': cp_annot_t,
    #     'seg_dict': cp_seg_dict,
    #     'which_dataset': self.data_list[cp_idx]['which_dataset'],
    #     'best_center': self.data_list[cp_idx]['best_center'],
    #     'view': self.data_list[cp_idx]['View']
    # }
