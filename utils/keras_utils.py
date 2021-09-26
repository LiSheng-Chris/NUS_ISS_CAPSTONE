
from __future__ import division, print_function
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
import scipy.ndimage as ndi


def preprocess_input_tf(x, data_format=None, **kwargs):
    return preprocess_input(x, data_format, mode='tf')

def preprocess_input_caffe(x, data_format=None, **kwargs):
    return preprocess_input(x, data_format, mode='caffe')

def center_crop(x, center_crop_size):
    centerh, centerw = x.shape[0] // 2, x.shape[1] // 2
    lh, lw = center_crop_size[0] // 2, center_crop_size[1] // 2
    rh, rw = center_crop_size[0] - lh, center_crop_size[1] - lw
    h_start, h_end = centerh - lh, centerh + rh
    w_start, w_end = centerw - lw, centerw + rw
    return x[h_start:h_end, w_start:w_end, :]

def random_crop(x, random_crop_size):
    h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    return x[h_start:h_end, w_start:w_end, :]

def transform_matrix_offset_center(matrix, x, y):
    """Return transform matrix offset center.

    Parameters
    ----------
    matrix : numpy array
        Transform matrix
    x, y : int
        Size of image.

    Examples
    --------
    - See ``rotation``, ``shear``, ``zoom``.
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix 
    
def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x
def random_transform(x,
                     dim_ordering='tf',
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     seed=None,
                     **kwargs):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
        # Returns
            A randomly transformed version of the input (same shape).
        """

        x = x.astype('float32')
        # x is a single image, so it doesn't have image number at index 0
        if dim_ordering == 'th':
            img_channel_axis = 0
            img_row_axis = 1
            img_col_axis = 2
        if dim_ordering == 'tf':
            img_channel_axis = 2
            img_row_axis = 0
            img_col_axis = 1

        if seed is not None:
            np.random.seed(seed)

        if np.isscalar(zoom_range):
            zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if rotation_range:
            theta = np.deg2rad(np.random.uniform(rotation_range, rotation_range))
        else:
            theta = 0

        if height_shift_range:
            tx = np.random.uniform(-height_shift_range, height_shift_range)
            if height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if width_shift_range:
            ty = np.random.uniform(-width_shift_range, width_shift_range)
            if width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if shear_range:
            shear = np.deg2rad(np.random.uniform(shear_range, shear_range))
        else:
            shear = 0

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            x = apply_transform(x, transform_matrix, img_channel_axis,
                                      fill_mode=fill_mode, cval=cval)

        if channel_shift_range != 0:
            x = image.random_channel_shift(x,
                                           channel_shift_range,
                                           img_channel_axis)
        if horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_axis)

        if vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_axis)

        return x


def balanced_generator(image_dir, filenames, classes, input_dim, crop_dim, batch_size,
                       data_augmentation=True, save_to_dir=None, add_classes=False,
                       preprocess_mode='tf'):

    # count number of samples and classes
    nb_class = len(np.unique(classes))
    nb_sample = len(filenames)
    print('Found %d ROIs belonging to %d classes.' % (nb_sample, nb_class))

    # build an index of the images in the different class subfolders
    filename_dict = {}
    for i, (subdir, i_class) in enumerate(zip(filenames, classes)):
        subpath = os.path.join(image_dir, subdir)
        pattern = ('class_%d.jpg' % i_class) if add_classes else '.jpg'
        found = False
        for fname in os.listdir(subpath):
            if fname.lower().endswith(pattern):
                found = True
                if i in filename_dict: 
                    filename_dict[i].append(os.path.join(subdir, fname))
                else:
                    filename_dict[i] = [os.path.join(subdir, fname)]
        if not found:
            filename_dict[i] = []

    while True:
        # shuffle the samples
        index_array = np.random.permutation(nb_sample)
        index_classes = classes[index_array]

        batch_x = None
        batch_y = np.zeros((batch_size, nb_class), dtype=np.float32)
        # start building the mini-batch
        i_patch = 0
        stop = batch_size // nb_class
        for i_class in range(nb_class):
            curr_index = index_array[index_classes == i_class]
            patch_cc, index_cc = 0, 0
            while patch_cc < stop:
                j = curr_index[index_cc]
                if filename_dict[j] == []:
                    index_cc += 1
                else:
                    fname = np.random.choice(filename_dict[j])
                    # read in the RGB image
                    pil_img = image.load_img(os.path.join(image_dir, fname),
                                             grayscale=False,
                                             target_size=(input_dim, input_dim))
                    img = image.img_to_array(pil_img)

                    if data_augmentation:
                        img = random_crop(img, random_crop_size=(input_dim-10, input_dim-10))
                        img = random_transform(img,
                                               dim_ordering='tf',
                                               rotation_range=30.,
                                               channel_shift_range=20.,
                                               fill_mode='reflect',
                                               horizontal_flip=True,
                                               vertical_flip=True
                                              )
                    # center crop anyway
                    x= center_crop(img, center_crop_size=(crop_dim, crop_dim))

                    if i_patch == 0:
                        batch_x = np.zeros((batch_size,) + x.shape)
                    batch_x[i_patch] = x
                    batch_y[i_patch, i_class] = 1.
                    # update the counters
                    i_patch += 1
                    index_cc += 1
                    patch_cc += 1
        batch_x, batch_y = batch_x[:i_patch], batch_y[:i_patch]

        # optionally save augmented images to disk for debugging purposes
        current_batch_size = i_patch
        if save_to_dir is not None:
            for i in range(current_batch_size):
                img = image.array_to_img(batch_x[i], scale=False)
                fname = '{prefix}_{hash}.{format}'.format(prefix='patch',
                                                          hash=np.random.randint(1e4),
                                                          format='jpg')
                img.save(os.path.join(save_to_dir, fname))

        # optional data-scaling
        if preprocess_mode == 'tf':
            batch_x = preprocess_input_tf(batch_x)
        elif preprocess_mode == 'caffe':
            batch_x = preprocess_input_caffe(batch_x)

        # return a mini-batch
        yield batch_x, batch_y











