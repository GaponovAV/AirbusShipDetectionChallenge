import tensorflow as tf
from scipy.ndimage import rotate


def custom_rotate(image, angle):
    return rotate(image, angle, reshape=False, mode='reflect')


def custom_shift(image, width_shift_range=0.1, height_shift_range=0.1):

    random_shift_layer = tf.keras.layers.experimental.preprocessing.RandomTranslation(
        height_factor=(-height_shift_range, height_shift_range),
        width_factor=(-width_shift_range, width_shift_range),
        fill_mode='reflect',
        interpolation='bilinear'
    )
    
    return random_shift_layer(image)


def custom_augmentation(image, mask):
    # Parameters
    rotation = 15.0
    shift = 0.1
    zoom = [0.9, 1.25]
    
    # Random rotation
    random_angles = tf.random.uniform(shape=(), minval=-rotation, maxval=rotation, dtype=tf.float32) * (3.14159 / 180)
    image = custom_rotate(image, random_angles)
    mask = custom_rotate(mask, random_angles)

    # Random horizontal flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        
    # Random vertical flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    # Random width & height shift
    image = custom_shift(image, shift, shift)
    mask = custom_shift(mask, shift, shift)
    
    # Random zoom
    zoom_level = tf.random.uniform(shape=(), minval=zoom[0], maxval=zoom[1], dtype=tf.float32)
    image_shape = tf.shape(image)
    new_shape = tf.cast(tf.cast(image_shape[:-1], tf.float32) * zoom_level, tf.int32)
    image = tf.image.resize(image, new_shape)
    mask = tf.image.resize(mask, new_shape)
    image = tf.image.resize_with_crop_or_pad(image, image_shape[0], image_shape[1])
    mask = tf.image.resize_with_crop_or_pad(mask, image_shape[0], image_shape[1])

    return image, mask

def custom_aug_gen(in_gen):
    for in_x, in_y in in_gen:
        aug_x, aug_y = [], []
        for x, y in zip(in_x, in_y):
            augmented_x, augmented_y = custom_augmentation(x, y)
            aug_x.append(augmented_x)
            aug_y.append(augmented_y)
        
        yield tf.stack(aug_x), tf.stack(aug_y)