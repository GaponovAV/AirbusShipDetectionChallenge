import keras.backend as K
import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np


def dice(targets, inputs, smooth=1e-6):
    targets = tf.cast(targets, tf.float32)
    inputs = tf.cast(inputs, tf.float32)
    
    axis = [1, 2, 3]
    intersection = K.sum(targets * inputs, axis=axis)
    dice_f = (2 * intersection + smooth) / (K.sum(targets, axis=axis) + K.sum(inputs, axis=axis) + smooth)
    return dice_f


def dice_loss(targets, inputs, smooth=1e-6):
    dice_coeff = dice(targets, inputs, smooth)
    return 1 - dice_coeff

def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred)
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)




def weighted_binary_crossentropy(targets, inputs, weight=10.0):
    bce = tf.keras.losses.binary_crossentropy(targets, inputs)
    mask = tf.where(tf.equal(targets, 1), weight, 1.0)
    weighted_bce = bce * mask
    return tf.reduce_mean(weighted_bce)


def weighted_dice_loss(targets, inputs, weight=10.0, smooth=1e-6):
    intersection = tf.reduce_sum(targets * inputs)
    mask_sum = tf.reduce_sum(targets * weight) + tf.reduce_sum(inputs)
    dice_coeff = (2. * intersection + smooth) / (mask_sum + smooth)
    return 1 - dice_coeff

