import tensorflow as tf
import os
import matplotlib.pyplot as plt 
from skimage.io import imread

def plot_train_image(img_data,path):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for i, (index, row) in enumerate(img_data.iterrows()):
        ax = axes[i // 3, i % 3]
        image_path = os.path.join(path, row['ImageId'])
        ax.imshow(imread(image_path))
        ax.axis('off')
    plt.show()
    
def plot_images_masks(images, masks):
    """Plot images and masks in a batch"""
    n = len(images)
    fig, axarr = plt.subplots(n, 2, figsize=(10, 4 * n)) 
    
    for idx, (img, mask) in enumerate(zip(images, masks)):
        if n == 1:  
            ax = axarr
        else:
            ax = axarr[idx]
        
        ax[0].imshow(img)
        ax[0].set_title(f'Image {idx+1}')
        ax[0].axis('off')
        ax[1].imshow(mask.squeeze(), cmap='gray')  
        ax[1].set_title(f'Mask {idx+1}')
        ax[1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_images_masks_augmentation(images, masks, num=5):
    fig, axs = plt.subplots(num, 2, figsize=(5, 5))
    for idx, ax in enumerate(axs):
        img = images[idx]
        mask = masks[idx]
        ax[0].imshow(img)
        ax[0].set_title(f'Image {idx+1}')
        ax[0].axis('off')
        ax[1].imshow(tf.squeeze(mask).numpy(), cmap='gray') 
        ax[1].set_title(f'Mask {idx+1}')
        ax[1].axis('off')
        
    plt.tight_layout()
    plt.show()



def learning_curve_by_history(history):

    if not type(history) == dict:
        raise TypeError("history must be dict ")

    loss_keys = [name for name in history.keys() if 'loss' in name]
    if len(loss_keys) < 1:
        raise TypeError("Dict history must have key 'loss' ")
    score_keys = [name for name in history.keys() if 'loss' not in name]

    if len(score_keys) > 0:
        fig = plt.gcf()
        fig.set_size_inches(fig.get_figwidth() * 1.5, fig.get_figheight() * 2);
        
        fig.subplots(2, 1, sharex=True)
        plt.subplot(2, 1, 1).xaxis.set_label_position('top')
        plt.tick_params(labelbottom = False, bottom=False,
                        top = True, labeltop=True)

    for key in loss_keys:
        plt.plot(range(1, len(history[key]) + 1), history[key])
    plt.ylabel('losses')
    plt.xlabel('epoch')
    plt.legend(loss_keys, loc='upper right')
    plt.grid()
    plt.xlim(left=1)

    if len(score_keys) > 0:
        plt.subplot(2, 1, 2)
        for key in score_keys:
            plt.plot(range(1, len(history[key]) + 1), history[key])
        plt.ylabel('metrics')
        plt.xlabel('epoch')
        plt.legend(score_keys, loc='lower right')
        plt.grid()
        plt.xlim(left=1)

        plt.subplots_adjust(hspace=0)

    plt.show()
    return