import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import gridspec
import os


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map_gt, seg_map_pred, label_names, image_name):
    """Visualizes input image, segmentation map and overlay view."""
    FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
    plt.figure(figsize=(20, 5))
    grid_spec = gridspec.GridSpec(1, 5, width_ratios=[6, 6, 6, 6, 1])

    # plot input image
    plt.subplot(grid_spec[0])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('input image')

    # plot ground truth mask
    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map_gt.astype(np.int32)).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('ground truth map')

    plt.subplot(grid_spec[2])
    seg_image = label_to_color_image(seg_map_pred.astype(np.int32)).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('prediction map')

    plt.subplot(grid_spec[3])
    plt.imshow(image, cmap='gray')
    plt.imshow(seg_image, alpha=0.4)
    plt.axis('off')
    plt.title('prediction overlay')

    unique_labels = np.unique(np.concatenate((np.unique(seg_map_gt), np.unique(seg_map_pred)), 0)).astype(np.int32)
    ax = plt.subplot(grid_spec[4])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), label_names[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig(image_name)


def plot_save_preds(images, masks, mask_preds, path, label_names):
    number = 0
    for image, mask, mask_pred in zip(images, masks, mask_preds):
        img_name = os.path.join(path, str(number)+'.png')
        vis_segmentation(image, mask, mask_pred, label_names, img_name)
        number += 1


if __name__ == '__main__':
    LABEL_NAMES = np.asarray(['background', 'liver', 'spleen', 'kidney', 'bone', 'vessel'])
    File_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/data_preparation/' \
                'our_data/4_correctMask_normalized/train/PV_anon_1579_5_232_ARLS1.h5'
    h5f = h5py.File(File_path, 'r')
    x = np.squeeze(h5f['x'][:])
    x_norm = np.squeeze(h5f['x_norm'][:])
    y = np.squeeze(h5f['y'][:])
    h5f.close()
    image = x_norm[:, :, 10]
    true_mask = y[:, :, 10]
    vis_segmentation(image, true_mask, true_mask, LABEL_NAMES)

print()
