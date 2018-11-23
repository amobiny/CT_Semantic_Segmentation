import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from utils.plot_utils import label_to_color_image

all_files = glob.glob('/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/'
                      'data_preparation/our_data/4_correctMask_normalized/no_spleen/*h5')
LABELS = ['background', 'liver', 'spleen', 'kidney', 'bone', 'vessel']


def vis_segmentation(image, seg_map, label_names=None, image_name=None):
    """Visualizes input image, segmentation map and overlay view."""
    FULL_LABEL_MAP = np.arange(len(label_names)).reshape(len(label_names), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
    plt.figure(figsize=(10, 5))
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])

    # plot input image
    plt.subplot(grid_spec[0])
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('input image')

    # plot ground truth mask
    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map.astype(np.int32)).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('ground truth map')

    unique_labels = np.unique(seg_map).astype(np.int32)
    ax = plt.subplot(grid_spec[2])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), [label_names[u] for u in unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig(image_name)


for i, file_path in enumerate(all_files):
    file_name = file_path.split('/')[-1].split('.')[0]
    dest_path = '/home/cougarnet.uh.edu/amobiny/Desktop/CT_Semantic_Segmentation/' \
                'data_preparation/our_data/4_correctMask_normalized/no_spleen_images/' + file_name
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    h5f = h5py.File(file_path, 'r')
    x_norm = h5f['x_norm'][:]
    y = h5f['y'][:]
    h5f.close()
    depth = y.shape[-1]
    slices = np.linspace(20, depth - 20, 50).astype(int)
    x_plot = np.transpose(np.squeeze(x_norm[:, :, :, slices]), [-1, 0, 1])
    y_plot = np.transpose(np.squeeze(y[:, :, :, slices]), [-1, 0, 1])
    for num, x, y in zip(slices, x_plot, y_plot):
        path = os.path.join(dest_path, str(num))
        vis_segmentation(x, y, label_names=LABELS, image_name=path)

print()




