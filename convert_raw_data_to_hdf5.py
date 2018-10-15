import os
import pydicom as dicom
import numpy as np
from slice_with_mask import get_image


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)


def load_scan(path):
    """
    loads all slices (corresponding to a scan) and sorts them
    :param path: path to all slices (dicom files) of a scan
    :return: sorted slices of the scan
    """
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: -x.ImagePositionPatient[-1])
    return slices


contour_path = '/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/Lung_CT_Segmentation_Challenge_2017/LCTSC/' \
               'LCTSC-Train-S3-012/05-04-2004-NM PET SCAN RADIATION-26270/1-.simplified-86523/000000.dcm'

image_path = '/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/Lung_CT_Segmentation_Challenge_2017/LCTSC/' \
             'LCTSC-Train-S3-012/05-04-2004-NM PET SCAN RADIATION-26270/1-64131'

get_image(image_path, contour_path, name='gi')

# patient = load_scan('/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/Lung_CT_Segmentation_Challenge_2017/LCTSC/LCTSC-Train-S3-012/05-04-2004-NM PET SCAN RADIATION-26270/1-.simplified-86523')
# imgs = get_pixels_hu(patient)
print()
