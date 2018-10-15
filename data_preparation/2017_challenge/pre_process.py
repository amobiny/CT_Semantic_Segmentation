import os
import matplotlib.pyplot as plt
from tqdm import tqdm

mask = True
if mask:
    import slice_with_mask
else:
    import slice
# import wrap_data
from collections import OrderedDict

root_path = "/home/cougarnet.uh.edu/amobiny/Desktop/DATASETS/Lung_CT_Segmentation_Challenge_2017/LCTSC/" \
            "LCTSC-Train-S3-012/05-04-2004-NM PET SCAN RADIATION-26270"
patient_list = sorted(os.listdir(root_path))
Save = False
ill_people = {}
normal_people = []


def get_data():
    for i in tqdm(range(len(patient_list))):
        patient = patient_list[i]
        # if int(patient.split('-')[-1]) < 371:
        # if patient != 'LUNG1-361':
        #     continue
        a = os.listdir(root_path + '/' + patient)
        if len(a) == 1:
            folders = os.path.join(root_path + '/' + patient, a[0])
            folders_base = os.listdir(folders)
            if len(folders_base) == 2:
                file = os.listdir(os.path.join(folders, folders_base[0]))
                if len(file) == 1:
                    contour_path = folders + '/' + folders_base[0] + '/' + file[0]
                    data_path = os.path.join(folders, folders_base[1])
                else:
                    file = os.listdir(os.path.join(folders, folders_base[1]))
                    data_path = os.path.join(folders, folders_base[0])
                    contour_path = folders + '/' + folders_base[1] + '/' + file[0]
                if mask:
                    sliced_img = slice_with_mask.get_image(data_path, contour_path, patient, Save)
                else:
                    sliced_img = slice.get_image(data_path, contour_path, patient, Save)
                if type(sliced_img) == int:
                    if sliced_img == -1:
                        print("missing contour: " + patient + "\n")
                    elif sliced_img == -2:
                        print("no data for this patient: " + patient + "\n")
                else:
                    ill_people[patient] = sliced_img
            else:
                data_path = os.path.join(folders, folders_base[0])
                if Save:
                    if mask:
                        slice_with_mask.save_image_normal(data_path, patient)
                    else:
                        slice.save_image_normal(data_path, patient)
                normal_people.append(patient)
        else:
            print("This file is broken: " + patient + "\n")

    print("finish pre-process the dataset!\n")

    # ill_people.sort(key=lambda x: x.split("-")[-1])
    ill_people_sorted = OrderedDict(sorted(ill_people.items(), key=lambda t: int(t[0].split("-")[-1])))
    training_data, training_label, test_data, test_label = wrap_data.get_data(ill_people_sorted, normal_people)

    return training_data, training_label, test_data, test_label

print("")