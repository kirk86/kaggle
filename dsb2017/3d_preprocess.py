import SimpleITK as sitk
import numpy as np
# import csv
import scipy
# from glob import glob
import pandas as pd
from scipy import ndimage
# from tqdm import tqdm
# import skimage
import os
from skimage import morphology, segmentation
from skimage.morphology import ball, disk, dilation, binary_erosion
from skimage.morphology import remove_small_objects, erosion, closing
from skimage.morphology import reconstruction, binary_closing
# from skimage.measure import label, regionprops, perimeter
# from skimage.morphology import binary_dilation, binary_opening
# from skimage.filters import roberts, sobel
# from skimage import measure, feature
# from skimage.segmentation import clear_border
# from skimage import data
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import dicom
import scipy.misc
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
# from numba import autojit
# from PIL import Image
# import cv2
from utils_3d import *


# In[7]:

def load_train():
    data_path = src
    folders = [x for x in os.listdir(data_path) if 'subset' in x]
    os.chdir(data_path)
    patients = []

    for i in folders:
        os.chdir(data_path + i)
        print('Changing folder to: {}'.format(data_path + i))
        patient_ids = [x for x in os.listdir(data_path + i) if '.mhd' in x]
        for id in patient_ids:
            j = '{}/{}'.format(i, id)
            patients.append(j)
    return patients


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return(f)


def plot_ct_scan(scan):
    f, plots = plt.subplots(int(scan.shape[0] / 20) + 1, 4, figsize=(25, 25))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('off')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)


def print_mask(lung_m, nodule_m):
    fig, ax = plt.subplots(1, 2, figsize=(20, 16))
    ax[0].imshow(lung_m, cmap=plt.cm.bone)
    ax[1].imshow(nodule_m, cmap=plt.cm.bone)
    # return


# In[8]:

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])


def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates


def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


# In[9]:

def generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros(image.shape, dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128
    return marker_internal, marker_external, marker_watershed


def get_segmented_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)

    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    #blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 14) # <- retains more of the area, 12 works well. Changed to 14, 12 still excluded some parts.
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

    #Apply the lungfilter (note the filtered areas being assigned threshold_min HU)
    segmented = np.where(lungfilter == 1, image, threshold_min*np.ones(image.shape))

    #return segmented, lungfilter, outline, watershed, sobel_gradient, marker_internal, marker_external, marker_watershed
    return segmented


def segment_lung_from_ct_scan(ct_scan):
    return np.asarray([get_segmented_lungs(slice) for slice in ct_scan])


def get_pixels_hu(image):
    image = image.astype(np.int16)
    image[image == threshold_min] = 0
    return np.array(image, dtype=np.int16)


def get_nodule_slices(lung_mask, nodule_mask, lung_raw):
    indexes = []
    for img_slice in range(nodule_mask.shape[0]):
        if np.max(nodule_mask[img_slice]) == 1.0:
            indexes.append(img_slice)

    print('Nodule_present on slices: {}'.format(indexes))
    lung_mask_pres = lung_mask[indexes, :, :]
    nod_mask_pres = nodule_mask[indexes, :, :]
    lung_raw_pres = lung_raw[indexes, :, :]
    return lung_mask_pres, nod_mask_pres, lung_raw_pres


def reshape_3d(image_3d):
    reshaped_img = image_3d.reshape([image_3d.shape[0], 1, height_mask, width_mask])
    print('Reshaped image shape:', reshaped_img.shape)
    return reshaped_img


def create_masks_for_patient_watershed(img_file, save=True):

    def draw_nodule_mask(node_idx, cur_row):

        #print('Working on node: {}, row: {}'.format(node_idx, cur_row), '\n')
        coord_x = cur_row["coordX"]
        coord_y = cur_row["coordY"]
        coord_z = cur_row["coordZ"]
        diam = cur_row["diameter_mm"]
        radius = np.ceil(diam/2)

        noduleRange = seq(-radius, radius, RESIZE_SPACING[0])
        print('Nodule range:', noduleRange)

        #imgs = np.zeros([noduleRange, height, width],dtype=np.float32)
        #print('imgs shape:', imgs.shape)
        #masks = np.zeros([noduleRange, height, width],dtype=np.float32)
        #print('masks shape:', masks.shape)

        world_center = np.array((coord_z,coord_y,coord_x))   # nodule center
        voxel_center = world_2_voxel(world_center, origin, new_spacing)

        image_mask = np.zeros(lung_img.shape)

        for x in noduleRange:
            for y in noduleRange:
                for z in noduleRange:
                    coords = world_2_voxel(np.array((coord_z+z,coord_y+y,coord_x+x)),origin,new_spacing)
                    if (np.linalg.norm(voxel_center - coords) * RESIZE_SPACING[0]) < radius:
                        image_mask[int(np.round(coords[0])),int(np.round(coords[1])),int(np.round(coords[2]))] = int(1)
        print(np.max(image_mask))

        return image_mask


    print("Getting mask for image file {}".format(img_file))
    patient_id = img_file.split('/')[-1][:-4]
    mini_df = df_node[df_node["file"] == img_file]

    if mini_df.shape[0] > 0: # some files may not have a nodule--skipping those

        img, origin, spacing = load_itk(src + img_file)
        height, width = img.shape[1], img.shape[2]

        #calculate resize factor
        #RESIZE_SPACING = [1, 1, 1]
        resize_factor = spacing / RESIZE_SPACING
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / img.shape
        new_spacing = spacing / real_resize

        lung_img = scipy.ndimage.interpolation.zoom(img, real_resize)
        print('Original image shape: {}'.format(img.shape))
        print('Resized image shape: {}'.format(lung_img.shape))


        lung_img = get_pixels_hu(lung_img)
        #lung_mask = segment_lung_from_ct_scan(lung_img)
        #lung_mask[lung_mask >= threshold_max] = threshold_max
        #lung_img[lung_img >= threshold_max] = threshold_max
        #lung_img[lung_img == 0] = threshold_min

        lung_mask = lung_img.copy()
        #lung_mask[lung_mask == 0] = threshold_min
        lung_mask[lung_mask >= threshold_max] = threshold_max
        lung_img[lung_img >= threshold_max] = threshold_max


        lung_imgs_512 = np.zeros([len(mini_df), lung_img.shape[0], 1, height_mask, width_mask], dtype = np.float32)
        lung_masks_512 = np.zeros([len(mini_df), lung_img.shape[0], 1, height_mask, width_mask], dtype = np.float32)
        nodule_masks_512 = np.zeros([len(mini_df), lung_img.shape[0], 1, height_mask, width_mask], dtype = np.float32)

        i = 0
        for node_idx, cur_row in mini_df.iterrows():

            nodule_mask = draw_nodule_mask(node_idx, cur_row)
            lung_img_512, lung_mask_512, nodule_mask_512 = np.zeros((lung_img.shape[0], height_mask, width_mask)), \
            np.zeros((lung_mask.shape[0], height_mask, width_mask)), \
            np.zeros((nodule_mask.shape[0], height_mask, width_mask))
            lung_mask_512[lung_mask_512 == 0] = threshold_min
            lung_img_512[lung_img_512 == 0] = threshold_min

            original_shape = lung_img.shape
            for z in range(lung_img.shape[0]):

                offset = (height_mask - original_shape[1])
                upper_offset = int(np.round(offset/2))
                lower_offset = int(offset - upper_offset)

                new_origin = voxel_2_world([-upper_offset,-lower_offset,0],origin,new_spacing)

                lung_img_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_img[z,:,:]
                lung_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = lung_mask[z,:,:]
                nodule_mask_512[z, upper_offset:-lower_offset,upper_offset:-lower_offset] = nodule_mask[z,:,:]

            print('Offsets shape for node index {} - main: {}, upper: {}, lower: {}'.format(node_idx, offset, upper_offset, lower_offset), '\n')

            #lung_mask_pres, nod_mask_pres, lung_raw_pres = get_nodule_slices(lung_mask_512, nodule_mask_512, lung_img_512)

            lung_mask_pres = reshape_3d(lung_mask_512)
            nod_mask_pres = reshape_3d(nodule_mask_512)
            if np.max(nod_mask_pres) == 0:
                print('Nodule lost')
            lung_mask_pres[lung_mask_pres <= threshold_min] = threshold_min
            lung_mask_pres[lung_mask_pres >= threshold_max] = threshold_max

            lung_mask_preproc = my_PreProc(lung_mask_pres)
            lung_mask_preproc = lung_mask_preproc.astype(np.float32)
            nod_mask_pres = (nod_mask_pres > 0.0).astype(np.float32)
            nod_mask_pres[nod_mask_pres == 1.0] = 255.

            patient_id_node_idx = '{}_{}'.format(patient_id, node_idx)
            #np.save('{}/lung_mask/{}_{}'.format(dst_nodules, patient_id, node_idx), lung_mask_pres)
            np.save('{}/lung_mask/{}_{}'.format(dst_nodules, patient_id, node_idx), lung_mask_preproc)
            np.save('{}/nodule_mask/{}_{}'.format(dst_nodules, patient_id, node_idx), nod_mask_pres)

        #return lung_mask_pres, nod_mask_pres
        return

    else:
        print('\n', 'No nodules found for patient: {}'.format(patient_id), '\n')
        return


annotations_path = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/evaluationScript/annotations/'

src = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/'
dst = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/Processed/'
dst_nodules = '/home/w/DS_Projects/Kaggle/DS Bowl 2017/LUNA/Data/nodules_3d/'


patients = load_train()
df_node = pd.read_csv(annotations_path+"annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(patients, file_name))
df_node = df_node.dropna()

threshold_min = -2000
threshold_max = 400
height_mask = 168
width_mask = 168
RESIZE_SPACING = [3, 3, 3]

Parallel(n_jobs=5)(delayed(create_masks_for_patient_watershed)(patient) for patient in df_node['file'].unique()[100:].tolist())
