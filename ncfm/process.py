import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import uniform_filter
from scipy import ndimage as ndi
from skimage.feature import match_descriptors, ORB
from skimage.feature import hog, daisy, CENSURE
from skimage import color, exposure, transform
from skimage.transform import pyramid_gaussian
# from skimage.util.montage import montage2d
from skimage.filters import gabor_kernel
from sklearn.feature_extraction.image import extract_patches_2d
# from numba import jit


def extract_features(imgs, feature_fns, verbose=True):
    """
    Given pixel data for images and several feature functions that can
    operate on single images, apply all feature functions to all
    images, concatenating the feature vectors for each image and
    storing the features for all images in a single matrix.

    Inputs:
    - imgs: N x H X W X C array of pixel data for N images.
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (F_1 + ... + F_k, N) where each column is the
    concatenation
    of all features for a single image.

    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature func. must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((total_feature_dim, num_images))
    imgs_features[:total_feature_dim, 0] = np.hstack(first_image_features)

    # Extract features for the rest of the images.
    for i in xrange(1, num_images):
        # idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            # next_idx = idx + feature_dim
            # imgs_features[idx:next_idx, i] = feature_fn(imgs[i].squeeze())
            # idx = next_idx
            imgs_features[:, i] = feature_fn(imgs[i].squeeze())
        if verbose and i % 100 == 0:
            print("Done extracting features for {}/{} images"
                  .format(i, num_images))

    return imgs_features.T


def rgb2gray(img):
    """Convert RGB image to grayscale

    Parameters:
      rgb : RGB image

    Returns:
      gray : grayscale image

    """
    return np.dot(img[..., :3], [0.299, 0.587, 0.144])


def padding_imgs(img, max_width=0, max_height=0):
    w, h, c = img.shape
    img = rgb2gray(img)
    if max_width != 0 and max_height != 0:
        if max_width > max_height:
            img = np.pad(img, ((0, max_width - w), (max_width - h, 0)),
                         'constant', constant_values=0)
        else:
            img = np.pad(img, ((0, max_height - w), (max_height - h, 0)),
                         'constant', constant_values=0)
    else:
        if h > w:
            img = np.pad(img, ((0, h-w), (0, 0)), 'constant',
                         constant_values=0)
        else:
            img = np.pad(img, ((0, 0), (w-h, 0)), 'constant',
                         constant_values=0)

    return img


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb square image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
        image = rgb2gray(im)
        # if im.ndim[0] > im.ndim[1] or im.ndim[1] > im.ndim[0]:
        #     image = padding_imgs(im)
        # w, h = image.shape
        # if h > w:
        #     image = np.pad(image, ((0, h-w), (0, 0)), 'constant',
        #                    constant_values=0)
        # else:
        #     image = np.pad(image, ((0, 0), (w-h, 0)), 'constant',
        #                    constant_values=0)
    else:
        image = np.atleast_2d(im)

    sx, sy = image.shape  # image size
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(
            temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin+1)
    hsv = matplotlib.colors.rgb_to_hsv(im/xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0],
                                     bins=bins,
                                     density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    return imhist.ravel()


def histogram_equalization(img):
    # Contrast stretching
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(img)

    # Adaptive equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    return img_rescale, img_eq, img_adapteq


def plot_hog(img):
    image = color.rgb2gray(img)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    fig, (ax1, ax2) = plt.subplots(1, 2,
                                   figsize=(8, 4),
                                   sharex=True,
                                   sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image,
                                                    in_range=(0, 0.02))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.show()


def pyramid(img):
    w, h, c = img.shape
    pyramid = tuple(pyramid_gaussian(img, downscale=2))
    composite_image = np.zeros((w, h + h // 2, 3), dtype=np.float32)
    composite_image[:w, :h, :] = pyramid[0]

    row_count = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[row_count:row_count + n_rows, h: h + n_cols] = p
        row_count += n_rows

    return composite_image


def daisy_feat(img):
    img = color.rgb2gray(img)
    descs, descs_img = daisy(img, step=180, radius=58, rings=2,
                             histograms=6,
                             orientations=8, visualize=True)

    return descs.ravel()


def censure(img):
    img = color.rgb2gray(img)
    # tform = tf.AffineTransform(scale=(1.5, 1.5), rotation=0.5,
    #                            translation=(150, -200))
    # img_warp = tf.warp(img, tform)
    detector = CENSURE()
    detector.detect(img)

    # return detector.keypoints, detector.scales
    return detector.scales


def orb(img):
    img1 = rgb2gray(img)
    img2 = transform.rotate(img1, 180)
    tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                                      translation=(0, -200))
    img3 = transform.warp(img1, tform)

    descriptor_extractor = ORB(n_keypoints=200)

    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img3)
    keypoints3 = descriptor_extractor.keypoints
    descriptors3 = descriptor_extractor.descriptors

    matches1 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    matches2 = match_descriptors(descriptors1, descriptors3, cross_check=True)

    return np.hstack((keypoints1[matches1[:, 0]].ravel(),
                      keypoints2[matches2[:, 1]].ravel()))
    # return descriptors1, descriptors2, descriptors3


def gabor_filters(img):
    """Prepare filter-bank kernels"""
    img = rgb2gray(img)
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma,
                                              sigma_y=sigma))
                kernels.append(kernel)

    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndi.convolve(img, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()

    return np.hstack((feats[:, 0].ravel(), feats[:, 1].ravel())).ravel()


def mean_removal(img):
    img[:, :, 0] -= 104.006
    img[:, :, 1] -= 116.669
    img[:, :, 2] -= 122.679

    return img


def remove_mean(X):
    mean_img = np.mean(X, axis=0)
    X -= mean_img

    return X


def whitening(X, k=8):
    # STEP 1a: Implement PCA to obtain the rotation matrix, U, which is
    # the eigenbases sigma.
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    Sigma = np.dot(X, X.T) / X.shape[1]  # [M x M]
    #  Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(Sigma)
    #  U: [M x M] eigenvectors of sigma.
    #  S: [M x 1] eigenvalues of sigma.
    #  V: [M x M] transpose of U

    # STEP 1b: Compute XRot, the projection on to the eigenbasis
    XRot = np.dot(U.T, X)

    # STEP 2: Reduce the number of dimensions from 2 to k
    XRot = np.dot(U[:, 0:k].T, X)
    XHat = np.dot(U[:, 0:k], XRot)

    # STEP 3: PCA Whitening
    #  Whitening constant: prevents division by zero
    epsilon = 1e-5
    # S * U'* X
    PCA_White = np.dot(np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T), X)

    # STEP 4: ZCA Whitening
    #  ZCA Whitening matrix: U * Lambda * U'
    # U * S * U'* X
    ZCA_White = np.dot(U, PCA_White)  # [M x M]

    return ZCA_White


def zca_whitening_matrix(X):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True)  # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U, S, V = np.linalg.svd(sigma)
    # U: [M x M] eigenvectors of sigma.
    # S: [M x 1] eigenvalues of sigma.
    # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))  # [M x M]

    return ZCAMatrix


# @jit
def sliding_window(img, stride=10, window_size=(20, 20)):
    """Extract patches according to a sliding window.

       Args:
       img (numpy array): The image to be processed.
       stride (int, optional): The sliding window stride (defaults to 10px).
       window_size (int, int, optional): The patch size (defaults to (20, 20)).

       Returns:
       list: list of patches with window_size dimensions
    """
    patches = []
    # slide window accross img
    for x in xrange(0, img.shape[0], stride):
        for y in xrange(0, img.shape[1], stride):
            new_patch = img[x:x + window_size[0], y:y + window_size[1]]
            if new_patch.shape[:2] == window_size:
                patches.append(new_patch)

    return patches


# def img_roi(imgs, bboxes, fname, img_names):

#     for bb_id, bb in enumerate(bboxes[fname]):
#         roi = imgs[img_names.index(fname)][
#             int(bb[1]):int(bb[1] + bb[3]),
#             int(bb[0]):int(bb[0] + bb[2]), :]

#     return roi


# @jit
def extract_rois(imgs, bboxes, img_names, img_labels):
    print("Extract ROIs from bounding boxes...")
    temp_lbl = np.argmax(img_labels, axis=1)
    labels = np.array([]).astype(np.int32)
    # rois = np.array([]).astype(np.float32)
    rois = []
    counter = 0
    for fname in bboxes:
        if len(bboxes[fname]) >= 1:
            labels = np.hstack((labels, np.repeat(
                temp_lbl[img_names.index(fname)],
                len(bboxes[fname]))
                                )
               )
            for bb_id, bb in enumerate(bboxes[fname]):
                roi = imgs[img_names.index(fname)][
                    int(bb[1]):int(bb[1] + bb[3]),
                    int(bb[0]):int(bb[0] + bb[2]), :]
                # if roi.shape[0] >= 64 and roi.shape[1] >= 64:
                # if counter == 0:
                #    rois = roi
                # else:
                #    rois = np.vstack((rois, roi))
                rois.append(roi)

                print("Image {} => trX_id index ({}/{}) bbox ({}/{})"
                      " label ({}/{}) nb_labels, roi {}"
                      .format(fname,
                              img_names.index(fname),
                              len(bboxes),
                              bb_id + 1,
                              len(bboxes[fname]),
                              temp_lbl[img_names.index(fname)],
                              labels.shape,
                              counter + 1))

                counter += 1
            np.delete(imgs, img_names.index(fname), axis=0)

    return rois, labels


def extract_patches_from_rois(rois, labels):
    targets = np.array([]).astype(np.int32)
    counter = 0
    for idx, roi in enumerate(rois):
        w, h, _ = roi.shape
        if w >= 64 and h >= 64:
            patch = extract_patches_2d(roi, (64, 64), random_state=2017)
            targets = np.hstack((targets, np.repeat(labels[idx],
                                                     patch.shape[0])))
            if counter == 0:
                patches = patch
            else:
                patches = np.vstack((patches, patch))

            print("patch {} => label ({}/{}) nb_labels, patch_shape {}"
                  .format(counter + 1,
                          labels[idx],
                          len(targets),
                          patches.shape))

            if counter == 185:
                break

            counter += 1

    return patches, targets


# @jit
def patches2d(imgs, bboxes, img_names, img_labels):
    print("Creating 2d-patches for each bbox...")
    temp_lbl = np.argmax(img_labels, axis=1)
    labels = np.array([]).astype(np.int32)
    patches = np.array([]).astype(np.float32)
    counter = 0
    for fname in bboxes:
        if len(bboxes[fname]) >= 1:
            for bb_id, bb in enumerate(bboxes[fname]):
                roi = imgs[img_names.index(fname)][
                    int(bb[1]):int(bb[1] + bb[3]),
                    int(bb[0]):int(bb[0] + bb[2]), :]
                if roi.shape[0] >= 64 and roi.shape[1] >= 64:
                    temp_patch = extract_patches_2d(roi, (64, 64),
                                                    random_state=2017)
                    if counter == 0:
                        patches = temp_patch
                    else:
                        patches = np.vstack((patches, temp_patch))

                    labels = np.hstack((labels,
                                        np.repeat(
                                            temp_lbl[img_names.index(fname)],
                                            temp_patch.shape[0])
                                        )
                                       )

                    print("Image {} => Index ({}/{}) bbox ({}/{})"
                          "label ({}/{}) nb_labels, patch {}"
                          .format(fname,
                                  img_names.index(fname),
                                  len(bboxes),
                                  bb_id + 1,
                                  len(bboxes[fname]),
                                  temp_lbl[img_names.index(fname)],
                                  labels.shape,
                                  counter + 1))

                    counter += 1
            np.delete(imgs, img_names.index(fname), axis=0)

    return patches, labels


def dwt2(img):
    """Multilevel 2D stationary wavelet transform.

       Parameters
       ----------
       data : array_like
           2D array with input datawavelet : Wavelet object or name string
           Wavelet to use
       level : int16    The number of decomposition steps to perform.
       start_level : int, optional
           The level at which the decomposition will start (default: 0)
       axes : 2-tuple of ints, optional
           Axes over which to compute the SWT. Repeated elements are not allowed.

       Returns
       -------
       coeffs : list
           Approximation and details coefficients::

               [
                   (cA_m,
                       (cH_m, cV_m, cD_m)
                   ),
                   (cA_m+1,
                       (cH_m+1, cV_m+1, cD_m+1)
                   ),
                   ...,
                   (cA_m+level,
                       (cH_m+level, cV_m+level, cD_m+level)
                   )
               ]

       where cA is approximation, cH is horizontal details, cV is
       vertical details, cD is diagonal details and m is ``start_level``"""

    cA, (cH, cV, cD) = pywt.dwt2(img, 'coif1')
    cA, (cH, cV, cD) = pywt.dwt2(cA, 'coif1')
    cA, (cH, cV, cD) = pywt.dwt2(cA, 'coif1')
    cA, (cH, cV, cD) = pywt.dwt2(cA, 'coif1')

    mean_coeff = np.mean(cH) + np.mean(cV) + np.mean(cD)
    std_coeff = np.std(cH) + np.std(cV) + np.std(cD)

    return np.stack((mean_coeff, std_coeff)).ravel()
