import numpy as np
import skimage
import skimage.segmentation as seg
import skimage.measure as mea
import skimage.morphology as morph
import tifffile as tiff

import argparse


def correct_detection(detection, c=0):
    """
    # See skimage.measure.regionprops: measure properties of labeled image regions
    props = mea.regionprops(detection)
    for p in props:
        print(
            f"Label {p.label}\tArea={p.area}"
        )  # if the a label is repeated several times, we can not see these instances, but only one entry
    """
    # First, skimage.morphology.remove_small_objects, if needed
    # This method removes objects smaller than the specified size
    detection = morph.remove_small_objects(detection, connectivity=8)

    labels = np.unique(detection)[1:]  # remove the background value
    # labels, counts = np.unique(detection, return_counts=True) # it returns not only the labels, but also the number of pixels with that value

    for label in labels:
        label_map = detection == label  # list of lists of Boolean values
        relabeled = mea.label(label_map)
        sublabels = np.unique(relabeled)[1:]  # remove the background value

        if np.unique(sublabels).size == 1:
            continue

        areas = []
        for sublabel in sublabels:
            area = np.sum(relabeled == sublabel)
            areas.append(area)
        largest = np.argmax(areas)
        label_of_largest = sublabels[largest]
        # Remove in detection all sublabels but sublabels[largest]
        for sublabel in sublabels:
            if sublabel == label_of_largest:
                continue
            detection[relabeled == sublabel] = 0

    final_detection, _, _ = seg.relabel_sequential(detection)
    final_detection[final_detection > 0] += c  # Apply cumulative offset

    max_label = final_detection.max()

    return final_detection, max_label


def main(args):
    path = args.input

    if ".tif" in path:
        detection = tiff.imread(path)
        if detection.ndim == 4 and detection.shape[-1] == 3:
            # if 3D mask and RGB image
            tp = detection.dtype
            detection = np.dot(detection[..., :3], [0.2989, 0.5870, 0.1140]).astype(
                tp
            )  # it casts a RGB image into a grayscale one
        detection = detection.squeeze()
    elif ".jpg" in path:
        detection = skimage.io.imread(path, as_gray=True)

    if detection.ndim >= 3:  # 3D images
        print(f"Shape: {detection.ndim}, {detection.shape}")
        final_detection = []
        print("XY")
        c = 0  # counter to avoid repeated labels, it allows to label sequentially the objects
        for z in range(detection.shape[0]):  # on XY plane
            print(f"{z+1}/{detection.shape[0]}")
            corrected_slice, c = correct_detection(detection[z, :, :], c)
            final_detection.append(corrected_slice)
        final_detection = np.array(final_detection)
        # final_detection = np.stack(final_detection, axis=0)
    elif detection.ndim == 2:  # 2D images
        final_detection, _ = correct_detection(detection)
    else:
        print("Error in shape")
        return

    tiff.imwrite(
        f"{path.replace('.tiff','') if '.tiff' in path else path.replace('.tif','')}_lab_corrected.tif",
        final_detection,
        imagej=not args.no_imagej,
    )
    # tiff.imsave is deprecated

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", type=str, default=None, help="Path for the input file"
    )
    parser.add_argument(
        "--no_imagej",
        action="store_true",
        help="Store the output not for ImageJ (tifffile.imwrite option)",
    )

    args = parser.parse_args()

    main(args)
