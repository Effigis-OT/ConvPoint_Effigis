#!/usr/bin/python()
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import warnings
import laspy
from pathlib import Path
from airborne_lidar_utils import write_features


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', default=None, help='Path to data folder')
    parser.add_argument("--dest", '-d', default=None, help='Path to destination folder')
    args = parser.parse_args()
    return args


def read_las_format(raw_path, normalize=True):
    """Extract data from a .las file.
    If normalize is set to True, will normalize XYZ and intensity between 0 and 1."""

    in_file = laspy.file.File(raw_path, mode='r')
    n_points = len(in_file)
    x = np.reshape(in_file.x, (n_points, 1))
    y = np.reshape(in_file.y, (n_points, 1))
    z = np.reshape(in_file.z, (n_points, 1))
    intensity = np.reshape(in_file.intensity, (n_points, 1))
    nb_return = np.reshape(in_file.num_returns, (n_points, 1))
    labels = np.reshape(in_file.classification, (n_points, 1))

    if normalize:
        # Converting data to relative xyz reference system.
        min_lbs = np.min(labels)
        max_lbs = np.max(labels)
        mask = (labels >= 0)
        x = x[mask].reshape((-1, 1))
        y = y[mask].reshape((-1, 1))
        z = z[mask].reshape((-1, 1))
        intensity = intensity[mask].reshape((-1, 1))
        nb_return = nb_return[mask].reshape((-1, 1))
        labels = labels[mask].reshape((-1, 1))
        norm_x = x - np.min(x)
        norm_y = y - np.min(y)
        norm_z = z - np.min(z)

        # Intensity is normalized based on min max values.
        norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        xyzni = np.hstack((norm_x, norm_y, norm_z, nb_return, norm_intensity)).astype(np.float16)
    else:
        xyzni = np.hstack((x, y, z, nb_return, intensity)).astype(np.float16)

    return xyzni, labels, n_points


def main():
    args = parse_args()
    base_dir = Path(args.folder)

    dataset_dict = {'trn': [], 'val': [], 'tst': []}

    # List .las files in each dataset.
    for dataset in dataset_dict.keys():
        for file in (base_dir / dataset).glob('*.las'):
            dataset_dict[dataset].append(file.name)
        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{base_dir / dataset} is empty")

    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n "
          f"Tst: {len(dataset_dict['tst'])}")

    # Write new hdfs of XYZ + number of return + intensity, with labels.
    for dst, values in dataset_dict.items():
        for elem in values:
            # make store directories
            path_prepare_label = Path(args.dest, dst)
            path_prepare_label.mkdir(exist_ok=True)

            xyzni, label, nb_pts = read_las_format(base_dir / dst / elem)

            write_features(f"{path_prepare_label / elem.split('.')[0]}_prepared.hdfs", xyzni=xyzni, labels=label)
            print(f"File {dst}/{elem} prepared. {nb_pts:,} points written.")


if __name__ == "__main__":
    main()
