# Airborne lidar example with ConvPoint

# add the parent folder to the python path to access convpoint library
import warnings
from pathlib import Path as P
import argparse
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
import time
import torch
import torch.utils.data
import torch.nn.functional as F
import h5py
from pathlib import Path
import wandb
import laspy
import socket
import torchvision
from platform import python_version

try:  # for local development at CRIM
    import src.nns.convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
    import src.utils.convpoint.metrics as metrics
    from src.utils.convpoint.airborne_lidar.airborne_lidar_utils import InformationLogger, \
        print_metric  # , write_config
    from src.utils.convpoint.airborne_lidar.airborne_lidar_viz import prediction2ply, error2ply
except:
    import convpoint.knn.lib.python.nearest_neighbors as nearest_neighbors
    import utils.metrics as metrics
    from examples.airborne_lidar.airborne_lidar_utils import InformationLogger, print_metric
    from examples.airborne_lidar.airborne_lidar_viz import prediction2ply, error2ply

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

#
from losses import FocalLoss, JointLoss
from optimizers import Lion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=True, help="Test the model")
    parser.add_argument("--use_wandb", action='store_true', help="Use wandb for logging")
    parser.add_argument("--wandb_project_name", default="convpoint_vexcel", type=str, help="Wandb project name")
    parser.add_argument("--loss", choices=['ce', "fl", "ce_fl"], required=True, help="Loss function to use")
    parser.add_argument("--fl_gamma", default=2.0, type=float, help="Focal loss gamma parameter")
    parser.add_argument("--weights", default=None, nargs='*', type=float, help="Classes weights for loss functions")
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd", "adamw", "lion"], type=str,
                        help="Optimizer to use")
    parser.add_argument("--savepts", action="store_true", help="Save the points in the ply files")
    parser.add_argument("--savedir", default=None, type=str, required=True, help="Directory to save the results")
    parser.add_argument("--rootdir", default=None, type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--testdir", default=None, type=str)
    parser.add_argument("--resdir", default=None, type=str)
    parser.add_argument("--batchsize", "-b", default=8, type=int, help="Batch size")
    parser.add_argument("--npoints", default=8168, type=int, help="Number of points to be sampled in the block.")
    parser.add_argument("--blocksize", default=25, type=int,
                        help="Size in meters of the infinite vertical column, to be processed.")
    parser.add_argument("--iter", default=600, type=int,
                        help="Number of mini-batches to run for training.")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--features", default="xyz", type=str,
                        help="Features to process. xyzni means xyz + number of returns + intensity. "
                             "Currently, only xyz and xyzni are supported for this dataset. Default is xyz.")
    parser.add_argument("--test_step", default=5, type=float,
                        help="Discretization step in meters applied at test time.")
    parser.add_argument("--test_labels", default=True, type=bool, help="Labels available for test dataset")
    parser.add_argument("--val_iter", default=60, type=int, help="Number of mini-batch iterations at validation.")
    parser.add_argument("--nepochs", default=20, type=int, help="Number of epochs to run for training.")
    parser.add_argument("--model", default="SegBig", type=str,
                        help="SegBig is the only available model at this time, for this dataset.")
    parser.add_argument("--model_state", default="init", type=str,
                        help="Model state file (pth).")
    parser.add_argument("--drop", default=0, type=float, help="Dropout rate.")

    parser.add_argument("--lr", default=1e-3, help="Learning rate")
    parser.add_argument("--mode", default=4, type=int,
                        help="Class mode:"
                             "1: building, water, ground."
                             "2: 5 classes: building, water, ground, low vegetation and medium + high vegetation"
                             "3: 6 classes: building, water, ground, low vegetation, medium and high vegetation"
                             "4: 8 classes: ground, vegetation, cars, trucks, power lines, fences, poles and buildings"
                             "5: 3 classes: ground, vegetation, building"
                             "6: 4 classes: ground, vegetation, building, pole"
                             "7: 5 classes: ground, vegetation, building, power line, pole")
    parser.add_argument("--dsname", default="",
                        help="Name of the dataset used for training")
    parser.add_argument("--finetune", default="NO",
                        help="Choose YES when the model must only be fine-tuned, freezing part of the layers")
    parser.add_argument("--decapit", default="NO",
                        help="Choose YES when the required classes are different from the one in the input model.")
    args = parser.parse_args()

    config = dict(blocksize=args.blocksize, npoints=args.npoints, architecture=args.model, epochs=args.nepochs)

    return args, config


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# wrap blue / green
def wblue(st):
    return bcolors.OKBLUE + st + bcolors.ENDC


def wgreen(st):
    return bcolors.OKGREEN + st + bcolors.ENDC


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def nearest_correspondance(pts_src, pts_dest, data_src, k=1):

    """ Find the nearest neighbors of pts_dest in pts_src

    Args:
        pts_src: (N, 3) array of source points
        pts_dest: (M, 3) array of destination points
        data_src: (N, C) array of source data
        k: number of nearest neighbors to find

    Returns:
        data_dest: (M, C) array of destination data

    pts_src and pts_dest are two numpy arrays containing the source and destination points respectively. These are
    the points that need to be matched together.
    data_src is a numpy array containing the data associated with each source point. This data will be used to create
    the output data for the matched destination points.
    k is an optional argument that specifies the number of nearest neighbors to consider when finding the matches
    between pts_src and pts_dest. By default, k is set to 1.

    It uses the knn function from the nearest_neighbors module to find the k nearest neighbors of each point
    in pts_src from pts_dest. The resulting indices of the nearest neighbors are stored in the indices variable.

    If k is equal to 1, then the indices array is flattened into a 1D array using the ravel method, and
    the corresponding data for each matched destination point is extracted from data_src using the flattened
    indices array. This means that the output data data_dest will have the same shape as pts_dest.

    If k is greater than 1, then the indices array is used to index into data_src to extract the data associated
    with the nearest neighbors for each point in pts_src. The mean of these data values is then computed along
    the second axis (i.e., the axis that corresponds to the k nearest neighbors) to create the output data data_dest.
    This means that the output data data_dest will have a shape that is the same as pts_dest, except for the
    second axis, which will be reduced by a factor of k.

    """
    print(pts_dest.shape)
    indices = nearest_neighbors.knn(pts_src.astype(np.float32), pts_dest.astype(np.float32), k, omp=True)
    print(indices.shape)
    if k == 1:
        indices = indices.ravel()
        data_dest = data_src[indices]
    else:
        data_dest = data_src[indices].mean(1)
    return data_dest


def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1], ])
    return np.dot(batch_data, rotation_matrix)


def class_mode(mode):
    """
    Dict containing the mapping of input (from the .las file) and the output classes (for the training step).
    ASPRS Codes used for this classification ASPRS 1 = Unclassified ASPRS 2 = Ground ASPRS 3 = Low Vegetation
    ASPRS 4 = Medium Vegetation ASPRS 5 = High Vegetation ASPRS 6 = Buildings ASPRS 7 = Low Noise
    ASPRS 8 = Model Key-Point ASPRS 9 = Water ASPRS 17 = Bridge ASPRS 18 = High Noise
    Entit
    """
    asprs_class_def = {'2': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                       '3': {'name': 'Low vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                       '4': {'name': 'Medium vegetation', 'color': [81, 163, 148], 'mode': 0},  # bluegreen
                       '5': {'name': 'High Vegetation', 'color': [108, 135, 75], 'mode': 0},  # dark green
                       '6': {'name': 'Building', 'color': [223, 52, 52], 'mode': 0},  # red
                       '9': {'name': 'Water', 'color': [95, 156, 196], 'mode': 0}  # blue
                       }
    dales_class_def = {'1': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                       '2': {'name': 'vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                       '3': {'name': 'cars', 'color': [255, 163, 148], 'mode': 0},  # bluegreen
                       '4': {'name': 'trucks', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '5': {'name': 'power lines', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '6': {'name': 'fences', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '7': {'name': 'poles', 'color': [255, 135, 75], 'mode': 0},  # dark green
                       '8': {'name': 'Building', 'color': [223, 52, 52], 'mode': 0}
                       }
    solarbresbati_class_def = {'2': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                               '5': {'name': 'Vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                               '6': {'name': 'Building', 'color': [255, 163, 148], 'mode': 0},  # bluegreen
                               }
    solarbresbatipoteau_class_def = {'2': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                                     '5': {'name': 'Vegetation', 'color': [77, 174, 84], 'mode': 0},  # bright green
                                     '6': {'name': 'Building', 'color': [255, 163, 148], 'mode': 0},  # bluegreen
                                     '16': {'name': 'Pole', 'color': [223, 52, 52], 'mode': 0}  # red
                                     }
    solarbresbatilignepoteau_class_def = {'2': {'name': 'Ground', 'color': [233, 233, 229], 'mode': 0},  # light grey
                                          '5': {'name': 'Vegetation', 'color': [77, 174, 84], 'mode': 0},
                                          # bright green
                                          '6': {'name': 'Building', 'color': [255, 163, 148], 'mode': 0},  # bluegreen
                                          '14': {'name': 'Power line', 'color': [255, 135, 75], 'mode': 0},
                                          # dark green
                                          '15': {'name': 'Pole', 'color': [223, 52, 52], 'mode': 0}  # red
                                          }

    coi = {}
    unique_class = []
    if mode == 1:
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3}
    elif mode == 2:
        # considering medium and high vegetation as the same class
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3, '3': 4, '4': 5, '5': 5}
    elif mode == 3:
        # considering medium and high vegetation as different classes
        asprs_class_to_use = {'6': 1, '9': 2, '2': 3, '3': 4, '4': 5, '5': 6}
    elif mode == 4:
        asprs_class_def = dales_class_def
        # ground(1), vegetation(2), car(3), truck(4), power line(5), fence(6), pole(7) and building(8)
        asprs_class_to_use = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
    elif mode == 5:
        asprs_class_def = solarbresbati_class_def
        # Ground(2), Tree(5), Building(6)
        asprs_class_to_use = {'2': 1, '5': 2, '6': 3}
    elif mode == 6:
        asprs_class_def = solarbresbatipoteau_class_def
        # Ground(2), Tree(5), Building(6), Pole(16)
        asprs_class_to_use = {'2': 1, '5': 2, '6': 3, '16': 4}
    elif mode == 7:
        asprs_class_def = solarbresbatilignepoteau_class_def
        # Ground(2), Tree(5), Building(6), Power line(14), Pole(15)
        asprs_class_to_use = {'2': 1, '5': 2, '6': 3, '14': 4, '15': 5}
    else:
        raise ValueError(f"Class mode provided ({mode}) is not defined.")

    for key, value in asprs_class_def.items():
        if key in asprs_class_to_use.keys():
            coi[key] = value
            coi[key]['mode'] = asprs_class_to_use[key]
            if asprs_class_to_use[key] not in unique_class:
                unique_class.append(asprs_class_to_use[key])

    nb_class = len(unique_class) + 1
    return {'class_info': coi, 'nb_class': nb_class}


# Part dataset only for training / validation
class PartDatasetTrainVal:

    def __init__(self, filelist, folder, training, block_size, npoints, iteration_number, features, class_info):

        self.filelist = filelist
        self.folder = Path(folder)
        self.training = training
        self.bs = block_size
        self.npoints = npoints
        self.iterations = iteration_number
        self.features = features
        self.class_info = class_info

    def __getitem__(self, index):

        # Load data
        index = random.randint(0, len(self.filelist) - 1)
        dataset = self.filelist[index]
        data_file = h5py.File(self.folder / f"{dataset}.hdfs", 'r')

        # Get the features
        xyzni = data_file["xyzni"][:]
        labels = data_file["labels"][:]
        labels = self.format_classes(labels)

        # pick a random point
        pt_id = random.randint(0, xyzni.shape[0] - 1)
        pt = xyzni[pt_id, :3]

        # Create the mask
        mask_x = np.logical_and(xyzni[:, 0] < pt[0] + self.bs / 2, xyzni[:, 0] > pt[0] - self.bs / 2)
        mask_y = np.logical_and(xyzni[:, 1] < pt[1] + self.bs / 2, xyzni[:, 1] > pt[1] - self.bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        pts = xyzni[mask]
        lbs = labels[mask]
        # print(pts.shape)
        # Random selection of npoints in the masked points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]
        lbs = lbs[choice]

        # Separate features from xyz
        if self.features is False:
            features = np.ones((pts.shape[0], 1))
        else:
            features = pts[:, 3:]
            features = features.astype(np.float32)
        pts = pts[:, :3]

        # Data augmentation (rotation)
        if self.training:
            pts = rotate_point_cloud_z(pts)

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(features).float()
        lbs = torch.from_numpy(lbs).long()

        return pts, fts, lbs

    def __len__(self):
        return self.iterations

    def format_classes(self, labels):
        """Format labels array to match the classes of interest.
        Labels with keys not defined in the coi dict will be set to 0.
        """
        labels2 = np.full(shape=labels.shape, fill_value=0, dtype=int)
        for key, value in self.class_info.items():
            labels2[labels == int(key)] = value['mode']

        return labels2


# Part dataset only for testing
class PartDatasetTest:

    def compute_mask(self, pt, bs):

        # This method builds a mask that identifies the points in a 3D point cloud that are within a certain distance
        # of a given point `pt`.
        # Inputs:
        #   - pt: a 3D point in the form of a tuple (x, y, z)
        #   - bs: the size of the box around the point `pt`
        # Output:
        #   - mask: a boolean array of length N that selects the points within the box around `pt`.

        # Select points whose x-coordinate is within the box
        mask_x = np.logical_and(self.xyzni[:, 0] < pt[0] + bs / 2, self.xyzni[:, 0] > pt[0] - bs / 2)
        # Select points whose y-coordinate is within the box
        mask_y = np.logical_and(self.xyzni[:, 1] < pt[1] + bs / 2, self.xyzni[:, 1] > pt[1] - bs / 2)
        # Combine the masks for x and y coordinates
        mask = np.logical_and(mask_x, mask_y)
        # Return the final mask
        return mask

    def __init__(self, filename, folder, block_size=8, npoints=8192, test_step=5, features=False, labels=True):

        """Initialize the dataset.

        Args:
            filename (str): the name of the file to load
            folder (str): the path to the folder containing the file
            block_size (int): the size of the blocks to extract
            npoints (int): the number of points to extract from each block
            test_step (int): the step to use when extracting blocks
            features (bool): whether to use the features or not
            labels (bool): whether to use the labels or not
        """

        self.filename = filename
        self.folder = Path(folder)
        self.bs = block_size
        self.npoints = npoints
        self.features = features
        self.step = test_step
        self.islabels = labels
        self.h5file = self.folder / f"{self.filename}.hdfs"
        # load the points
        with h5py.File(self.h5file, 'r') as data_file:
            self.xyzni = data_file["xyzni"][:]
            if self.islabels:
                self.labels = data_file["labels"][:]
            else:
                self.labels = None

            discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
            # discretized = ((self.xyzni[0:100, :2]).astype(float) / self.step).astype(int)
            self.pts = np.unique(discretized, axis=0)
            self.pts = self.pts.astype(np.float32) * self.step

    def __getitem__(self, index):

        if self.pts is None:
            with h5py.File(self.h5file, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
                if self.islabels:
                    self.labels = data_file["labels"][:]
                else:
                    self.labels = None

                discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
                self.pts = np.unique(discretized, axis=0)
                self.pts = self.pts.astype(np.float32) * self.step
        # get the data
        mask = self.compute_mask(self.pts[index], self.bs)
        pts = self.xyzni[mask]

        # choose right number of points
        choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
        pts = pts[choice]

        # indices in the original point cloud
        indices = np.where(mask)[0][choice]

        # separate between features and points
        if self.features is False:
            fts = np.ones((pts.shape[0], 1))
        else:
            fts = pts[:, 3:]
            fts = fts.astype(np.float32)

        pts = pts[:, :3].copy()

        pts = torch.from_numpy(pts).float()
        fts = torch.from_numpy(fts).float()
        indices = torch.from_numpy(indices).long()

        return pts, fts, indices

    def __len__(self):
        if self.pts is None:
            with h5py.File(self.h5file, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
                if self.islabels:
                    self.labels = data_file["labels"][:]
                else:
                    self.labels = None

                discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
                self.pts = np.unique(discretized, axis=0)
                self.pts = self.pts.astype(np.float32) * self.step
        return len(self.pts)


def get_model(nb_classes, args):
    """
    Get the model to be used for training.

    This is a Python function that returns a neural network model to be used for training.
    The function takes two arguments: nb_classes, which is the number of classes in the dataset, and args,
    which is a collection of arguments passed to the script.
    The function first checks the value of the args.model argument to determine which model to use.
    If args.model is "SegBig", the function imports the SegBig model from the networks.network_seg
    module. If args.model is any other value, the function raises a NotImplemented exception with an appropriate
    error message.
    The function then checks the value of the args.features argument to determine the number of input channels
    for the model. If args.features is "xyzni", the model will have two input channels, one for position (xyz)
    and one for intensity (i). If args.features is "xyz", the model will have one input channel for position only.
    If args.features is any other value, the function raises a NotImplemented exception with an appropriate
    error message.
    The function returns the created model along with a boolean variable features indicating whether the model
    uses intensity (True) or not (False) as input features.

    Args:
        nb_classes (int): Number of classes in the dataset.
        args (argparse.Namespace): Arguments passed to the script.

    Returns:
        model (torch.nn.Module): The model to be used for training.
    """
    # Select the model
    if args.model == "SegBig":
        try:
            # For CRIM
            from src.nns.convpoint.networks.network_seg import SegBig as Net
        except:
            from networks.network_seg import SegBig as Net
    else:
        raise NotImplemented(f"The model {args.model} does not exist. Only SegBig is available at this time.")

    # Number of features as input
    if args.features == "xyzni":
        input_channels = 2
        features = True
    elif args.features == "xyz":
        input_channels = 1
        features = False
    else:
        raise NotImplemented(f"Features {args.features} are not supported. Only xyzni or xyz, at this time.")

    return Net(input_channels, output_channels=nb_classes, args=args), features


def create_optimizer(net, nb_class, args):
    """
    Create an optimizer object to be used for training a neural network model.

    This is a Python function called create_optimizer() that creates an optimizer object to be used for training
    a neural network model. The function takes three parameters:
    net: the neural network model to be optimized
    nb_class: the number of classes in the dataset
    args: a set of arguments passed to the function that control various settings related to the optimizer and
    the training process.
    The function starts by checking the values of two arguments args.finetune and args.decapit.
    If either of them is set to "YES", the function proceeds to set up the optimizer for fine-tuning or decapitation
    of the neural network.
    The function then sets the leaning rate (base_lr) for each layer of the neural network, based on the number
    of parameters in that layer. It also freezes the parameters of the layers that are not being fine-tuned
    or decapitated.
    Next, the function sets up the parameters to be updated during training and chooses an optimizer based on
    the args.optimizer_selection argument.
    If neither args.finetune nor args.decapit is set to "YES", the function sets up the optimizer for training
    the entire neural network, and also chooses an optimizer based on the args.optimizer_selection argument.
    Finally, the function returns the optimizer object and a string typ_entr that describes the type of training
    being performed (fine-tuning, decapitation, or full training). The function also prints a message indicating
    which optimizer is being used.

    Args:
        net: the neural network model to be optimized
        nb_class: the number of classes in the dataset
        args: a set of arguments passed to the function that control various settings related to the optimizer and

    Returns:
        optimizer: the optimizer object
    """

    if args.finetune == "YES" or args.decapit == "YES":
        base_lr = float(args.lr)
        n_param = 0
        name_block = dict()
        lr_blcok = dict()
        for l1, (name, param) in enumerate(net.named_parameters()):
            n_param += 1
        for l2, (name, param) in enumerate(net.named_parameters()):
            blcok = name.split('.')[0]
            name_block[l2] = name
            lr_blcok[l2] = base_lr  # * 10**(2*(l2/n_param-1))
        params_to_update = []
        if args.finetune == "YES":
            n_limit_param = n_param / 2  # we will freeze the first half of the model (encoder)
            typ_entr = "f"
        else:
            n_limit_param = n_param - 2  # we freeze all the layers but the last 2 (classifier)
            typ_entr = "nf"
        for l3, (name, param) in enumerate(net.named_parameters()):
            if l3 < n_limit_param:
                param.requires_grad = False  # we are freezing this part of the net
            else:
                param.requires_grad = True  # modified layers
                # you can choose to apply different lr value for each layer
                params_to_update.append({"params": param, "lr": lr_blcok[l3], })
            # if param.requires_grad is True:
            #     print("\t",name)
        if args.decapit == "YES":
            # print("\nnb_class 1: {0}".format(nb_class))
            # print("net.fcout.weight.size 1: {0}".format(net.fcout.weight.size()))
            net.fcout = torch.nn.Linear(128, nb_class)
            typ_entr += "d"
        else:
            typ_entr += "nd"
        if args.optimizer == "adam":
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(params_to_update, lr=float(base_lr))
        elif args.optimizer == "sgd":
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(params_to_update, lr=float(base_lr), momentum=0.9)
        elif args.optimizer == "adamw":
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(params_to_update, lr=float(base_lr))
        elif args.optimizer == "lion":
            print("Using LION optimizer")
            optimizer = Lion(params_to_update, lr=float(base_lr))
        else:
            raise NotImplemented(
                f"Optimizer {args.optimizer_selection} is not supported. Only adam, sgd, adamw and lion are available at this time.")
    else:
        if args.optimizer == "adam":
            print("Using Adam optimizer")
            optimizer = torch.optim.Adam(net.parameters(), lr=float(args.lr))
        elif args.optimizer == "sgd":
            print("Using SGD optimizer")
            optimizer = torch.optim.SGD(net.parameters(), lr=float(args.lr), momentum=0.9)
        elif args.optimizer == "adamw":
            print("Using AdamW optimizer")
            optimizer = torch.optim.AdamW(net.parameters(), lr=float(args.lr))
        elif args.optimizer == "lion":
            print("Using LION optimizer")
            optimizer = Lion(net.parameters(), lr=float(args.lr))
        else:
            raise NotImplemented(
                f"Optimizer {args.optimizer_selection} is not supported. Only adam, sgd, adamw and lion are available at this time.")
        typ_entr = "nfnd"

    print("done")
    return optimizer, typ_entr


def create_loss_function(args):
    '''
    Create the loss function based on the arguments passed to the program.

    The function takes one argument, args, which is a collection of arguments passed to the program.
    The function first checks the value of the args.loss argument to determine which type of loss function to create.
    If args.loss is "ce", the function creates a cross-entropy loss function using
    the torch.nn.CrossEntropyLoss() method. If args.loss is "fl", the function creates a focal loss function
    using a custom implementation of Focal Loss. If args.loss is "ce_fl", the function creates a joint loss function
    which is a combination of cross-entropy loss and focal loss.
    The weight and gamma parameters of the loss functions are set based on the args.weights and args.fl_gamma arguments
    respectively. If the loss type is not one of the supported types ("ce", "fl", or "ce_fl"),
    the function raises a NotImplemented exception with an appropriate error message.
    The function returns the created loss function.

    Args:
        args: The arguments passed to the program.
    Returns:
        The loss function
    '''

    if args.loss == "ce":
        print("Using cross-entropy loss")
        loss_function = torch.nn.CrossEntropyLoss(weight=args.weights)
    elif args.loss == "fl":
        print("Using focal loss")
        loss_function = FocalLoss(gamma=args.fl_gamma, weight=args.weights)
    elif args.loss == "ce_fl":
        ce = torch.nn.CrossEntropyLoss(weight=args.weights)
        fl = FocalLoss(weight=args.weights, gamma=args.fl_gamma)
        loss_function = JointLoss(ce, fl, 0.5, 0.5)
        print("Using cross-entropy loss with focal loss")
    else:
        raise NotImplemented(f"Loss {args.loss} is not supported. Only ce, ce_fl and fl are available at this time.")

    return loss_function


def train(args, dataset_dict,
          info_class,
          time_string,
          use_wandb=False):
    '''
    Train the model.

    Args:
        args: The arguments passed to the program.
        dataset_dict: A dictionary containing the training and validation datasets.
        info_class: A dictionary containing the number of classes and the class names.
        time_string: A string containing the current time.
        use_wandb: A boolean indicating whether to use Weights and Biases.
    '''

    # Create the loss function
    criterion = create_loss_function(args)
    if args.loss == "ce":
        loss_name = "crossentropy"
    elif args.loss == "fl":
        loss_name = "focal"
    else:
        loss_name = "joint"

    print("Creating network...")
    if args.model_state != "init":
        state0 = torch.load(args.model_state)
        nb_class0 = state0['state_dict']['fcout.weight'].size()[0]
        net, features = get_model(nb_class0, args)
        net.load_state_dict(state0['state_dict'])
    else:
        nb_class0 = info_class['nb_class']
        net, features = get_model(nb_class0, args)

    # net.cuda()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    nb_class = info_class['nb_class']

    print("Creating dataloader and optimizer...", end="")
    ds_trn = PartDatasetTrainVal(filelist=dataset_dict['trn'], folder=args.rootdir, training=True,
                                 block_size=args.blocksize, npoints=args.npoints,
                                 iteration_number=args.batchsize * args.iter, features=features,
                                 class_info=info_class['class_info'])
    train_loader = torch.utils.data.DataLoader(ds_trn, batch_size=args.batchsize, shuffle=True,
                                               num_workers=args.num_workers)

    ds_val = PartDatasetTrainVal(filelist=dataset_dict['val'], folder=args.rootdir, training=False,
                                 block_size=args.blocksize, npoints=args.npoints,
                                 iteration_number=int(args.batchsize * args.iter * 0.1), features=features,
                                 class_info=info_class['class_info'])
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=args.batchsize, shuffle=False,
                                             num_workers=args.num_workers)

    assert len(val_loader) > 0, "The validation set is empty. Please check iter."

    # create the optimizer
    optimizer, typ_entr = create_optimizer(net, nb_class, args)

    net.cuda()

    # create the root folder
    print("Creating results folder...", end="")

    # root_folder = Path(f"{args.savedir}/{args.model}_{args.dsname}_{args.npoints}pts_{args.blocksize}blocksize_
    # {args.drop}drop_{args.lr}lr_{args.nepochs}epochs_{time_string}")
    # root_folder.mkdir(exist_ok=True)
    args_dict = vars(args)
    args_dict['data'] = dataset_dict
    # write_config(root_folder, args_dict)
    # print("done at", root_folder)

    # create the log file
    # trn_logs = InformationLogger(root_folder, 'trn')
    # val_logs = InformationLogger(root_folder, 'val')
    if use_wandb:
        wandb.watch(net)

    # iterate over epochs
    for epoch in range(args.nepochs):

        #######
        # training
        net.train()

        train_loss = 0
        cm = np.zeros((nb_class, nb_class))
        print("")
        print("")
        t = tqdm(train_loader, ncols=150, desc="Training Epoch {}".format(epoch))
        for pts, features, seg in t:
            features = features.cuda()
            pts = pts.cuda()
            seg = seg.cuda()

            optimizer.zero_grad()
            outputs = net(features, pts)
            # print("\nnb_class 2: {0}".format(nb_class))
            # print("net.fcout.weight.size 2: {0}".format(net.fcout.weight.size()))

            # compute the loss
            loss = criterion(outputs.view(-1, nb_class), seg.view(-1))
            loss.backward()
            optimizer.step()
            output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
            target_np = seg.cpu().numpy().copy()

            cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(nb_class)))
            cm += cm_

            oa = f"{metrics.stats_overall_accuracy(cm):.4f}"
            acc = metrics.stats_accuracy_per_class(cm)
            iou = metrics.stats_iou_per_class(cm)

            train_loss += loss.detach().cpu().item()

            t.set_postfix(OA=wblue(oa), AA=wblue(f"{acc[0]:.4f}"), IOU=wblue(f"{iou[0]:.4f}"),
                          LOSS=wblue(f"{train_loss / cm.sum():.4e}"))
            if use_wandb and 0:
                wandb.log({'trn/oa': cm, 'trn/loss': train_loss / cm.sum(), 'trn/iou': iou[0]})
        if use_wandb and 0:
            wandb.log({'trn/oa': float(oa), 'trn/loss': train_loss / cm.sum(), 'trn/iou': iou[0]})

        fscore = metrics.stats_f1score_per_class(cm)
        # trn_metrics_values = {'loss': f"{train_loss / cm.sum():.4e}", 'acc': acc[0], 'iou': iou[0],
        #                       'fscore': fscore[0]}
        # trn_class_score = {'acc': acc[1], 'iou': iou[1], 'fscore': fscore[1]}
        # trn_logs.add_metric_values(trn_metrics_values, epoch)
        # trn_logs.add_class_scores(trn_class_score, epoch)
        print_metric('Training', 'F1-Score', fscore)

        ######
        # validation
        net.eval()
        cm_val = np.zeros((nb_class, nb_class))
        val_loss = 0
        print("")
        t = tqdm(val_loader, ncols=150, desc="  Validation epoch {}".format(epoch))
        with torch.no_grad():
            for pts, features, seg in t:
                features = features.cuda()
                pts = pts.cuda()
                seg = seg.cuda()

                outputs = net(features, pts)
                loss = F.cross_entropy(outputs.view(-1, nb_class), seg.view(-1))

                output_np = np.argmax(outputs.cpu().detach().numpy(), axis=2).copy()
                target_np = seg.cpu().numpy().copy()

                cm_ = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(nb_class)))
                cm_val += cm_

                oa_val = f"{metrics.stats_overall_accuracy(cm_val):.4f}"
                acc_val = metrics.stats_accuracy_per_class(cm_val)
                iou_val = metrics.stats_iou_per_class(cm_val)

                val_loss += loss.detach().cpu().item()

                t.set_postfix(OA=wgreen(oa_val), AA=wgreen(f"{acc_val[0]:.4f}"), IOU=wgreen(f"{iou_val[0]:.4f}"),
                              LOSS=wgreen(f"{val_loss / cm_val.sum():.4e}"))
                if use_wandb and 0:
                    wandb.log({'val/oa': cm_val, 'val/loss': val_loss / cm_val.sum(), 'val/iou': iou_val[0]})
        if use_wandb and 0:
            wandb.log({'val/oa': float(oa_val), 'val/loss': val_loss / cm_val.sum(), 'val/iou': iou_val[0]})
        tloss = train_loss / cm.sum()
        vloss = val_loss / cm_val.sum()
        loss_difference = vloss - tloss

        if use_wandb:
            wandb.log({'trn/oa': float(oa), 'trn/loss': tloss, 'trn/iou': iou[0], 'val/oa': float(oa_val),
                       'val/loss': vloss, 'val/iou': iou_val[0], 'diff/loss_diff': loss_difference, 'epoch': epoch})

        fscore_val = metrics.stats_f1score_per_class(cm_val)

        if epoch == 0:
            min_val_loss = val_loss / cm_val.sum()
            save_model = True
        else:
            if val_loss / cm_val.sum() < min_val_loss:
                min_val_loss = val_loss / cm_val.sum()
                save_model = True
            else:
                save_model = False

        if save_model:
            # save the model
            state = {
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args
            }
            best_epoch = epoch
            mod_folder = P(args.savedir)
            mod_folder.mkdir(exist_ok=True, parents=True)
            last_saved_model = f"CP_{typ_entr}_{args.dsname}_{args.npoints}pts_" \
                               f"{args.blocksize}block_{args.drop}drop_{loss_name}loss_" \
                               f"{args.lr.replace('.', 'p')}lr_{args.nepochs}ep_" \
                               f"{args.iter}iter_batch{args.batchsize}" \
                               f"_{args.num_workers}work_{time_string}.pth"

            torch.save(state, os.path.join(mod_folder, last_saved_model))
            print("Model saved for epoch {0}".format(epoch))

        # write the logs
        # val_metrics_values = {'loss': f"{val_loss / cm_val.sum():.4e}", 'acc': acc_val[0], 'iou': iou_val[0],
        #                       'fscore': fscore_val[0]}
        # val_class_score = {'acc': acc_val[1], 'iou': iou_val[1], 'fscore': fscore_val[1]}

        # val_logs.add_metric_values(val_metrics_values, epoch)
        # val_logs.add_class_scores(val_class_score, epoch)
        print_metric('Validation', 'F1-Score', fscore_val)

    print("")
    print("Best epoch: {0}".format(best_epoch))

    return mod_folder, last_saved_model


def test(args, flist_test, model_folder, model_name, info_class, use_wandb=False):
    """
    Test the model

    Args:
        args: arguments
        flist_test: list of test files
        model_folder: folder where the model is saved
        model_name: name of the model
        info_class: dictionary with the number of classes
        use_wandb: use wandb to log the results

    """
    nb_class = info_class['nb_class']
    # create the network
    print("Creating network...")
    net, features = get_model(nb_class, args)
    state = torch.load(os.path.join(model_folder, model_name))
    net.load_state_dict(state['state_dict'])
    net.cuda()
    net.eval()
    print(f"Number of parameters in the model: {count_parameters(net):,}")

    for filename in flist_test:
        print(f"Processing: {filename}")
        ds_tst = PartDatasetTest(filename, args.rootdir, block_size=args.blocksize,
                                 npoints=args.npoints, test_step=args.test_step, features=features,
                                 labels=args.test_labels)
        tst_loader = torch.utils.data.DataLoader(ds_tst, batch_size=args.batchsize, shuffle=False,
                                                 num_workers=args.num_workers)

        xyz = ds_tst.xyzni[:, :3]
        scores = np.zeros((xyz.shape[0], nb_class))

        total_time = 0
        iter_nb = 0
        with torch.no_grad():
            t = tqdm(tst_loader, ncols=150)
            for pts, features, indices in t:
                t1 = time.time()
                features = features.cuda()
                pts = pts.cuda()
                outputs = net(features, pts)
                t2 = time.time()

                outputs_np = outputs.cpu().numpy().reshape((-1, nb_class))
                scores[indices.cpu().numpy().ravel()] += outputs_np

                iter_nb += 1
                total_time += (t2 - t1)
                t.set_postfix(time=f"{total_time / (iter_nb * args.batchsize):05e}")

        mask = np.logical_not(scores.sum(1) == 0)
        scores = scores[mask]
        pts_src = xyz[mask]

        # create the scores for all points
        scores = nearest_correspondance(pts_src, xyz, scores, k=1)

        # compute softmax
        scores = scores - scores.max(axis=1)[:, None]
        scores = np.exp(scores) / np.exp(scores).sum(1)[:, None]
        scores = np.nan_to_num(scores)
        scores = scores.argmax(1)

        # Compute confusion matrix
        if args.test_labels:
            # tst_logs = InformationLogger(model_folder, 'tst')
            log_tst = os.path.join(args.resdir, "{0}_preds_{1}_log.txt".format(filename, model_name[:-4]))
            lbl = ds_tst.labels[:, :]

            # Transfert des classes 1 à n vers classes ASPRS
            scores_eval = pred_to_asprs(scores, info_class["class_info"])

            cm = confusion_matrix(lbl.ravel(), scores_eval.ravel(),
                                  labels=list(map(int, list(info_class['class_info'].keys()))))

            print("")
            print("Confusion matrix - test:")
            print(cm)

            cl_acc = metrics.stats_accuracy_per_class(cm)
            cl_iou = metrics.stats_iou_per_class(cm)
            cl_fscore = metrics.stats_f1score_per_class(cm)

            print(f"Stats for test dataset:")
            print_metric('Test', 'Accuracy', cl_acc)
            print_metric('Test', 'iou', cl_iou)
            print_metric('Test', 'F1-Score', cl_fscore)

            if use_wandb:
                wandb.log({'test/acc': float(cl_acc[0]), 'test/fscore': cl_fscore[0], 'test/iou': cl_iou[0]})

            tst_avg_score = {'loss': -1, 'acc': cl_acc[0], 'iou': cl_iou[0], 'fscore': [0]}
            tst_class_score = {'acc': cl_acc[1], 'iou': cl_iou[1], 'fscore': cl_fscore[1]}
            # tst_logs.add_metric_values(tst_avg_score, -1)
            # tst_logs.add_class_scores(tst_class_score, -1)

            ligne_classes = "".rjust(15)
            for code1 in info_class["class_info"].keys():
                ligne_classes += "{0:>15}\t".format(info_class["class_info"][code1]["name"])
            with open(log_tst, "a") as log:
                log.write("Stats - test:\n\n\n")
                log.write("Confusion matrix:\n\n")
                log.write(ligne_classes + "\n")
                # log.write("\n".join(["\t".join([str(cell).rjust(15) for cell in row]) for row in cm]))
                h = 0
                for row in cm:
                    code_cla = list(info_class["class_info"].keys())[h]
                    cla = info_class["class_info"][code_cla]['name']
                    log.write(str(cla).rjust(15))
                    log.write("\t".join([str(cell).rjust(15) for cell in row]))
                    log.write("\n")
                    h += 1
                log.write("\n\n")

                log.write("Accuracy:\n\n")
                i = 0
                for code2 in info_class["class_info"].keys():
                    log.write("\t{0}: {1:.3f}\n".format(info_class["class_info"][code2]["name"], cl_acc[1][i]))
                    i += 1
                log.write("\n\tAverage: {:.3f}\n".format(cl_acc[0]))
                log.write("\n\n")

                log.write("IoU:\n\n")
                j = 0
                for code3 in info_class["class_info"].keys():
                    log.write("\t{0}: {1:.3f}\n".format(info_class["class_info"][code3]["name"], cl_iou[1][j]))
                    j += 1
                log.write("\n\tAverage: {:.3f}\n".format(cl_iou[0]))
                log.write("\n\n")

                log.write("F1-Score:\n\n")
                k = 0
                for code4 in info_class["class_info"].keys():
                    log.write("\t{0}: {1:.3f}\n".format(info_class["class_info"][code4]["name"], cl_fscore[1][k]))
                    k += 1
                log.write("\n\tAverage: {:.3f}\n".format(cl_fscore[0]))

            # write error file.
            # error2ply(model_folder / f"{filename}_error.ply", xyz=xyz, labels=lbl, prediction=scores,
            #           info_class=info_class['class_info'])

        if args.savepts:
            # Save predictions
            out_folder = os.path.join(args.resdir, 'tst')
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            lastest0 = [os.path.join(args.testdir, fich) for fich in os.listdir(args.testdir)
                        if fich.upper().endswith('.LAS')][0]
            with laspy.file.File(lastest0) as in_file:
                header = in_file.header
                xyz = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()
                out_file = os.path.join(args.resdir, "{0}_preds_{1}.las".format(filename, model_name[:-4]))
                write_to_las(out_file, xyz=xyz, pred=scores, header=header, info_class=info_class['class_info'])


def pred_to_asprs(pred, info_class):
    """Converts predicted values (0->n) to the corresponding ASPRS class."""
    labels2 = np.full(shape=pred.shape, fill_value=0, dtype=int)
    for key, value in info_class.items():
        labels2[pred == value['mode']] = int(key)
    return labels2


def write_to_las(filename, xyz, pred, header, info_class):
    """Write xyz and ASPRS predictions to las file format. """
    # TODO: Write CRS info with file.
    with laspy.file.File(filename, mode='w', header=header) as out_file:
        out_file.x = xyz[:, 0]
        out_file.y = xyz[:, 1]
        out_file.z = xyz[:, 2]
        pred = pred_to_asprs(pred, info_class)
        out_file.classification = pred


def main():
    '''
    Main function
    '''

    print(f"Host: {socket.gethostname()}")
    print(f"Python version: {python_version()}")
    print(f"Pytorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")

    # parse arguments
    args, config = parse_args()
    print(args)

    # use wandb if needed
    use_wandb = args.use_wandb

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # create the experiment name
    experiment_name = f"CP_{args.dsname}_{args.npoints}pts_" \
                      f"{args.blocksize}block_{args.drop}drop_" \
                      f"{args.loss}loss_" \
                      f"{args.optimizer}optim" \
                      f"{args.lr.replace('.', 'p')}lr_{args.nepochs}ep_" \
                      f"{args.iter}iter_batch{args.batchsize}" \
                      f"_{args.num_workers}work_{time_string}"

    # setup wandb
    if use_wandb:
        wandb_project_name = args.wandb_project_name
        wandb.login(key=os.environ.get("WANDB_API_KEY"))
        wandb.init(project=wandb_project_name,
                   job_type="train",
                   name=experiment_name,
                   config=config,
                   dir=args.savedir,
                   )
    print("Save directory: ", args.savedir)

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = Path(args.rootdir)
    dataset_dict = {'trn': [], 'val': [], 'tst': []}

    for dataset in dataset_dict.keys():
        for file in (base_dir / dataset).glob('*.hdfs'):
            dataset_dict[dataset].append(f"{dataset}/{file.stem}")

        if len(dataset_dict[dataset]) == 0:
            warnings.warn(f"{base_dir / dataset} is empty")

    print(f"Las files per dataset:\n Trn: {len(dataset_dict['trn'])} \n Val: {len(dataset_dict['val'])} \n "
          f"Tst: {len(dataset_dict['tst'])}")

    info_class = class_mode(args.mode)
    # Train + Validate model
    print("")
    print("TRAINING")

    # Train model
    model_folder, model_name = train(args, dataset_dict, info_class, time_string=time_string, use_wandb=use_wandb)
    # Test model
    if args.test:
        print("")
        print("TEST")
        test(args, dataset_dict['tst'], model_folder, model_name, info_class, use_wandb=use_wandb)


# Main
if __name__ == '__main__':
    main()