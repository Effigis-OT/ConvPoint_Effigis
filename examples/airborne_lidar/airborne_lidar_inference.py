# add the parent folder to the python path to access convpoint library
import os
import warnings
# sys.path.append('/home/ubuntu/anaconda2/envs/pytorch_convpoint_p36/ConvPoint')
# sys.path.append('C:\\Users\\pbug\\.conda\\envs\\pytorch_convpoint_p36\\ConvPoint')

import argparse
import h5py
import laspy
import numpy as np
import time
import torch
import torch.utils.data
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

from airborne_lidar_seg_wandb import get_model, nearest_correspondance, count_parameters, class_mode
from airborne_lidar_utils import write_features


def parse_args():
    parser = argparse.ArgumentParser()
    # '--modeldir' for backward compatibility
    parser.add_argument("--model_pth", "--modeldir", default=None, type=str)
    parser.add_argument("--rootdir", default=None, type=str,
                        help="Directory containing input test las files.")
    parser.add_argument("--outdir", default=None, type=str,
                        help="Directory where to output inference results (default: <rootdir>/out).")
    parser.add_argument("--test_step", default=5, type=float)
    parser.add_argument("--batchsize", "-b", default=32, type=int)
    parser.add_argument("--npoints", default=8168, type=int, help="Number of points to be sampled in the block.")
    parser.add_argument("--blocksize", default=25, type=int,
                        help="Size in meters of the infinite vertical column, to be processed.")

    parser.add_argument("--num_workers", default=3, type=int)
    parser.add_argument("--model", default="SegBig", type=str,
                        help="SegBig is the only available model at this time, for this dataset.")
    parser.add_argument("--features", default="xyz", type=str,
                        help="Features to process. xyzni means xyz + number of returns + intensity. Default is xyz."
                             "Currently, only xyz and xyzni are supported for this dataset.")
    parser.add_argument("--mode", default=5, type=int,
                        help="Class mode."
                             "1: building, water, ground."
                             "2: 5 classes: building, water, ground, low vegetation and medium + high vegetation"
                             "3: 6 classes: building, water, ground, low vegetation, medium and high vegetation"
                             "4: DALES."
                             "5: 3 classes: ground, vegetation, building"
                             "6: 4 classes: ground, vegetation, building, pole"
                             "7: 5 classes: ground, vegetation, building, power line, pole")
    args = parser.parse_args()
    print(args)
    return args


def read_las_format(in_file):
    """Extract data from a .las file.
    Will normalize XYZ and intensity between 0 and 1.
    """

    n_points = len(in_file)
    x = np.reshape(in_file.x, (n_points, 1))
    y = np.reshape(in_file.y, (n_points, 1))
    z = np.reshape(in_file.z, (n_points, 1))
    intensity = np.reshape(in_file.intensity, (n_points, 1))
    nb_return = np.reshape(in_file.num_returns, (n_points, 1))

    # Converting data to relative xyz reference system.
    min_x = np.min(x)
    min_y = np.min(y)
    min_z = np.min(z)
    norm_x = x - min_x
    norm_y = y - min_y
    norm_z = z - min_z
    # Intensity is normalized based on min max values.
    norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    xyzni = np.hstack((norm_x, norm_y, norm_z, nb_return, norm_intensity)).astype(np.float16)

    return xyzni


def write_las_to_h5(filename):
    with laspy.file.File(filename) as in_file:
        xyzni = read_las_format(in_file)

        filename = f"{filename.parent / filename.name.split('.')[0]}_prepared.hdfs"
        write_features(filename, xyzni=xyzni)
        return filename


def write_to_las(filename, xyz, pred, header, info_class):
    """Write xyz and ASPRS predictions to las file format. """
    # TODO: Write CRS info with file.
    with laspy.file.File(filename, mode='w', header=header) as out_file:
        out_file.x = xyz[:, 0]
        out_file.y = xyz[:, 1]
        out_file.z = xyz[:, 2]
        pred = pred_to_asprs(pred, info_class)
        out_file.classification = pred


def pred_to_asprs(pred, info_class):
    """Converts predicted values (0->n) to the corresponding ASPRS class."""
    labels2 = np.full(shape=pred.shape, fill_value=0, dtype=int)
    for key, value in info_class.items():
        labels2[pred == value['mode']] = int(key)
    return labels2


# Part dataset only for testing
class PartDatasetTest(Dataset):

    def compute_mask(self, pt, bs):
        # build the mask
        mask_x = np.logical_and(self.xyzni[:, 0] < pt[0] + bs / 2, self.xyzni[:, 0] > pt[0] - bs / 2)
        mask_y = np.logical_and(self.xyzni[:, 1] < pt[1] + bs / 2, self.xyzni[:, 1] > pt[1] - bs / 2)
        mask = np.logical_and(mask_x, mask_y)
        return mask

    def __init__(self, in_file, block_size=25, npoints=8192, test_step=0.8, features=False):

        self.filename = in_file
        self.bs = block_size
        self.npoints = npoints
        self.features = features
        self.step = test_step
        self.xyzni = None
        # load the points
        if self.xyzni is None:
            # load the points
            with h5py.File(self.filename, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]

        discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
        self.pts = np.unique(discretized, axis=0)
        self.pts = self.pts.astype(np.float) * self.step

    def __getitem__(self, index):
        if self.xyzni is None:
            # load the points
            with h5py.File(self.filename, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
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
        if self.xyzni is None:
            # load the points
            with h5py.File(self.filename, 'r') as data_file:
                self.xyzni = data_file["xyzni"][:]
            discretized = ((self.xyzni[:, :2]).astype(float) / self.step).astype(int)
            self.pts = np.unique(discretized, axis=0)
            self.pts = self.pts.astype(np.float) * self.step
        return len(self.pts)


def load_model_eval(model_path, nb_class, args):
    # create the network
    print("Creating network...")
    if torch.cuda.is_available():
        state = torch.load(model_path)
    else:
        state = torch.load(model_path, map_location=torch.device('cpu'))
    arg_dict = args.__dict__
    config_dict = state['args'].__dict__
    for key, value in config_dict.items():
        if key not in ['rootdir', 'num_workers', 'batchsize']:
            arg_dict[key] = value
    net, features = get_model(nb_class, args)
    net.load_state_dict(state['state_dict'])
    if torch.cuda.is_available():
        net.cuda()
    else:
        net.cpu()
    net.eval()
    return net, features


def test(filename, model, model_features, info_class, args):
    nb_class = info_class['nb_class']
    print(f"Number of parameters in the model: {count_parameters(model):,}")
    las_filename = filename
    # for filename in flist_test:
    print(filename)
    filename0 = Path(args.rootdir) / f"{filename}.las"
    filename = write_las_to_h5(filename0)
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path(args.rootdir) / 'out'
    outdir.mkdir(exist_ok=True)

    ds_tst = PartDatasetTest(filename, block_size=args.blocksize, npoints=args.npoints,
                             test_step=args.test_step, features=model_features)
    tst_loader = torch.utils.data.DataLoader(ds_tst, batch_size=args.batchsize,
                                             shuffle=False, num_workers=args.num_workers)

    xyz = ds_tst.xyzni[:, :3]
    scores = np.zeros((xyz.shape[0], nb_class))

    total_time = 0
    iter_nb = 0
    with torch.no_grad():
        t = tqdm(tst_loader, ncols=150)
        for pts, features, indices in t:
            t1 = time.time()
            if torch.cuda.is_available():
                features = features.cuda()
                pts = pts.cuda()
            outputs = model(features, pts)
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

    # Save predictions

    with laspy.file.File(filename0) as in_file:
        header = in_file.header
        xyz = np.vstack((in_file.x, in_file.y, in_file.z)).transpose()
        str_modele = os.path.basename(args.model_pth)[:-24]
        write_to_las(outdir / f"{las_filename}_classif_{str_modele}.las",
                     xyz=xyz, pred=scores, header=header, info_class=info_class['class_info'])


def main():
    args = parse_args()

    # create the file lists (trn / val / tst)
    print("Create file list...")
    base_dir = Path(args.rootdir)
    dataset_dict = []

    for file in base_dir.glob('*.las'):
        dataset_dict.append(file.stem)

    if len(dataset_dict) == 0:
        warnings.warn(f"{base_dir} is empty")

    print(f"Las files in tst dataset: {len(dataset_dict)}")

    info_class = class_mode(args.mode)
    model, feats = load_model_eval(Path(args.model_pth), info_class['nb_class'], args)
    num = 0
    nb = len(dataset_dict)
    for filename in dataset_dict:
        num += 1
        #filename2 = Path(args.rootdir) / f"{filename}.las"
        #nuage = laspy.file.File(filename2, mode="r")
        #nbpts = len(nuage)
        #nuage.close()
        #if nbpts > 5000:
        print("")
        print("Classification du fichier {0}/{1}...".format(num, nb))
        test(filename, model, feats, info_class, args)
        #else:
        #    print("")
        #    print("Omission du fichier {0}/{1} (seulement {2} points)".format(num, nb, nbpts))


if __name__ == '__main__':
    main()
