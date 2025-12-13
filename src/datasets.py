from torch.utils.data import Dataset
import numpy as np
from pandaset.geometry import lidar_points_to_ego
from src.preprocess import voxel_downsample, random_sample,augment
class PointNeXtDataset(Dataset):
    """
    PandaSet → PointNeXt preprocessing.
    
    Includes:
      - Ego-frame transform
      - Voxel downsample
      - Random sampling
      - Random rotation / scaling / jitter / flip
      - Normalization
    """

    def __init__(
        self,
        lidar_scenes,         # list[list[df]]
        label_scenes,         # list[list[np.array]]
        poses,                # list[list[pose_dict]]
        num_points=16384,
        voxel_size=0.05,
        augment=True
    ):
        self.frames = []

        # Flatten scenes → frames
        for scene, lbl_scene, pose_scene in zip(lidar_scenes, label_scenes, poses):
            for df, labels, pose in zip(scene, lbl_scene, pose_scene):

                xyz = df[["x", "y", "z"]].values
                intensity = df["i"].values.reshape(-1, 1)

                # Ego-frame transform 
                xyz_ego = lidar_points_to_ego(xyz, pose)

                pts = np.hstack([xyz_ego, intensity])
                self.frames.append((pts.astype(np.float32),
                                    labels.astype(np.int64)))

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        pts, labels = self.frames[idx]

        # Voxel downsample
        pts, labels = voxel_downsample(pts, labels,voxel_size=self.voxel_size)

        # Random sampling 
        pts, labels = random_sample(pts, labels, self.num_points)

        #  Optional augmentation 

        if self.augment:
            pts = augment(pts)
        #  Normalize (center cloud) 
        pts[:, :3] -= pts[:, :3].mean(axis=0)

        return pts, labels
