import open3d as o3d
import numpy as np
import pandas as pd
from pandaset import DataSet

def load_dataset(dataset_root="./data/pandaset", scene_range=(1, 48)):
    """
    Load PandaSet LiDAR + semantic labels using the official devkit.
    Returns lists of dataframes and label arrays aligned per-frame.
    """

    dataset = DataSet(dataset_root)
    
    # sequences with semantic labels
    scenes = dataset.sequences(with_semseg=True)
    
    lidar_data = []
    labels = []
    poses = []

    for scene in scenes:
        
        seq = dataset[scene]
        # Load lidar data and semseg labels
        seq.load_lidar().load_semseg()
        print(f"Loading scene {scene}. Number of frames:", len(seq.lidar._data))

        # List of LiDAR dataframe
        pc_df = seq.lidar[:]
        # Semantic labels (class IDs)
        semseg_df = seq.semseg[:]
        # lidar pose
        pose = seq.lidar.poses[:]

        # Append to lists
        lidar_data.append(pc_df)
        labels.append(semseg_df)  
        poses.append(pose)

    print(f"Loaded {len(lidar_data)} LiDAR sweeps with semantic labels.")
    return lidar_data, labels, poses

