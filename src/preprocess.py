
import numpy as np
from collections import Counter

def filter_data(lidar_data, segmantic_labels, lidar_poses):
    """
    Check for any empty frame (lidar,labels), and check for data and label mismatch
    """
    data = []
    labels = []
    poses = []

    for scene_idx, (scene_lidar, scene_label, scene_pose) in enumerate(zip(lidar_data, segmantic_labels, lidar_poses)):

        frames_by_scene = []
        labels_by_scene = []
        poses_by_scene = []

        for frame_idx, (frame_df, label_df, pose) in enumerate(zip(scene_lidar, scene_label, scene_pose)):

            # Check if LiDAR frame is empty
            if frame_df is None or len(frame_df) == 0:
                print(f"[SKIP] Scene {scene_idx}, Frame {frame_idx}: empty LiDAR frame.")
                continue

            # Extract labels for this frame
            labels_arr = label_df["class"].to_numpy()

            # Check if label array is empty
            if labels_arr is None or len(labels_arr) == 0:
                print(f"[SKIP] Scene {scene_idx}, Frame {frame_idx}: empty label array.")
                continue

            # Check LiDAR–label mismatch
            if len(frame_df) != len(labels_arr):
                print(
                    f"[SKIP] Scene {scene_idx}, Frame {frame_idx}: "
                    f"point mismatch → LiDAR={len(frame_df)}, labels={len(labels_arr)}"
                )
                continue

            # If all checks pass, add to dataset
            frames_by_scene.append(frame_df)
            labels_by_scene.append(labels_arr)
            poses_by_scene.append(pose)

        data.append(frames_by_scene)
        labels.append(labels_by_scene)
        poses.append(poses_by_scene)

    print(f"Removed invalid frames (empty or mismatched).")
    print(f"Total valid frames: {sum(len(scene) for scene in data)}")

    return data, labels, poses

def voxel_downsample(pts, labels, voxel_size):
    """
    pts: (N, 4) array [x,y,z,intensity]
    labels: (N,) semantic labels
    voxel_size: float

    returns:
        pts_ds: (M, 4) downsampled pts
        labels_ds: (M,) majority-vote labels
    """

    xyz = pts[:, :3]
    intensity = pts[:, 3]

    # Compute voxel indices
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int32)
    
    # Dictionary: voxel_hash -> list of point indices
    vox = {}
    for i, v in enumerate(voxel_indices):
        key = tuple(v)
        if key not in vox:
            vox[key] = []
        vox[key].append(i)

    down_xyz = []
    down_intensity = []
    down_labels = []

    # Iterate over voxels
    for key, idx_list in vox.items():
        pts_voxel = xyz[idx_list]
        intens_voxel = intensity[idx_list]
        labels_voxel = labels[idx_list]

        # Centroid of voxel = downsampled XYZ
        centroid = pts_voxel.mean(axis=0)

        # Mean intensity
        mean_int = intens_voxel.mean()

        # Majority vote for labels
        label = Counter(labels_voxel).most_common(1)[0][0]

        down_xyz.append(centroid)
        down_intensity.append(mean_int)
        down_labels.append(label)

    pts_ds = np.hstack([
        np.array(down_xyz, dtype=np.float32),
        np.array(down_intensity, dtype=np.float32).reshape(-1, 1)
    ])

    labels_ds = np.array(down_labels, dtype=np.int64)

    return pts_ds, labels_ds

#  random sampling if N >= number of points set for model input
def random_sample(pts, labels, num_points):
    N = pts.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        pad = np.random.choice(N, num_points - N, replace=True)
        idx = np.concatenate([np.arange(N), pad])
    return pts[idx], labels[idx]

def augment(pts):
    # Copy xyz
    xyz = pts[:, :3].copy()

    # 1. Random rotation (around Z axis)
    theta = np.random.uniform(-np.pi, np.pi)
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0,              0,             1]])
    xyz = xyz @ R.T

    # 2. Random scaling
    scale = np.random.uniform(0.9, 1.1)
    xyz *= scale

    # 3. Random flip
    flip_type = np.random.randint(4)
    if flip_type == 1:
        xyz[:, 0] = -xyz[:, 0]   # flip X
    elif flip_type == 2:
        xyz[:, 1] = -xyz[:, 1]   # flip Y
    elif flip_type == 3:
        xyz[:, :2] = -xyz[:, :2] # flip XY (180° rot)

    # 4. Jittering (Gaussian noise)
    noise = np.random.normal(0, 0.01, xyz.shape)
    xyz += noise

    # Assign back
    pts[:, :3] = xyz
    
    return pts


