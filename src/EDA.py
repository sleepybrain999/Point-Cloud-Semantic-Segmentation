
import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from pandaset.geometry import lidar_points_to_ego

def run_eda(data, labels,class_map):
    """
    Show and plot EDA analysis including number of scene, frame, points per frame, and class distribution
    
    """

    num_scenes = len(data)                          #Number of scene
    num_frame = sum(len(scene) for scene in data)   #Sum of frames in all the scenes
    points_per_frame  = [len(df)                    #Number of points per frame
                    for scene in data
                    for df in scene]

    all_labels = np.hstack([lbl for scene in labels for lbl in scene])    # Flatten the labels
    unique_classes, counts = np.unique(all_labels, return_counts=True)    # Unique class and their counts
    classes_per_frame = [len(np.unique(lbl))                              # Number of class per frame
                        for scene in labels
                        for lbl in scene
                        ]

    # Sort indices by count (descending)
    sorted_idx = np.argsort(counts)[::-1]
    sorted_classes = unique_classes[sorted_idx]
    sorted_counts = counts[sorted_idx]

    names = []
    cls_ids = []
    # Apply sorted order
    for cls_id, count in zip(sorted_classes[:5], sorted_counts[:5]):
      name = class_map.get(str(cls_id), "Unknown")
      names.append(name)
      cls_ids.append(cls_id)

    sorted_classes = unique_classes[sorted_idx]
    sorted_counts = counts[sorted_idx]
    # -------------------------------
    # Create figure
    # -------------------------------
    fig = plt.figure(figsize=(12, 10))

    # Text summary on top
    ax_text = fig.add_axes([0.05, 0.55, 0.9, 0.4])  # x, y, width, height
    ax_text.axis("off")
    summary_text = f"""
===== PANDASET EDA SUMMARY =====

Number of scenes: {num_scenes}
Number of frames: {num_frame}

POINTS PER FRAME:
  min  = {np.min(points_per_frame)}
  max  = {np.max(points_per_frame)}
  mean = {np.mean(points_per_frame):.2f}
  std  = {np.std(points_per_frame):.2f}

SEMANTIC CLASS INFORMATION:
  Total unique classes = {len(unique_classes)}
  Classes present      = {unique_classes}

CLASSES PER FRAME:
  min  = {np.min(classes_per_frame)}
  max  = {np.max(classes_per_frame)}
  mean = {np.mean(classes_per_frame):.2f}

Top 5 most common classes: 
  Class {cls_ids[0]} ({names[0]})  Class {cls_ids[1]} ({names[1]})  Class {cls_ids[2]} ({names[2]})
  Class {cls_ids[3]} ({names[3]})  Class {cls_ids[4]} ({names[4]})

"""
    ax_text.text(0, 1, summary_text, va="top", fontsize=10, family="monospace")

    # Histogram for class distribution
    ax_hist = fig.add_axes([0.05, 0.05, 0.9, 0.45])
    ax_hist.hist(all_labels, bins=np.arange(unique_classes.max() + 2) - 0.5, rwidth=0.8)
    ax_hist.set_title("Semantic Class Distribution")
    ax_hist.set_xlabel("Class ID")
    ax_hist.set_ylabel("Point Count")

    plt.show()  # Display immediately

    print("EDA complete!")


def show_pcd_example(df,pose, nb=20, std=3, voxel_size=0.05, radius=0.3, max_nn=30):
    """
    Show different stages of a point cloud using Open3D.
    - Raw
    - Filtered vs Outliers
    - Downsampled
    - Normals
    
    Parameters:
        pcd         : Open3D point cloud
        nb, std     : statistical outlier removal parameters
        voxel_size  : voxel downsample size
        radius, max_nn : normal estimation parameters
    """
    # 1. Raw point cloud

    points_world = df[["x", "y", "z"]].values
    points_ego = lidar_points_to_ego(points_world, pose)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_ego)

    o3d.visualization.draw_geometries([pcd], window_name="Raw Point Cloud")

    # 2. Filtered vs Outliers
    filtered, idx = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    outliers = pcd.select_by_index(idx, invert=True)
    outliers.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([filtered, outliers], window_name="Filtered vs Outliers")

    # 3. Downsampled
    down_pcd = filtered.voxel_down_sample(voxel_size=voxel_size)
    o3d.visualization.draw_geometries([down_pcd], window_name="Downsampled Point Cloud")


    # 4. Normals
    down_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

    o3d.visualization.draw_geometries([down_pcd], window_name="Point Cloud Normals")

