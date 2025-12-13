For this project, the goal is to select a practical and well-justified model architecture for semantic segmentation of outdoor LiDAR point clouds. I evaluated several families of point-based and voxel-based neural architectures, including PointNeXt, KPConv, and Cylinder3D.
After comparing performance, implementation complexity, computational cost, and suitability for a short time-boxed engineering task, I selected PointNeXt as the primary model.

Why PointNeXt? (Modernized PointNet++)

PointNeXt is a revised and significantly improved version of PointNet++, one of the most influential and widely used architectures for point cloud understanding.
It incorporates two major modernization strategies:

(1) Receptive field scaling

(2) Model scaling

The modernized strategies make model scalable to larger point cloud scenes, such as some outdoor scenes. The authors provide a clear outdoor benchmark comparison in SemanticKITTI, showing large improvements over PointNet++.

From the authors’ NeurIPS 2022 response:

Method	Test mIoU	Notes
PointNet++	20.1	Baseline
PointNeXt-S	48.4	4× faster throughput

It is also easy to implement. This makes PointNeXt a balanced choice for speed, accuracy, and engineering simplicity.

Alternatives Considered
(A) Cylinder3D

Cylinder3D is one of the strongest architectures for outdoor LiDAR segmentation.

Strengths:

Cylinder voxelization handles uneven LiDAR density very well

Excellent accuracy on outdoor lidar scene (state-of-the-art for large scenes)

Weaknesses:

Difficult to implement

Complex preprocessing pipeline

Not suitable for a lightweight 3-hour pipeline implementation.

(B) KPConv

Kernel Point Convolution is another strong outdoor model.

Strengths:

Continuous convolution kernels adapt to point density, handles varying densities in outdoor scene well

Good performance on outdoor datasets like SemanticKITTI

Weaknesses:

Harder to implement 

Requires their cpp_subsampling for data preparation

Again, great model but too complex for the scope of this assignment.

6. Final Choice: PointNeXt

I selected PointNeXt because:

Modern, high-quality architecture with fair outdoor performance (up to 48.4 mIoU on SemanticKITTI)

Fast and lightweight, no heavy voxelization or sparse convolutions

Simple training loop and preprocessing

MIT licensed → safe for commercial use

Provided reference implementations are easy to adapt

PointNeXt offers the best trade-off between speed, accuracy, and engineering simplicity.