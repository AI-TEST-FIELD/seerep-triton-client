import itertools
import open3d as o3d
import numpy as np


pcd_counter = 0
O3D_DEVICE = o3d.core.Device("CPU:0")


def load_pcd() -> o3d.t.geometry.PointCloud:
    """
    Loads a sample PointCloud, that includes a human from the AI-Test-Field.
    Returns the PointCloud in Numpy and Open3D data format.
    """

    global pcd_counter
    o3d_pcd = o3d.t.io.read_point_cloud("ouster_human-sample.pcd", format="pcd")

    return o3d_pcd


def pcd_o3d_to_numpy(
    o3d_pcd: o3d.t.geometry.PointCloud, feature_field: str = "reflectivity"
) -> np.ndarray:
    """
    Converts a open3D pointcloud to a Numpy array, adhering to triton client standards.

    Args:
        feature_field: str - one of [reflectivity, intensity]

    """

    num_points = o3d_pcd.point.positions.numpy().shape[0]
    np_pcd = np.zeros((num_points, 4), dtype=np.float32)
    tmp_positions = o3d_pcd.point.positions.numpy()
    np_pcd[:, 0] = tmp_positions[:, 0]
    np_pcd[:, 1] = tmp_positions[:, 1]
    np_pcd[:, 2] = tmp_positions[:, 2]
    np_pcd[:, 3] = o3d_pcd.point[feature_field].numpy()[:, 0]

    return np_pcd


def pcd_numpy_to_o3d(
    np_pcd: np.ndarray, feature_field: str = "reflectivity"
) -> o3d.t.geometry.PointCloud:
    global O3D_DEVICE

    out_pcd = o3d.t.geometry.PointCloud(device=O3D_DEVICE)
    out_pcd.point["positions"] = o3d.core.Tensor(
        np_pcd[:, 0:3], device=O3D_DEVICE, dtype=o3d.core.Dtype.Float32
    )

    out_pcd.point[feature_field] = o3d.core.Tensor(
        np_pcd[:, 3, None],
        device=O3D_DEVICE,
        dtype=o3d.core.Dtype.Float32,
    )

    return out_pcd

def pcd_ros_to_o3d(ros_pcd, feature_field: str) -> o3d.t.geometry.PointCloud:

    tmp_positions = np.zeros((ros_pcd["x"]["data"].shape[0], 3), dtype=np.float32)
    tmp_positions[:, 0] = ros_pcd["x"]["data"][:, 0]
    tmp_positions[:, 1] = ros_pcd["y"]["data"][:, 0]
    tmp_positions[:, 2] = ros_pcd["z"]["data"][:, 0]

    out_pcd = o3d.t.geometry.PointCloud(device=O3D_DEVICE)
    out_pcd.point["positions"] = o3d.core.Tensor(
        tmp_positions, device=O3D_DEVICE, dtype=o3d.core.Dtype.Float32
    )
    
    out_pcd.point[feature_field] = o3d.core.Tensor(
        ros_pcd[feature_field]["data"],
        device=O3D_DEVICE,
        dtype=o3d.core.Dtype.Float32,
    )

    return out_pcd



def transform_pcd_to_kitti_frame(
    o3d_pcd: o3d.t.geometry.PointCloud,
) -> o3d.t.geometry.PointCloud:
    kitti_translation = o3d.core.Tensor([0.0, 0.0, -1.026558971], device=O3D_DEVICE)
    
    return o3d_pcd.translate(kitti_translation)


def transform_pcd_to_base_frame(
    o3d_pcd: o3d.t.geometry.PointCloud,
) -> o3d.t.geometry.PointCloud:
    "Ouster specific"
    base_transform = o3d.core.Tensor(
        [
            [0.82638931, -0.02497454, 0.56254509, 0.191287],
            [0.01212522, 0.99957356, 0.02656451, -0.35169424],
            [-0.56296864, -0.01513165, 0.82633973, 1.90064396],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=O3D_DEVICE,
    )
    return o3d_pcd.transform(base_transform)


def crop_pcd(
    o3d_pcd: o3d.t.geometry.PointCloud, xyz_limits: list
) -> o3d.t.geometry.PointCloud:
    bounding_box_coordinates = list(itertools.product(*xyz_limits))
    bounding_box = o3d.t.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.core.Tensor(bounding_box_coordinates, device=O3D_DEVICE)
    )

    return o3d_pcd.crop(bounding_box)


def do_test_preprocess() -> np.ndarray:
    global pcd_counter
    global O3D_DEVICE

    raw_o3d_pcd = load_pcd()
    preprocessed_o3d_pcd = transform_pcd_to_base_frame(raw_o3d_pcd.clone())
    preprocessed_o3d_pcd = transform_pcd_to_kitti_frame(preprocessed_o3d_pcd)
    # preprocessed_o3d_pcd = crop_pcd(preprocessed_o3d_pcd, [[0.4, 14.], [-15., 15.], [-2, 0.5]])

    if pcd_counter == 0:
        # do preprocessing as performed by inference pipeline (no transform, np.max normalization and intensity field)
        print('Preprocessing Config: no transform, np.max normalization, and intensity field')
        preprocessed_np_pcd = pcd_o3d_to_numpy(
            o3d_pcd=raw_o3d_pcd, feature_field="intensity"
        )
        preprocessed_np_pcd[:, 3] = preprocessed_np_pcd[:, 3] / np.max(
            preprocessed_np_pcd[:, 3]
        )

    elif pcd_counter == 1:
        # do preprocessing with transformed pcd (kitti frame), 255 normalization, and intensity field
        print('Preprocessing Config: transformed pcd (kitti frame), 255 normalization, and intensity field')
        preprocessed_np_pcd = pcd_o3d_to_numpy(
            o3d_pcd=preprocessed_o3d_pcd, feature_field="intensity"
        )
        preprocessed_np_pcd[:, 3] = preprocessed_np_pcd[:, 3] / 255

    elif pcd_counter == 2:
        # do preprocessing with transformed pcd (kitti frame), 255 normalization, and reflectivity field
        print('Preprocessing Config: transformed pcd (kitti frame), 255 normalization, and reflectivity field')
        preprocessed_np_pcd = pcd_o3d_to_numpy(
            o3d_pcd=preprocessed_o3d_pcd, feature_field="reflectivity"
        )
        preprocessed_np_pcd[:, 3] = preprocessed_np_pcd[:, 3] / 255

    else:
        exit()

    pcd_counter += 1

    return preprocessed_np_pcd


def do_triton_preprocess(client_pcd: np.ndarray) -> np.ndarray:
    triton_o3d_pcd = pcd_numpy_to_o3d(client_pcd)
    preprocessed_o3d_pcd = transform_pcd_to_base_frame(triton_o3d_pcd)
    preprocessed_o3d_pcd = transform_pcd_to_kitti_frame(preprocessed_o3d_pcd)

    preprocessed_np_pcd = pcd_o3d_to_numpy(
        o3d_pcd=preprocessed_o3d_pcd, feature_field="reflectivity"
    )

    preprocessed_np_pcd[:, 3] = preprocessed_np_pcd[:, 3] / 255

    return preprocessed_np_pcd


def janosch_pcd_magic(client_pcd: np.ndarray, triton_inference: bool):
    if triton_inference:
        np_pcd = do_triton_preprocess(client_pcd=client_pcd)
    else:
        np_pcd = do_test_preprocess()

    return np_pcd
