import os
import laspy
import numpy as np
from scipy.spatial import KDTree


def is_within_tile_boundary(point, tile_boundary, buffer):
    """
    Check if a point is within the tile boundary, considering a buffer.

    Args:
        point (tuple or array): The (x, y, z) coordinates of the point.
        tile_boundary (tuple): (min_x, max_x, min_y, max_y) of the tile.
        buffer (float): Buffer distance to shrink the boundary.

    Returns:
        bool: True if the point is within the buffered boundary, False otherwise.
    """
    x, y, z = point
    return (tile_boundary[0] + buffer <= x <= tile_boundary[1] - buffer) and (
        tile_boundary[2] + buffer <= y <= tile_boundary[3] - buffer
    )


def find_whole_trees(tile_points, tile_pred_instance, tile_boundary, buffer=0.2):
    """
    Identify tree instance IDs whose points are entirely within the tile boundary (with buffer).

    Args:
        tile_points (np.ndarray): Nx3 array of tile point coordinates.
        tile_pred_instance (np.ndarray): Array of instance IDs for each point.
        tile_boundary (tuple): (min_x, max_x, min_y, max_y) of the tile.
        buffer (float): Buffer distance to shrink the boundary.

    Returns:
        set: Set of instance IDs that are fully contained within the buffered tile boundary.
    """
    whole_tree_ids = set()
    unique_ids = np.unique(tile_pred_instance)

    # Get counts for each ID
    unique_with_counts, counts = np.unique(tile_pred_instance, return_counts=True)
    id_counts = dict(zip(unique_with_counts, counts))

    print(f"Unique IDs in tile: {sorted(unique_ids)}")
    print("Cluster sizes:")
    for tree_id in sorted(unique_ids):
        if tree_id > 0:  # Only show actual tree IDs
            print(f"Tree ID {tree_id}: {id_counts[tree_id]} points")

    for tree_id in unique_ids:
        # Skip background (0) and unassigned (-1 or any negative)
        if tree_id <= 0:
            continue
        tree_points = tile_points[tile_pred_instance == tree_id]
        if all(
            is_within_tile_boundary(point, tile_boundary, buffer)
            for point in tree_points
        ):
            whole_tree_ids.add(tree_id)
    return whole_tree_ids


def reassign_small_clusters(
    merged_pred_instance,
    original_points,
    min_cluster_size=300,
    initial_radius=1.0,
    max_radius=5.0,
    radius_step=1.0,
):
    """
    Reassign points in small clusters to the nearest larger cluster, or mark as -1 if not possible.

    Args:
        merged_pred_instance (np.ndarray): Array of instance IDs for all points.
        original_points (np.ndarray): Nx3 array of point coordinates.
        min_cluster_size (int): Minimum size for a cluster to be considered valid.
        initial_radius (float): Starting search radius for reassignment.
        max_radius (float): Maximum search radius for reassignment.
        radius_step (float): Step size to increase the search radius.
    """
    unique, counts = np.unique(merged_pred_instance, return_counts=True)
    small_instances = unique[counts < min_cluster_size]
    print(
        f"Found {len(small_instances)} small_instance clusters (less than {min_cluster_size} points) out of {len(unique)} unique clusters."
    )

    kdtree = KDTree(original_points)

    reassigned_count = 0
    not_reassigned_count = 0
    for small_instance in small_instances:
        if small_instance == -1:
            continue
        indices = np.where(merged_pred_instance == small_instance)[0]
        for idx in indices:
            point = original_points[idx]
            radius = initial_radius
            while radius <= max_radius:
                nearest_indices = kdtree.query_ball_point(point, radius)
                for nearest_idx in nearest_indices:
                    nearest_instance = merged_pred_instance[nearest_idx]
                    if nearest_instance != small_instance and nearest_instance != -1:
                        merged_pred_instance[idx] = nearest_instance
                        reassigned_count += 1
                        break
                else:
                    radius += radius_step
                    continue
                break
            else:
                merged_pred_instance[idx] = -1
                not_reassigned_count += 1
    print(f"Reassigned {reassigned_count} points to nearest other tree instance.")
    print(
        f"{not_reassigned_count} points could not be reassigned and have a value of -1."
    )


def merge_tiles(
    tile_folder,
    original_point_cloud,
    output_file,
    buffer,
    min_cluster_size,
    initial_radius,
    max_radius,
    radius_step,
):
    """
    Merge predicted instance and semantic labels from tile files back into the original point cloud.

    For each tile, reindex instance IDs globally, assign whole-tree instances to the original cloud,
    reassign small clusters, and save the merged result. Also generates a 2D top-view image.

    Args:
        tile_folder (str): Path to the folder containing tile LAS/LAZ files.
        original_point_cloud (str): Path to the original point cloud file.
        output_file (str): Path to save the merged point cloud.
        buffer (float): Buffer distance for whole-tree assignment.
        min_cluster_size (int): Minimum cluster size for reassignment.
        initial_radius (float): Initial search radius for point reassignment.
        max_radius (float): Maximum search radius for point reassignment.
        radius_step (float): Radius increment step for point reassignment.
    """
    print("Loading the original point cloud...")
    original_las = laspy.read(original_point_cloud)
    original_points = np.vstack(
        (np.array(original_las.x), np.array(original_las.y), np.array(original_las.z))
    ).T

    merged_pred_instance = np.full(len(original_points), -1, dtype=int)
    merged_pred_semantic = np.full(len(original_points), -1, dtype=int)

    global_instance_counter = 0  # Start from 0

    print("Processing tiles...")
    tile_files = [
        f for f in os.listdir(tile_folder) if f.endswith(".las") or f.endswith(".laz")
    ]

    print(
        f"Found {len(tile_files)} tile(s) - buffer logic will be applied to all tiles"
    )

    tile_data = []
    for filename in os.listdir(tile_folder):
        if filename.endswith(".las") or filename.endswith(".laz"):
            print(f"Loading tile: {filename}")
            tile_las = laspy.read(os.path.join(tile_folder, filename))
            tile_points = np.vstack(
                (np.array(tile_las.x), np.array(tile_las.y), np.array(tile_las.z))
            ).T

            tile_pred_instance = np.array(tile_las.PredInstance)
            tile_pred_semantic = np.array(tile_las.PredSemantic)

            # Reindex PredInstance IDs globally, but keep 0 as 0 as it reflects ground points
            unique_ids = np.unique(tile_pred_instance)
            id_map = {
                old_id: (global_instance_counter + i if old_id != 0 else 0)
                for i, old_id in enumerate(unique_ids)
            }
            global_instance_counter += len([uid for uid in unique_ids if uid != 0])

            reindexed_pred_instance = np.array(
                [id_map[pid] for pid in tile_pred_instance]
            )

            # Store tile data for later processing
            tile_boundary = (
                np.min(tile_points[:, 0]),
                np.max(tile_points[:, 0]),
                np.min(tile_points[:, 1]),
                np.max(tile_points[:, 1]),
            )
            tile_data.append(
                (
                    tile_points,
                    reindexed_pred_instance,
                    tile_pred_semantic,
                    filename,
                    tile_boundary,
                )
            )

    # Assign PredInstance for Whole Trees (buffer logic always applied)
    print("Assigning PredInstance for whole trees...")
    for (
        tile_points,
        reindexed_pred_instance,
        tile_pred_semantic,
        filename,
        tile_boundary,
    ) in tile_data:
        kdtree = KDTree(original_points)
        distances, indices = kdtree.query(tile_points)

        print(f"Processing tile: {filename} — applying buffer logic.")

        # Identify whole trees within the tile using buffer (always applied)
        whole_tree_ids = find_whole_trees(
            tile_points, reindexed_pred_instance, tile_boundary, buffer
        )

        print(f"Found {len(whole_tree_ids)} whole trees in tile {filename}")
        print(f"Whole tree IDs: {sorted(whole_tree_ids)}")

        for i, idx in enumerate(indices):
            if reindexed_pred_instance[i] in whole_tree_ids:
                merged_pred_instance[idx] = reindexed_pred_instance[i]
                merged_pred_semantic[idx] = tile_pred_semantic[i]

    # Rest of the function remains the same...
    print("Reassigning small PredInstance clusters...")
    reassign_small_clusters(
        merged_pred_instance,
        original_points,
        min_cluster_size,
        initial_radius,
        max_radius,
        radius_step,
    )

    # Rest of the function remains the same...
    print("Saving merged point cloud...")
    merged_las = laspy.create(
        point_format=original_las.header.point_format,
        file_version=str(original_las.header.version),
    )
    merged_las.points = original_las.points
    merged_las.header = (
        original_las.header
    )  # Preserve the header information including the coordinate system

    # Initialize new extra attributes
    merged_las.add_extra_dim(
        laspy.ExtraBytesParams(name="PredInstance", type=np.dtype(np.int32))
    )
    # merged_las.add_extra_dim(
    #     laspy.ExtraBytesParams(name="PredSemantic", type=np.dtype(np.int32))
    # )

    # Assign the merged attributes
    merged_las.PredInstance = merged_pred_instance
    # merged_las.PredSemantic = merged_pred_semantic

    # Copy all extra attributes except PredInstance and PredSemantic
    for dimension in original_las.point_format.dimensions:
        if dimension.name not in ["PredInstance", "PredSemantic"]:
            setattr(merged_las, dimension.name, getattr(original_las, dimension.name))

    merged_las.write(
        output_file, do_compress=True, laz_backend=laspy.LazBackend.LazrsParallel
    )
    print(f"Merged point cloud saved to {output_file}")


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge PredInstance and (PredSemantic) from tiles back into the original point cloud."
    )
    parser.add_argument("--tile_folder", help="Folder containing LAS/LAZ tiles.")
    parser.add_argument("--original_point_cloud", help="subsampled point cloud file.")
    parser.add_argument("--output_file", help="Output file for the merged point cloud.")
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.2,
        help="Buffer distance to consider for whole trees.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=300,
        help="Minimum cluster size for PredInstance reassignment.",
    )
    parser.add_argument(
        "--initial_radius",
        type=float,
        default=1.0,
        help="Initial search radius for point reassignment.",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=5.0,
        help="Maximum search radius for point reassignment.",
    )
    parser.add_argument(
        "--radius_step",
        type=float,
        default=1.0,
        help="Radius increment step for point reassignment.",
    )
    args = parser.parse_args()

    merge_tiles(
        args.tile_folder,
        args.original_point_cloud,
        args.output_file,
        args.buffer,
        args.min_cluster_size,
        args.initial_radius,
        args.max_radius,
        args.radius_step,
    )
