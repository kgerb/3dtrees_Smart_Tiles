#!/usr/bin/env python
"""
Merge LAZ tiles with species predictions from detailview_predictions folder.

This script merges overlapping tiles, handles instance renumbering, resolves
species ID conflicts using higher probability, and generates per-tile outputs.
"""

import os
import argparse
import pandas as pd
import numpy as np
import laspy
from scipy.spatial import KDTree
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from functools import partial


@dataclass
class TreeSpeciesInfo:
    """Species information for a tree instance."""
    species_id: int
    species_prob: float
    species_name: str
    tree_height: float


def find_whole_trees(
    tile_points: np.ndarray, 
    tile_pred_instance: np.ndarray, 
    tile_boundary: Tuple[float, float, float, float], 
    buffer: float = 0.2
) -> Set[int]:
    """
    Identify tree instance IDs whose points are entirely within the tile boundary (with buffer).
    
    Uses vectorized NumPy operations for speed.

    Args:
        tile_points: Nx3 array of tile point coordinates.
        tile_pred_instance: Array of instance IDs for each point.
        tile_boundary: (min_x, max_x, min_y, max_y) of the tile.
        buffer: Buffer distance to shrink the boundary.

    Returns:
        Set of instance IDs that are fully contained within the buffered tile boundary.
    """
    min_x, max_x, min_y, max_y = tile_boundary
    
    # Vectorized boundary check for ALL points at once
    x_in = (tile_points[:, 0] >= min_x + buffer) & (tile_points[:, 0] <= max_x - buffer)
    y_in = (tile_points[:, 1] >= min_y + buffer) & (tile_points[:, 1] <= max_y - buffer)
    points_inside = x_in & y_in
    
    # For each tree, check if ALL its points are inside
    whole_tree_ids = set()
    unique_ids = np.unique(tile_pred_instance)
    
    for tree_id in unique_ids:
        # Skip background (0) and unassigned (-1 or any negative)
        if tree_id <= 0:
            continue
        mask = tile_pred_instance == tree_id
        if np.all(points_inside[mask]):
            whole_tree_ids.add(tree_id)
    
    return whole_tree_ids


def filter_by_lowest_point_in_buffer(
    tile_points: np.ndarray,
    tile_pred_instance: np.ndarray,
    tile_boundary: Tuple[float, float, float, float],
    buffer: float = 30.0,
    buffer_percent: float = None,
    min_buffer: float = 2.0,
) -> Set[int]:
    """
    Find instances whose lowest Z point is within the buffer zone.
    
    These instances will be removed (filtered out).
    
    Uses size-aware buffering: if buffer_percent is provided, uses percentage of
    smallest tile dimension, otherwise uses fixed buffer. Always applies at least min_buffer.
    
    Args:
        tile_points: Nx3 array of tile point coordinates.
        tile_pred_instance: Array of instance IDs for each point.
        tile_boundary: (min_x, max_x, min_y, max_y) of the tile.
        buffer: Fixed buffer distance from tile edges (used if buffer_percent is None).
        buffer_percent: Percentage of smallest dimension to use as buffer (e.g., 0.2 = 20%).
        min_buffer: Minimum buffer to always apply (default 2.0m).
    
    Returns:
        Set of instance IDs to remove (lowest point in buffer zone).
    """
    min_x, max_x, min_y, max_y = tile_boundary
    
    # Calculate tile dimensions
    tile_width = max_x - min_x
    tile_height = max_y - min_y
    min_dimension = min(tile_width, tile_height)
    
    # Determine actual buffer to use
    if buffer_percent is not None:
        # Use percentage of smallest dimension
        calculated_buffer = min_dimension * buffer_percent
        actual_buffer = max(calculated_buffer, min_buffer)
    else:
        # Use fixed buffer, but cap at half the smallest dimension to avoid removing everything
        actual_buffer = min(buffer, min_dimension * 0.4)
        actual_buffer = max(actual_buffer, min_buffer)
    
    # Define buffer zone boundaries
    buf_min_x = min_x + actual_buffer
    buf_max_x = max_x - actual_buffer
    buf_min_y = min_y + actual_buffer
    buf_max_y = max_y - actual_buffer
    
    # Find instances to remove
    instances_to_remove = set()
    unique_ids = np.unique(tile_pred_instance)
    
    for tree_id in unique_ids:
        # Skip background (0) and unassigned (-1 or any negative)
        if tree_id <= 0:
            continue
        
        mask = tile_pred_instance == tree_id
        tree_points = tile_points[mask]
        
        # Find lowest Z point for this instance
        lowest_z_idx = np.argmin(tree_points[:, 2])
        lowest_point = tree_points[lowest_z_idx]
        
        # Check if lowest point is in buffer zone
        x_in_buffer = (lowest_point[0] < buf_min_x) or (lowest_point[0] > buf_max_x)
        y_in_buffer = (lowest_point[1] < buf_min_y) or (lowest_point[1] > buf_max_y)
        
        if x_in_buffer or y_in_buffer:
            instances_to_remove.add(tree_id)
    
    return instances_to_remove


def filter_by_lowest_point_inner_edges(
    tile_points: np.ndarray,
    tile_pred_instance: np.ndarray,
    tile_boundary: Tuple[float, float, float, float],
    tile_name: str,
    all_tile_names: List[str],
    buffer: float = 30.0,
) -> Tuple[Set[int], Dict[str, bool]]:
    """
    Find instances whose lowest Z point is within the buffer zone, but only on edges
    that border neighboring tiles (inner edges), not on outer edges.
    
    Args:
        tile_points: Nx3 array of tile point coordinates.
        tile_pred_instance: Array of instance IDs for each point.
        tile_boundary: (min_x, max_x, min_y, max_y) of the tile.
        tile_name: Name of the tile (e.g., "c00_r00").
        all_tile_names: List of all tile names to determine neighbors.
        buffer: Buffer distance from inner edges.
    
    Returns:
        Tuple of (set of instance IDs to remove, dict with edge info: {'east': bool, 'west': bool, 'north': bool, 'south': bool})
        where True means that edge has a neighbor (inner edge, will be filtered).
    """
    min_x, max_x, min_y, max_y = tile_boundary
    
    # Parse tile name to get column and row
    # Format: c{col}_r{row} (e.g., "c00_r00")
    parts = tile_name.split('_')
    col_str = parts[0][1:]  # Extract number after 'c' (e.g., "00")
    row_str = parts[1][1:]  # Extract number after 'r' (e.g., "00")
    col = int(col_str)
    row = int(row_str)
    
    # Determine padding length from original strings
    col_padding = len(col_str)
    row_padding = len(row_str)
    
    # Format neighbor names with same padding
    def format_tile_name(c, r):
        return f"c{str(c).zfill(col_padding)}_r{str(r).zfill(row_padding)}"
    
    # Determine which edges have neighbors
    has_east_neighbor = format_tile_name(col+1, row) in all_tile_names
    has_west_neighbor = col > 0 and format_tile_name(col-1, row) in all_tile_names
    has_north_neighbor = format_tile_name(col, row+1) in all_tile_names
    has_south_neighbor = row > 0 and format_tile_name(col, row-1) in all_tile_names
    
    # Calculate tile dimensions for safety capping
    tile_width = max_x - min_x
    tile_height = max_y - min_y
    min_dimension = min(tile_width, tile_height)
    
    # Cap buffer at 40% of smallest dimension
    actual_buffer = min(buffer, min_dimension * 0.4)
    actual_buffer = max(actual_buffer, 2.0)
    
    # Define buffer zone boundaries only for inner edges
    buf_min_x = min_x + (actual_buffer if has_west_neighbor else 0)
    buf_max_x = max_x - (actual_buffer if has_east_neighbor else 0)
    buf_min_y = min_y + (actual_buffer if has_south_neighbor else 0)
    buf_max_y = max_y - (actual_buffer if has_north_neighbor else 0)
    
    # Find instances to remove
    instances_to_remove = set()
    unique_ids = np.unique(tile_pred_instance)
    
    for tree_id in unique_ids:
        # Skip background (0) and unassigned (-1 or any negative)
        if tree_id <= 0:
            continue
        
        mask = tile_pred_instance == tree_id
        tree_points = tile_points[mask]
        
        # Find lowest Z point for this instance
        lowest_z_idx = np.argmin(tree_points[:, 2])
        lowest_point = tree_points[lowest_z_idx]
        
        # Check if lowest point is in buffer zone (only on inner edges)
        x_in_buffer = False
        y_in_buffer = False
        
        if has_west_neighbor and lowest_point[0] < buf_min_x:
            x_in_buffer = True
        if has_east_neighbor and lowest_point[0] > buf_max_x:
            x_in_buffer = True
        if has_south_neighbor and lowest_point[1] < buf_min_y:
            y_in_buffer = True
        if has_north_neighbor and lowest_point[1] > buf_max_y:
            y_in_buffer = True
        
        if x_in_buffer or y_in_buffer:
            instances_to_remove.add(tree_id)
    
    edge_info = {
        'east': has_east_neighbor,
        'west': has_west_neighbor,
        'north': has_north_neighbor,
        'south': has_south_neighbor,
    }
    
    return instances_to_remove, edge_info


def load_tile_predictions(tile_folder: str) -> Dict[int, TreeSpeciesInfo]:
    """
    Load predictions.csv and build instance->species mapping.

    Args:
        tile_folder: Path to tile folder containing predictions.csv

    Returns:
        Dictionary mapping local instance ID (tree number) to TreeSpeciesInfo
    """
    predictions_path = os.path.join(tile_folder, "predictions.csv")
    if not os.path.exists(predictions_path):
        print(f"Warning: No predictions.csv found in {tile_folder}")
        return {}
    
    df = pd.read_csv(predictions_path)
    species_map = {}
    
    for _, row in df.iterrows():
        # Extract tree number from filename (e.g., "tree_1" -> 1)
        tree_name = row['filename']
        tree_id = int(tree_name.split('_')[1])
        
        species_map[tree_id] = TreeSpeciesInfo(
            species_id=int(row['species_id']),
            species_prob=float(row['species_prob']),
            species_name=str(row['species']),
            tree_height=float(row['tree_H'])
        )
    
    return species_map


def resolve_species_conflict(species_list: List[TreeSpeciesInfo]) -> TreeSpeciesInfo:
    """
    Pick species with highest probability when duplicates exist.

    Args:
        species_list: List of TreeSpeciesInfo from different tiles

    Returns:
        TreeSpeciesInfo with highest probability
    """
    if not species_list:
        raise ValueError("Empty species list provided")
    
    return max(species_list, key=lambda x: x.species_prob)


def deduplicate_points(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    tolerance: float = 0.001,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove duplicate points from overlapping tiles.
    
    When duplicate points exist, prefer the one with a non-zero instance ID
    (tree point over background).
    
    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        species_ids: Array of species IDs
        tolerance: Distance tolerance for considering points as duplicates (default 1mm)
    
    Returns:
        Tuple of (unique_points, unique_instances, unique_species_ids)
    """
    # Round coordinates to tolerance for grouping
    scale = 1.0 / tolerance
    rounded = np.round(points * scale).astype(np.int64)
    
    # Create unique key for each point
    # Use structured array for efficient unique finding
    dtype = [('x', np.int64), ('y', np.int64), ('z', np.int64)]
    keys = np.empty(len(rounded), dtype=dtype)
    keys['x'] = rounded[:, 0]
    keys['y'] = rounded[:, 1]
    keys['z'] = rounded[:, 2]
    
    # Find unique points, preferring those with instance > 0
    # Sort by key, then by instance (descending) so non-zero comes first
    sort_order = np.lexsort((-instances, keys['z'], keys['y'], keys['x']))
    
    sorted_keys = keys[sort_order]
    sorted_points = points[sort_order]
    sorted_instances = instances[sort_order]
    sorted_species = species_ids[sort_order]
    
    # Find first occurrence of each unique key (which will have highest instance due to sort)
    unique_mask = np.ones(len(sorted_keys), dtype=bool)
    unique_mask[1:] = (sorted_keys[1:] != sorted_keys[:-1])
    
    unique_points = sorted_points[unique_mask]
    unique_instances = sorted_instances[unique_mask]
    unique_species = sorted_species[unique_mask]
    
    removed = len(points) - len(unique_points)
    print(f"  Removed {removed} duplicate points ({100*removed/len(points):.1f}%)")
    
    return unique_points, unique_instances, unique_species


def load_single_tile(tile_subfolder: str, input_folder: str, buffer: float, filter_method: str = "whole_tree", all_tile_names: List[str] = None) -> Optional[dict]:
    """
    Load a single tile's data (LAZ file and predictions CSV).
    
    Args:
        tile_subfolder: Name of the tile subfolder
        input_folder: Base input folder path
        buffer: Buffer distance for filtering
        filter_method: "whole_tree" (all points inside buffer), "lowest_point" (lowest Z in buffer), or "lowest_point_inner" (lowest Z only on inner edges)
        all_tile_names: List of all tile names (needed for inner edge filtering)
        
    Returns:
        Dictionary with tile data or None if loading fails
    """
    tile_path = os.path.join(input_folder, tile_subfolder)
    laz_file = os.path.join(tile_path, "pc_with_species.laz")
    
    if not os.path.exists(laz_file):
        print(f"Warning: No pc_with_species.laz in {tile_subfolder}, skipping")
        return None
    
    print(f"Loading {tile_subfolder}...")
    
    # Load point cloud (laspy already uses parallel LAZ backend)
    tile_las = laspy.read(laz_file, laz_backend=laspy.LazBackend.LazrsParallel)
    tile_points = np.vstack((
        np.array(tile_las.x), 
        np.array(tile_las.y), 
        np.array(tile_las.z)
    )).T
    
    # Get instance IDs
    tile_pred_instance = np.array(tile_las.PredInstance)
    
    # Load species predictions
    species_map = load_tile_predictions(tile_path)
    
    # Compute tile boundary
    tile_boundary = (
        np.min(tile_points[:, 0]),
        np.max(tile_points[:, 0]),
        np.min(tile_points[:, 1]),
        np.max(tile_points[:, 1]),
    )
    
    # Apply filtering based on method
    if filter_method == "lowest_point_inner":
        # Find instances to remove (lowest point in buffer) only on inner edges (towards neighbors)
        if all_tile_names is None:
            raise ValueError("all_tile_names required for lowest_point_inner filter method")
        instances_to_remove, edge_info = filter_by_lowest_point_inner_edges(
            tile_points, tile_pred_instance, tile_boundary, tile_subfolder,
            all_tile_names, buffer
        )
        # Keep instances NOT in the remove set
        whole_tree_ids = set(np.unique(tile_pred_instance)) - instances_to_remove - {0, -1}
        # Report tile size for context
        tile_width = tile_boundary[1] - tile_boundary[0]
        tile_height = tile_boundary[3] - tile_boundary[2]
        # Build edge description
        filtered_edges = []
        preserved_edges = []
        if edge_info['east']:
            filtered_edges.append('east')
        else:
            preserved_edges.append('east')
        if edge_info['west']:
            filtered_edges.append('west')
        else:
            preserved_edges.append('west')
        if edge_info['north']:
            filtered_edges.append('north')
        else:
            preserved_edges.append('north')
        if edge_info['south']:
            filtered_edges.append('south')
        else:
            preserved_edges.append('south')
        edge_desc = f"filter: {', '.join(filtered_edges) if filtered_edges else 'none'}, preserve: {', '.join(preserved_edges) if preserved_edges else 'none'}"
        print(f"  {tile_subfolder}: {len(tile_points)} points, {len(instances_to_remove)} removed (lowest in {buffer:.1f}m buffer, {edge_desc}, tile {tile_width:.1f}m x {tile_height:.1f}m), {len(whole_tree_ids)} kept")
    elif filter_method == "lowest_point":
        # Find instances to remove (lowest point in buffer) on all edges
        instances_to_remove = filter_by_lowest_point_in_buffer(
            tile_points, tile_pred_instance, tile_boundary, 
            buffer=buffer, buffer_percent=None, min_buffer=buffer
        )
        # Keep instances NOT in the remove set
        whole_tree_ids = set(np.unique(tile_pred_instance)) - instances_to_remove - {0, -1}
        # Report tile size for context
        tile_width = tile_boundary[1] - tile_boundary[0]
        tile_height = tile_boundary[3] - tile_boundary[2]
        print(f"  {tile_subfolder}: {len(tile_points)} points, {len(instances_to_remove)} removed (lowest in {buffer:.1f}m buffer, tile {tile_width:.1f}m x {tile_height:.1f}m), {len(whole_tree_ids)} kept")
    else:
        # Default: whole tree method (all points inside buffer)
        whole_tree_ids = find_whole_trees(
            tile_points, tile_pred_instance, tile_boundary, buffer
        )
        print(f"  {tile_subfolder}: {len(tile_points)} points, {len(whole_tree_ids)} whole trees")
    
    return {
        'name': tile_subfolder,
        'points': tile_points,
        'pred_instance': tile_pred_instance,
        'whole_tree_ids': whole_tree_ids,
        'species_map': species_map,
        'boundary': tile_boundary,
        'las': tile_las,
    }


def merge_tiles_with_species(
    input_folder: str,
    output_merged_laz: str,
    output_tiles_folder: str,
    buffer: float = 0.2,
    min_cluster_size: int = 300,
    initial_radius: float = 1.0,
    max_radius: float = 5.0,
    radius_step: float = 1.0,
    num_threads: int = 8,
    filter_method: str = "whole_tree",
):
    """
    Merge tiles with species predictions into a single point cloud.

    Args:
        input_folder: Path to detailview_predictions folder containing tile subfolders
        output_merged_laz: Path for output merged LAZ file
        output_tiles_folder: Path for per-tile output folder
        buffer: Buffer distance for filtering (default 0.2m)
        min_cluster_size: Minimum cluster size for reassignment
        filter_method: "whole_tree" (all points inside buffer), "lowest_point" (lowest Z in buffer on all edges), or "lowest_point_inner" (lowest Z only on inner edges)
        initial_radius: Initial search radius for point reassignment
        max_radius: Maximum search radius for point reassignment
        radius_step: Radius increment step for point reassignment
        num_threads: Number of threads for parallel processing (default 8)
    """
    print("=" * 60)
    print("Merge Species Tiles")
    print("=" * 60)
    
    # Find all tile subfolders
    tile_subfolders = sorted([
        d for d in os.listdir(input_folder) 
        if os.path.isdir(os.path.join(input_folder, d)) and d.endswith('_segmented_remapped')
    ])
    
    print(f"Found {len(tile_subfolders)} tile folders to merge")
    
    # Extract tile names (remove _segmented_remapped suffix)
    tile_names = [name.replace('_segmented_remapped', '') for name in tile_subfolders]
    
    # First pass: Load all tiles in parallel
    print(f"\n--- Loading tiles (using {num_threads} threads) ---")
    
    # Load tiles in parallel using ThreadPoolExecutor
    load_func = partial(load_single_tile, input_folder=input_folder, buffer=buffer, filter_method=filter_method, all_tile_names=tile_names)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        loaded_tiles = list(executor.map(load_func, tile_subfolders))
    
    # Filter out None (failed loads) and sort by name for consistent ordering
    loaded_tiles = [t for t in loaded_tiles if t is not None]
    loaded_tiles.sort(key=lambda x: x['name'])
    
    # Second pass: Assign global instance IDs sequentially
    print("\n--- Assigning global instance IDs ---")
    all_tile_data = []
    global_instance_counter = 1  # Start from 1 (0 is reserved for ground)
    
    # Mapping: (tile_name, local_instance_id) -> global_instance_id
    instance_mapping = {}
    # Mapping: global_instance_id -> TreeSpeciesInfo
    global_species_map = {}
    
    for tile_data in loaded_tiles:
        tile_subfolder = tile_data['name']
        whole_tree_ids = tile_data['whole_tree_ids']
        species_map = tile_data['species_map']
        tile_pred_instance = tile_data['pred_instance']
        
        # Create local-to-global instance mapping for whole trees
        local_to_global = {0: 0}  # Ground stays 0
        for local_id in whole_tree_ids:
            if local_id not in local_to_global:
                local_to_global[local_id] = global_instance_counter
                instance_mapping[(tile_subfolder, local_id)] = global_instance_counter
                
                # Store species info
                if local_id in species_map:
                    global_species_map[global_instance_counter] = species_map[local_id]
                
                global_instance_counter += 1
        
        # Reindex instances (vectorized)
        reindexed_pred_instance = np.array([
            local_to_global.get(pid, -1) for pid in tile_pred_instance
        ])
        
        # Update tile data with global mappings
        tile_data['local_to_global'] = local_to_global
        tile_data['reindexed_instance'] = reindexed_pred_instance
        all_tile_data.append(tile_data)
    
    print(f"\nTotal global instances: {global_instance_counter - 1}")
    
    # Second pass: Merge all points
    print("\n--- Merging points ---")
    
    # Collect all points and attributes
    all_points = []
    all_instances = []
    all_species_ids = []
    
    for tile_data in all_tile_data:
        # Include ALL points from tile (whole trees get global IDs, others get 0)
        points_to_add = tile_data['points']
        instances_to_add = tile_data['reindexed_instance'].copy()
        
        # Points not belonging to whole trees get instance 0 (background)
        # They will be handled in overlap resolution later
        non_whole_tree_mask = ~np.isin(tile_data['pred_instance'], list(tile_data['whole_tree_ids']))
        instances_to_add[non_whole_tree_mask] = 0
        
        # Get species IDs for each point
        species_ids = np.zeros(len(instances_to_add), dtype=np.int32)
        for i, inst_id in enumerate(instances_to_add):
            if inst_id in global_species_map:
                species_ids[i] = global_species_map[inst_id].species_id
        
        all_points.append(points_to_add)
        all_instances.append(instances_to_add)
        all_species_ids.append(species_ids)
        
        whole_tree_points = np.sum(~non_whole_tree_mask)
        print(f"  {tile_data['name']}: {len(points_to_add)} total points ({whole_tree_points} in whole trees)")
    
    # Concatenate all arrays
    merged_points = np.vstack(all_points)
    merged_instances = np.concatenate(all_instances)
    merged_species_ids = np.concatenate(all_species_ids)
    
    print(f"\nTotal merged points (before dedup): {len(merged_points)}")
    
    # Remove duplicate points from overlapping tiles
    # Keep the point with non-zero instance if available (prefer tree over background)
    print("\n--- Removing duplicate points from tile overlaps ---")
    merged_points, merged_instances, merged_species_ids = deduplicate_points(
        merged_points, merged_instances, merged_species_ids
    )
    
    print(f"Total merged points (after dedup): {len(merged_points)}")
    print(f"Unique instances in merged cloud: {len(np.unique(merged_instances))}")
    
    # Reassign small clusters
    print("\n--- Reassigning small clusters ---")
    reassign_small_clusters(
        merged_instances,
        merged_points,
        merged_species_ids,
        global_species_map,
        min_cluster_size,
        initial_radius,
        max_radius,
        radius_step,
        num_threads,
    )
    
    # Merge overlapping instances (same tree detected in multiple tiles)
    print("\n--- Merging overlapping instances ---")
    merge_overlapping_instances(
        merged_points,
        merged_instances,
        merged_species_ids,
        global_species_map,
        min_iou=0.1,
        max_centroid_dist=5.0,
    )
    
    # Renumber instances to remove gaps
    print("\n--- Renumbering instances to remove gaps ---")
    merged_instances, global_species_map = renumber_instances(merged_instances, global_species_map)
    
    # Update species_ids after renumbering
    for i, inst_id in enumerate(merged_instances):
        if inst_id in global_species_map:
            merged_species_ids[i] = global_species_map[inst_id].species_id
        elif inst_id > 0:
            merged_species_ids[i] = -1  # Unknown species
    
    # Save merged LAZ
    print("\n--- Saving merged LAZ ---")
    save_merged_laz(
        merged_points, 
        merged_instances, 
        merged_species_ids,
        output_merged_laz,
    )
    
    # Generate merged predictions CSV
    merged_csv_path = output_merged_laz.replace('.laz', '_predictions.csv')
    save_merged_predictions(global_species_map, merged_csv_path)
    
    # Generate per-tile outputs with updated LAZ files
    print("\n--- Generating per-tile LAZ outputs ---")
    generate_per_tile_laz_outputs(
        all_tile_data,
        merged_points,
        merged_instances,
        merged_species_ids,
        global_species_map,
        output_tiles_folder,
        num_threads,
    )
    
    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"  Merged LAZ: {output_merged_laz}")
    print(f"  Merged CSV: {merged_csv_path}")
    print(f"  Per-tile outputs: {output_tiles_folder}")
    print("=" * 60)


def reassign_small_clusters(
    merged_instances: np.ndarray,
    merged_points: np.ndarray,
    merged_species_ids: np.ndarray,
    global_species_map: Dict[int, TreeSpeciesInfo],
    min_cluster_size: int = 300,
    initial_radius: float = 1.0,
    max_radius: float = 5.0,
    radius_step: float = 1.0,
    num_threads: int = 8,
):
    """
    Reassign points in small clusters to the nearest larger cluster.

    Args:
        merged_instances: Array of instance IDs (modified in-place)
        merged_points: Nx3 array of point coordinates
        merged_species_ids: Array of species IDs (modified in-place)
        global_species_map: Mapping of instance ID to species info
        min_cluster_size: Minimum size for a cluster to be considered valid
        initial_radius: Starting search radius for reassignment
        max_radius: Maximum search radius for reassignment
        radius_step: Step size to increase the search radius
        num_threads: Number of threads for parallel KDTree queries
    """
    unique, counts = np.unique(merged_instances, return_counts=True)
    small_instances = unique[(counts < min_cluster_size) & (unique > 0)]
    
    print(f"Found {len(small_instances)} small clusters (< {min_cluster_size} points)")
    
    if len(small_instances) == 0:
        return
    
    kdtree = KDTree(merged_points)
    
    reassigned_count = 0
    not_reassigned_count = 0
    
    for small_instance in small_instances:
        indices = np.where(merged_instances == small_instance)[0]
        for idx in indices:
            point = merged_points[idx]
            radius = initial_radius
            reassigned = False
            
            while radius <= max_radius and not reassigned:
                nearest_indices = kdtree.query_ball_point(point, radius)
                for nearest_idx in nearest_indices:
                    nearest_instance = merged_instances[nearest_idx]
                    if nearest_instance != small_instance and nearest_instance > 0:
                        merged_instances[idx] = nearest_instance
                        # Update species ID
                        if nearest_instance in global_species_map:
                            merged_species_ids[idx] = global_species_map[nearest_instance].species_id
                        reassigned_count += 1
                        reassigned = True
                        break
                radius += radius_step
            
            if not reassigned:
                merged_instances[idx] = -1
                merged_species_ids[idx] = -1
                not_reassigned_count += 1
    
    print(f"  Reassigned {reassigned_count} points")
    print(f"  Could not reassign {not_reassigned_count} points (set to -1)")


def merge_overlapping_instances(
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_species_ids: np.ndarray,
    global_species_map: Dict[int, TreeSpeciesInfo],
    min_iou: float = 0.3,
    max_centroid_dist: float = 5.0,
):
    """
    Merge instances that represent the same tree detected in multiple tiles.
    
    Uses strict criteria to avoid merging different trees:
    - High IoU (Intersection over Union) of bounding boxes
    - AND centroids must be close
    
    Args:
        merged_points: Nx3 array of point coordinates
        merged_instances: Array of instance IDs (modified in-place)
        merged_species_ids: Array of species IDs (modified in-place)
        global_species_map: Mapping of instance ID to species info
        min_iou: Minimum IoU to consider merging (default 0.3)
        max_centroid_dist: Maximum centroid distance in meters (default 5.0)
    """
    unique_instances = np.unique(merged_instances)
    unique_instances = unique_instances[unique_instances > 0]
    
    if len(unique_instances) < 2:
        print("  No instances to merge")
        return
    
    # Compute 2D bounding box and centroid for each instance
    instance_bboxes = {}  # instance_id -> (min_x, max_x, min_y, max_y)
    instance_centroids = {}  # instance_id -> (cx, cy)
    instance_counts = {}  # instance_id -> point count
    
    for inst_id in unique_instances:
        mask = merged_instances == inst_id
        inst_points = merged_points[mask]
        instance_bboxes[inst_id] = (
            np.min(inst_points[:, 0]), np.max(inst_points[:, 0]),
            np.min(inst_points[:, 1]), np.max(inst_points[:, 1]),
        )
        instance_centroids[inst_id] = (
            np.mean(inst_points[:, 0]), np.mean(inst_points[:, 1])
        )
        instance_counts[inst_id] = np.sum(mask)
    
    # Union-Find data structure
    parent = {inst_id: inst_id for inst_id in unique_instances}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            # Merge smaller into larger (by point count)
            if instance_counts.get(px, 0) >= instance_counts.get(py, 0):
                parent[py] = px
            else:
                parent[px] = py
    
    # Find overlapping pairs
    instance_list = list(unique_instances)
    merge_count = 0
    
    for i in range(len(instance_list)):
        for j in range(i + 1, len(instance_list)):
            id_a, id_b = instance_list[i], instance_list[j]
            
            # Already in same group?
            if find(id_a) == find(id_b):
                continue
            
            # Check centroid distance first (fast filter)
            cx_a, cy_a = instance_centroids[id_a]
            cx_b, cy_b = instance_centroids[id_b]
            centroid_dist = np.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
            
            if centroid_dist > max_centroid_dist:
                continue  # Centroids too far apart
            
            # Get bboxes
            min_x_a, max_x_a, min_y_a, max_y_a = instance_bboxes[id_a]
            min_x_b, max_x_b, min_y_b, max_y_b = instance_bboxes[id_b]
            
            # Compute intersection
            inter_min_x = max(min_x_a, min_x_b)
            inter_max_x = min(max_x_a, max_x_b)
            inter_min_y = max(min_y_a, min_y_b)
            inter_max_y = min(max_y_a, max_y_b)
            
            if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
                continue  # No overlap
            
            intersection = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
            
            # Compute union
            area_a = (max_x_a - min_x_a) * (max_y_a - min_y_a)
            area_b = (max_x_b - min_x_b) * (max_y_b - min_y_b)
            union_area = area_a + area_b - intersection
            
            # Compute IoU
            if union_area > 0:
                iou = intersection / union_area
                
                if iou >= min_iou:
                    union(id_a, id_b)
                    merge_count += 1
    
    if merge_count == 0:
        print("  No overlapping instances to merge")
        return
    
    # Group instances by their root
    groups = {}
    for inst_id in unique_instances:
        root = find(inst_id)
        if root not in groups:
            groups[root] = []
        groups[root].append(inst_id)
    
    # Merge groups (reassign all instances in group to root)
    merged_groups = 0
    for root, members in groups.items():
        if len(members) > 1:
            merged_groups += 1
            for member in members:
                if member != root:
                    # Reassign points from member to root
                    mask = merged_instances == member
                    merged_instances[mask] = root
                    # Update species if root has species info
                    if root in global_species_map:
                        merged_species_ids[mask] = global_species_map[root].species_id
    
    # Update instance counts after merging
    final_unique = np.unique(merged_instances)
    final_unique = final_unique[final_unique > 0]
    
    print(f"  Found {merge_count} overlapping pairs (IoU >= {min_iou}, centroid dist <= {max_centroid_dist}m)")
    print(f"  Merged {merged_groups} groups of instances")
    print(f"  Instances after merge: {len(final_unique)}")


def renumber_instances(
    merged_instances: np.ndarray, 
    global_species_map: Dict[int, TreeSpeciesInfo]
) -> Tuple[np.ndarray, Dict[int, TreeSpeciesInfo]]:
    """
    Renumber instances to remove gaps (1, 2, 3, ... instead of 1, 5, 12, ...).

    Args:
        merged_instances: Array of instance IDs
        global_species_map: Original species mapping

    Returns:
        Tuple of (renumbered instances array, new species mapping)
    """
    unique_instances = sorted(set(merged_instances) - {-1, 0})
    
    # Create old -> new mapping
    old_to_new = {0: 0, -1: -1}
    new_species_map = {}
    
    for new_id, old_id in enumerate(unique_instances, start=1):
        old_to_new[old_id] = new_id
        if old_id in global_species_map:
            new_species_map[new_id] = global_species_map[old_id]
    
    # Apply renumbering
    renumbered = np.array([old_to_new.get(x, -1) for x in merged_instances])
    
    print(f"  Renumbered {len(unique_instances)} instances (1 to {len(unique_instances)})")
    
    return renumbered, new_species_map


def save_merged_laz(
    points: np.ndarray,
    instances: np.ndarray,
    species_ids: np.ndarray,
    output_path: str,
):
    """
    Save merged point cloud to LAZ file.

    Args:
        points: Nx3 array of point coordinates
        instances: Array of instance IDs
        species_ids: Array of species IDs
        output_path: Output file path
    """
    # Create output directory if needed
    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create header with extra dimensions FIRST (before creating point record)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.add_extra_dim(laspy.ExtraBytesParams(name="PredInstance", type=np.int32))
    header.add_extra_dim(laspy.ExtraBytesParams(name="species_id", type=np.int32))
    
    # Create LAS data with the header
    merged_las = laspy.LasData(header)
    
    # Set coordinates and attributes
    merged_las.x = points[:, 0]
    merged_las.y = points[:, 1]
    merged_las.z = points[:, 2]
    merged_las.PredInstance = instances.astype(np.int32)
    merged_las.species_id = species_ids.astype(np.int32)
    
    # Write file
    merged_las.write(
        output_path, 
        do_compress=True, 
        laz_backend=laspy.LazBackend.LazrsParallel
    )
    
    print(f"  Saved merged LAZ: {output_path}")
    print(f"  Total points: {len(points)}")


def save_merged_predictions(
    global_species_map: Dict[int, TreeSpeciesInfo],
    output_path: str,
):
    """
    Save merged predictions to CSV file.

    Args:
        global_species_map: Mapping of global instance ID to species info
        output_path: Output CSV path
    """
    rows = []
    for instance_id, info in sorted(global_species_map.items()):
        rows.append({
            'filename': f'tree_{instance_id}',
            'species_id': info.species_id,
            'species_prob': info.species_prob,
            'tree_H': info.tree_height,
            'species': info.species_name,
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    print(f"  Saved merged predictions: {output_path}")
    print(f"  Total trees: {len(rows)}")


def generate_per_tile_outputs(
    all_tile_data: List[dict],
    global_species_map: Dict[int, TreeSpeciesInfo],
    output_folder: str,
):
    """
    Generate per-tile output folders with updated predictions.csv.

    Args:
        all_tile_data: List of tile data dictionaries
        global_species_map: Global species mapping after merging
        output_folder: Base output folder path
    """
    os.makedirs(output_folder, exist_ok=True)
    
    for tile_data in all_tile_data:
        tile_name = tile_data['name'].replace('_segmented_remapped', '_merged')
        tile_output_folder = os.path.join(output_folder, tile_name)
        os.makedirs(tile_output_folder, exist_ok=True)
        
        # Create mapping file: local_id -> global_id -> species_info
        rows = []
        for local_id, global_id in tile_data['local_to_global'].items():
            if local_id == 0:  # Skip ground
                continue
            
            if global_id in global_species_map:
                info = global_species_map[global_id]
                rows.append({
                    'filename': f'tree_{global_id}',
                    'local_instance_id': local_id,
                    'global_instance_id': global_id,
                    'species_id': info.species_id,
                    'species_prob': info.species_prob,
                    'tree_H': info.tree_height,
                    'species': info.species_name,
                })
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(tile_output_folder, 'predictions.csv')
            df.to_csv(csv_path, index=False)
            
            print(f"  {tile_name}: {len(rows)} trees")


def generate_per_tile_laz_outputs(
    all_tile_data: List[dict],
    merged_points: np.ndarray,
    merged_instances: np.ndarray,
    merged_species_ids: np.ndarray,
    global_species_map: Dict[int, TreeSpeciesInfo],
    output_folder: str,
    num_threads: int = 8,
):
    """
    Generate per-tile LAZ files with updated instance IDs from the merged cloud.
    
    This remaps the merged/reassigned instances back to each original tile's points
    using nearest-neighbor matching.

    Args:
        all_tile_data: List of tile data dictionaries with original points
        merged_points: Merged point cloud coordinates
        merged_instances: Updated instance IDs after merge/reassignment
        merged_species_ids: Updated species IDs
        global_species_map: Global species mapping after merging
        output_folder: Base output folder path
        num_threads: Number of threads for KDTree queries
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Build KDTree from merged points for fast lookup
    print("  Building KDTree from merged points...")
    tree = KDTree(merged_points)
    
    for tile_data in all_tile_data:
        tile_name = tile_data['name'].replace('_segmented_remapped', '_merged')
        tile_output_folder = os.path.join(output_folder, tile_name)
        os.makedirs(tile_output_folder, exist_ok=True)
        
        tile_points = tile_data['points']
        tile_las = tile_data['las']
        
        # Find nearest neighbors in merged cloud for each tile point
        distances, indices = tree.query(tile_points, workers=num_threads)
        
        # Map instances and species from merged to tile
        tile_instances = merged_instances[indices].astype(np.int32)
        tile_species = merged_species_ids[indices].astype(np.int32)
        
        # Points too far from merged cloud get background (0)
        # This handles points that were in border instances (excluded)
        max_distance = 0.1  # 10cm tolerance for matching
        too_far = distances > max_distance
        tile_instances[too_far] = 0
        tile_species[too_far] = 0
        
        # Add/update extra dimensions
        extra_dims = {dim.name for dim in tile_las.point_format.extra_dimensions}
        
        if "PredInstance" not in extra_dims:
            tile_las.add_extra_dim(
                laspy.ExtraBytesParams(name="PredInstance", type=np.int32)
            )
        if "species_id" not in extra_dims:
            tile_las.add_extra_dim(
                laspy.ExtraBytesParams(name="species_id", type=np.int32)
            )
        
        tile_las.PredInstance = tile_instances
        tile_las.species_id = tile_species
        
        # Save LAZ file
        output_laz = os.path.join(tile_output_folder, "pc_with_species_merged.laz")
        tile_las.write(
            output_laz,
            do_compress=True,
            laz_backend=laspy.LazBackend.LazrsParallel
        )
        
        # Count instances
        unique_instances = np.unique(tile_instances)
        num_trees = len(unique_instances[unique_instances > 0])
        num_background = np.sum(tile_instances == 0)
        
        # Save predictions CSV
        rows = []
        for inst_id in unique_instances:
            if inst_id <= 0:
                continue
            if inst_id in global_species_map:
                info = global_species_map[inst_id]
                rows.append({
                    'filename': f'tree_{inst_id}',
                    'global_instance_id': inst_id,
                    'species_id': info.species_id,
                    'species_prob': info.species_prob,
                    'tree_H': info.tree_height,
                    'species': info.species_name,
                    'num_points': np.sum(tile_instances == inst_id),
                })
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(tile_output_folder, 'predictions.csv')
            df.to_csv(csv_path, index=False)
        
        print(f"  {tile_name}: {num_trees} trees, {num_background} background pts → {output_laz}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LAZ tiles with species predictions"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="/home/kg281/data/kaltenborn25_lidar/bechstedt/tiles_200m/detailview_predictions",
        help="Path to detailview_predictions folder"
    )
    parser.add_argument(
        "--output_merged_laz",
        type=str,
        default="/home/kg281/data/kaltenborn25_lidar/bechstedt/merged_detailview.laz",
        help="Output path for merged LAZ file"
    )
    parser.add_argument(
        "--output_tiles_folder",
        type=str,
        default="/home/kg281/data/kaltenborn25_lidar/bechstedt/tiles_200m/detailview_merged",
        help="Output folder for per-tile results"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.2,
        help="Buffer distance for whole-tree check (default: 0.2m)"
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=300,
        help="Minimum cluster size for reassignment (default: 300)"
    )
    parser.add_argument(
        "--initial_radius",
        type=float,
        default=1.0,
        help="Initial search radius for point reassignment (default: 1.0)"
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=5.0,
        help="Maximum search radius for point reassignment (default: 5.0)"
    )
    parser.add_argument(
        "--radius_step",
        type=float,
        default=1.0,
        help="Radius increment step for point reassignment (default: 1.0)"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads for parallel processing (default: 8)"
    )
    parser.add_argument(
        "--filter_method",
        type=str,
        default="whole_tree",
        choices=["whole_tree", "lowest_point", "lowest_point_inner"],
        help="Filtering method: 'whole_tree' (all points inside buffer), 'lowest_point' (lowest Z in buffer on all edges), or 'lowest_point_inner' (lowest Z only on inner edges towards neighbors) (default: whole_tree)"
    )
    
    args = parser.parse_args()
    
    merge_tiles_with_species(
        input_folder=args.input_folder,
        output_merged_laz=args.output_merged_laz,
        output_tiles_folder=args.output_tiles_folder,
        buffer=args.buffer,
        min_cluster_size=args.min_cluster_size,
        initial_radius=args.initial_radius,
        max_radius=args.max_radius,
        radius_step=args.radius_step,
        num_threads=args.num_threads,
        filter_method=args.filter_method,
    )


if __name__ == "__main__":
    main()
