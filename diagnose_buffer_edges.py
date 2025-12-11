#!/usr/bin/env python3
"""
Diagnostic script to verify buffer is applied at all 4 edges of each tile
and check for trees excluded from ALL tiles (outer edge gaps).
"""

import os
import numpy as np
import laspy
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, List


def get_tree_info(laz_path: str) -> Dict[int, dict]:
    """
    Load LAZ and compute tree info (centroid, bounding box, point count).
    
    Returns dict: tree_id -> {centroid, bbox, count}
    """
    las = laspy.read(laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    points = np.vstack((las.x, las.y, las.z)).T
    instances = np.array(las.PredInstance)
    
    tree_info = {}
    unique_ids = np.unique(instances)
    
    for tree_id in unique_ids:
        if tree_id <= 0:
            continue
        mask = instances == tree_id
        tree_points = points[mask]
        tree_info[tree_id] = {
            'centroid': (np.mean(tree_points[:, 0]), np.mean(tree_points[:, 1])),
            'bbox': (
                np.min(tree_points[:, 0]), np.max(tree_points[:, 0]),
                np.min(tree_points[:, 1]), np.max(tree_points[:, 1]),
            ),
            'count': len(tree_points),
        }
    
    return tree_info


def analyze_buffer_edges(tile_folder: str, buffer: float = 0.2) -> dict:
    """
    Analyze which trees are excluded due to touching each edge's buffer zone.
    
    Returns dict with:
    - tile_boundary: (min_x, max_x, min_y, max_y)
    - all_trees: set of all tree IDs
    - whole_trees: set of trees fully inside buffer
    - excluded_trees: set of trees touching buffer
    - excluded_by_edge: dict mapping edge name to set of tree IDs
    """
    laz_path = os.path.join(tile_folder, "pc_with_species.laz")
    if not os.path.exists(laz_path):
        return None
    
    las = laspy.read(laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    points = np.vstack((las.x, las.y, las.z)).T
    instances = np.array(las.PredInstance)
    
    # Tile boundary
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])
    
    # Buffered boundary
    buf_min_x = min_x + buffer
    buf_max_x = max_x - buffer
    buf_min_y = min_y + buffer
    buf_max_y = max_y - buffer
    
    # Analyze each tree
    all_trees = set()
    whole_trees = set()
    excluded_by_edge = {
        'left': set(),    # touches min_x buffer
        'right': set(),   # touches max_x buffer
        'bottom': set(),  # touches min_y buffer
        'top': set(),     # touches max_y buffer
    }
    
    unique_ids = np.unique(instances)
    
    for tree_id in unique_ids:
        if tree_id <= 0:
            continue
        
        all_trees.add(tree_id)
        mask = instances == tree_id
        tree_points = points[mask]
        
        # Check each edge
        touches_left = np.any(tree_points[:, 0] < buf_min_x)
        touches_right = np.any(tree_points[:, 0] > buf_max_x)
        touches_bottom = np.any(tree_points[:, 1] < buf_min_y)
        touches_top = np.any(tree_points[:, 1] > buf_max_y)
        
        if touches_left:
            excluded_by_edge['left'].add(tree_id)
        if touches_right:
            excluded_by_edge['right'].add(tree_id)
        if touches_bottom:
            excluded_by_edge['bottom'].add(tree_id)
        if touches_top:
            excluded_by_edge['top'].add(tree_id)
        
        # Whole tree = doesn't touch any buffer
        if not (touches_left or touches_right or touches_bottom or touches_top):
            whole_trees.add(tree_id)
    
    excluded_trees = all_trees - whole_trees
    
    return {
        'tile_boundary': (min_x, max_x, min_y, max_y),
        'all_trees': all_trees,
        'whole_trees': whole_trees,
        'excluded_trees': excluded_trees,
        'excluded_by_edge': excluded_by_edge,
        'tree_info': get_tree_info(laz_path),
    }


def find_tiles_containing_point(x: float, y: float, tile_boundaries: Dict[str, Tuple]) -> List[str]:
    """Find which tiles contain a given point."""
    containing = []
    for tile_name, (min_x, max_x, min_y, max_y) in tile_boundaries.items():
        if min_x <= x <= max_x and min_y <= y <= max_y:
            containing.append(tile_name)
    return containing


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose buffer edge behavior")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to detailview_predictions folder")
    parser.add_argument("--buffer", type=float, default=0.2,
                        help="Buffer distance (default 0.2m)")
    
    args = parser.parse_args()
    
    # Find all tile folders
    tile_folders = sorted([
        d for d in os.listdir(args.input_folder)
        if os.path.isdir(os.path.join(args.input_folder, d)) and d.endswith('_segmented_remapped')
    ])
    
    print("=" * 80)
    print("Buffer Edge Diagnostic")
    print("=" * 80)
    print(f"Buffer distance: {args.buffer}m")
    print(f"Found {len(tile_folders)} tiles")
    print()
    
    # Analyze each tile
    tile_data = {}
    tile_boundaries = {}
    
    for tile_name in tile_folders:
        tile_path = os.path.join(args.input_folder, tile_name)
        result = analyze_buffer_edges(tile_path, args.buffer)
        if result:
            tile_data[tile_name] = result
            tile_boundaries[tile_name] = result['tile_boundary']
    
    # Print per-tile results
    print("=" * 80)
    print("PER-TILE BUFFER ANALYSIS")
    print("=" * 80)
    
    for tile_name, data in tile_data.items():
        min_x, max_x, min_y, max_y = data['tile_boundary']
        print(f"\n{tile_name}:")
        print(f"  Boundary: X[{min_x:.2f}, {max_x:.2f}], Y[{min_y:.2f}, {max_y:.2f}]")
        print(f"  Total trees: {len(data['all_trees'])}")
        print(f"  Whole trees (inside buffer): {len(data['whole_trees'])}")
        print(f"  Excluded trees (touch buffer): {len(data['excluded_trees'])}")
        print(f"  Excluded by edge:")
        for edge, trees in data['excluded_by_edge'].items():
            print(f"    {edge:8s}: {len(trees):3d} trees")
    
    # Check for trees excluded from ALL tiles
    print("\n" + "=" * 80)
    print("TREES EXCLUDED FROM ALL TILES (outer edge gaps)")
    print("=" * 80)
    
    # Collect all excluded trees with their centroids
    all_excluded_trees = []  # (tile_name, tree_id, centroid_x, centroid_y)
    
    for tile_name, data in tile_data.items():
        for tree_id in data['excluded_trees']:
            info = data['tree_info'][tree_id]
            cx, cy = info['centroid']
            all_excluded_trees.append((tile_name, tree_id, cx, cy))
    
    # For each excluded tree, check if its centroid is inside any other tile's buffered region
    trees_with_no_rescue = []
    
    for tile_name, tree_id, cx, cy in all_excluded_trees:
        # Check all other tiles
        rescued = False
        for other_tile, other_data in tile_data.items():
            if other_tile == tile_name:
                continue
            
            min_x, max_x, min_y, max_y = other_data['tile_boundary']
            buf_min_x = min_x + args.buffer
            buf_max_x = max_x - args.buffer
            buf_min_y = min_y + args.buffer
            buf_max_y = max_y - args.buffer
            
            # Check if centroid is inside other tile's buffered region
            if buf_min_x <= cx <= buf_max_x and buf_min_y <= cy <= buf_max_y:
                rescued = True
                break
        
        if not rescued:
            trees_with_no_rescue.append((tile_name, tree_id, cx, cy))
    
    if trees_with_no_rescue:
        print(f"\nFound {len(trees_with_no_rescue)} trees at outer edges (no overlap to rescue):")
        for tile_name, tree_id, cx, cy in trees_with_no_rescue[:20]:  # Show first 20
            info = tile_data[tile_name]['tree_info'][tree_id]
            print(f"  {tile_name} tree {tree_id}: centroid ({cx:.2f}, {cy:.2f}), {info['count']} points")
        if len(trees_with_no_rescue) > 20:
            print(f"  ... and {len(trees_with_no_rescue) - 20} more")
    else:
        print("\nAll excluded trees are rescued by overlapping tiles!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_all = sum(len(d['all_trees']) for d in tile_data.values())
    total_whole = sum(len(d['whole_trees']) for d in tile_data.values())
    total_excluded = sum(len(d['excluded_trees']) for d in tile_data.values())
    
    print(f"Total trees across all tiles: {total_all}")
    print(f"Total whole trees (inside buffer): {total_whole}")
    print(f"Total excluded (touch buffer): {total_excluded}")
    print(f"Trees at outer edge (no rescue): {len(trees_with_no_rescue)}")
    print(f"Trees rescued by overlap: {total_excluded - len(trees_with_no_rescue)}")


if __name__ == "__main__":
    main()


