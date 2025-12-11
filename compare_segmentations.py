#!/usr/bin/env python3
"""
Compare two segmentations of the same point cloud to understand differences.
Uses point-level matching to identify:
- Matched instances (same tree detected in both)
- Split instances (one reference tree = multiple test trees)
- Missing instances (reference tree not in test)
- Extra instances (test tree not in reference)
"""

import numpy as np
import laspy
from scipy.spatial import KDTree
from collections import defaultdict
import argparse


def load_segmentation(laz_path):
    """Load LAZ and return points and instances."""
    print(f"Loading {laz_path}...")
    las = laspy.read(laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    points = np.vstack((las.x, las.y, las.z)).T
    instances = np.array(las.PredInstance)
    return points, instances


def compute_instance_stats(points, instances):
    """Compute stats for each instance."""
    stats = {}
    unique_ids = np.unique(instances)
    
    for inst_id in unique_ids:
        if inst_id <= 0:
            continue
        mask = instances == inst_id
        inst_points = points[mask]
        stats[inst_id] = {
            'count': len(inst_points),
            'centroid': (np.mean(inst_points[:, 0]), np.mean(inst_points[:, 1]), np.mean(inst_points[:, 2])),
            'bbox': (
                np.min(inst_points[:, 0]), np.max(inst_points[:, 0]),
                np.min(inst_points[:, 1]), np.max(inst_points[:, 1]),
                np.min(inst_points[:, 2]), np.max(inst_points[:, 2]),
            ),
        }
    return stats


def match_instances(ref_points, ref_instances, test_points, test_instances):
    """
    Match instances between reference and test segmentations.
    
    Returns:
        overlap_matrix: dict mapping (ref_id, test_id) -> overlap_count
        ref_stats: stats for reference instances
        test_stats: stats for test instances
    """
    # Build KDTree for test points
    print("Building KDTree for point matching...")
    test_tree = KDTree(test_points)
    
    # For each reference point, find nearest test point
    print("Finding nearest neighbors...")
    distances, indices = test_tree.query(ref_points, workers=-1)
    
    # Count overlaps: for each ref instance, count how many points map to each test instance
    print("Computing instance overlaps...")
    overlap_matrix = defaultdict(int)
    
    ref_unique = np.unique(ref_instances)
    for ref_id in ref_unique:
        if ref_id <= 0:
            continue
        ref_mask = ref_instances == ref_id
        # Get test instances for these points
        matched_test_instances = test_instances[indices[ref_mask]]
        # Count occurrences
        test_unique, test_counts = np.unique(matched_test_instances, return_counts=True)
        for test_id, count in zip(test_unique, test_counts):
            if test_id > 0:  # Skip background
                overlap_matrix[(ref_id, test_id)] = count
    
    ref_stats = compute_instance_stats(ref_points, ref_instances)
    test_stats = compute_instance_stats(test_points, test_instances)
    
    return overlap_matrix, ref_stats, test_stats


def analyze_matches(overlap_matrix, ref_stats, test_stats, min_overlap_ratio=0.3):
    """
    Analyze the overlap matrix to classify instances.
    
    Returns dict with:
        - matched: list of (ref_id, test_id, overlap_ratio) for 1:1 matches
        - split: list of (ref_id, [test_ids]) for 1:N splits
        - missing: list of ref_ids not matched
        - extra: list of test_ids not matched
    """
    # For each ref instance, find best matching test instance(s)
    ref_to_test = defaultdict(list)  # ref_id -> [(test_id, overlap_count), ...]
    test_to_ref = defaultdict(list)  # test_id -> [(ref_id, overlap_count), ...]
    
    for (ref_id, test_id), count in overlap_matrix.items():
        ref_to_test[ref_id].append((test_id, count))
        test_to_ref[test_id].append((ref_id, count))
    
    matched = []
    split = []
    missing = []
    extra = []
    
    # Analyze reference instances
    for ref_id, ref_stat in ref_stats.items():
        ref_count = ref_stat['count']
        matches = ref_to_test.get(ref_id, [])
        
        if not matches:
            missing.append(ref_id)
            continue
        
        # Sort by overlap count
        matches.sort(key=lambda x: -x[1])
        
        # Check if primary match is strong enough
        primary_test_id, primary_overlap = matches[0]
        primary_ratio = primary_overlap / ref_count
        
        if primary_ratio >= min_overlap_ratio:
            # Check for splits (multiple significant matches)
            significant_matches = [(tid, cnt) for tid, cnt in matches 
                                   if cnt / ref_count >= 0.1]  # 10% threshold for split detection
            
            if len(significant_matches) > 1:
                split.append((ref_id, [tid for tid, _ in significant_matches]))
            else:
                matched.append((ref_id, primary_test_id, primary_ratio))
        else:
            missing.append(ref_id)
    
    # Find extra test instances (not matched to any ref)
    matched_test_ids = set(tid for _, tid, _ in matched)
    matched_test_ids.update(tid for _, tids in split for tid in tids)
    
    for test_id in test_stats.keys():
        if test_id not in matched_test_ids:
            # Check if it has any overlap with ref
            overlaps = test_to_ref.get(test_id, [])
            total_overlap = sum(cnt for _, cnt in overlaps)
            test_count = test_stats[test_id]['count']
            if total_overlap / test_count < min_overlap_ratio:
                extra.append(test_id)
    
    return {
        'matched': matched,
        'split': split,
        'missing': missing,
        'extra': extra,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two segmentations")
    parser.add_argument("--reference", type=str, required=True,
                        help="Reference LAZ file (e.g., no-tile segmentation)")
    parser.add_argument("--test", type=str, required=True,
                        help="Test LAZ file (e.g., tiled segmentation)")
    parser.add_argument("--min_overlap", type=float, default=0.3,
                        help="Minimum overlap ratio for matching (default 0.3)")
    
    args = parser.parse_args()
    
    # Load both segmentations
    ref_points, ref_instances = load_segmentation(args.reference)
    test_points, test_instances = load_segmentation(args.test)
    
    print(f"\nReference: {len(np.unique(ref_instances[ref_instances > 0]))} instances")
    print(f"Test: {len(np.unique(test_instances[test_instances > 0]))} instances")
    
    # Match instances
    overlap_matrix, ref_stats, test_stats = match_instances(
        ref_points, ref_instances, test_points, test_instances
    )
    
    # Analyze matches
    results = analyze_matches(overlap_matrix, ref_stats, test_stats, args.min_overlap)
    
    # Print results
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n1:1 MATCHED INSTANCES: {len(results['matched'])}")
    print(f"   (Reference tree matches single test tree)")
    
    print(f"\nSPLIT INSTANCES: {len(results['split'])}")
    print(f"   (Reference tree split into multiple test trees)")
    if results['split']:
        print("   Examples:")
        for ref_id, test_ids in results['split'][:10]:
            ref_count = ref_stats[ref_id]['count']
            test_counts = [test_stats[tid]['count'] for tid in test_ids]
            print(f"     Ref {ref_id} ({ref_count} pts) -> Test {test_ids} ({test_counts} pts)")
    
    print(f"\nMISSING INSTANCES: {len(results['missing'])}")
    print(f"   (Reference trees not found in test)")
    if results['missing']:
        print("   Examples:")
        for ref_id in results['missing'][:10]:
            ref_count = ref_stats[ref_id]['count']
            cx, cy, cz = ref_stats[ref_id]['centroid']
            print(f"     Ref {ref_id}: {ref_count} pts at ({cx:.1f}, {cy:.1f})")
    
    print(f"\nEXTRA INSTANCES: {len(results['extra'])}")
    print(f"   (Test trees not in reference)")
    if results['extra']:
        print("   Examples:")
        for test_id in results['extra'][:10]:
            test_count = test_stats[test_id]['count']
            cx, cy, cz = test_stats[test_id]['centroid']
            print(f"     Test {test_id}: {test_count} pts at ({cx:.1f}, {cy:.1f})")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Reference instances: {len(ref_stats)}")
    print(f"Test instances: {len(test_stats)}")
    print(f"1:1 matches: {len(results['matched'])} ({100*len(results['matched'])/len(ref_stats):.1f}% of ref)")
    print(f"Splits: {len(results['split'])} ref instances -> {sum(len(t) for _, t in results['split'])} test instances")
    print(f"Missing from test: {len(results['missing'])}")
    print(f"Extra in test: {len(results['extra'])}")
    
    # Instance size analysis
    print("\n" + "=" * 80)
    print("SIZE ANALYSIS OF DIFFERENCES")
    print("=" * 80)
    
    if results['split']:
        split_ref_sizes = [ref_stats[rid]['count'] for rid, _ in results['split']]
        print(f"\nSplit instances (ref size): min={min(split_ref_sizes)}, max={max(split_ref_sizes)}, median={np.median(split_ref_sizes):.0f}")
    
    if results['missing']:
        missing_sizes = [ref_stats[rid]['count'] for rid in results['missing']]
        print(f"Missing instances (ref size): min={min(missing_sizes)}, max={max(missing_sizes)}, median={np.median(missing_sizes):.0f}")
    
    if results['extra']:
        extra_sizes = [test_stats[tid]['count'] for tid in results['extra']]
        print(f"Extra instances (test size): min={min(extra_sizes)}, max={max(extra_sizes)}, median={np.median(extra_sizes):.0f}")


if __name__ == "__main__":
    main()

