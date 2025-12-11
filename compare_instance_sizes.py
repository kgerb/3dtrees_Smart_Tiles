#!/usr/bin/env python3
"""
Compare instance size distributions between two segmentations.
"""

import numpy as np
import laspy
import matplotlib.pyplot as plt
import argparse


def get_instance_sizes(laz_path):
    """Load LAZ and return list of instance sizes."""
    las = laspy.read(laz_path, laz_backend=laspy.LazBackend.LazrsParallel)
    instances = np.array(las.PredInstance)
    
    unique, counts = np.unique(instances, return_counts=True)
    tree_sizes = [c for inst_id, c in zip(unique, counts) if inst_id > 0]
    
    return tree_sizes, len(las.x), np.sum(instances == 0)


def main():
    parser = argparse.ArgumentParser(description="Compare instance size distributions")
    parser.add_argument("--ref", type=str, required=True, help="Reference LAZ file")
    parser.add_argument("--test", type=str, required=True, help="Test LAZ file")
    parser.add_argument("--ref_name", type=str, default="Reference", help="Reference name")
    parser.add_argument("--test_name", type=str, default="Test", help="Test name")
    parser.add_argument("--output", type=str, default="instance_size_comparison.png", help="Output plot")
    
    args = parser.parse_args()
    
    print(f"Loading {args.ref_name}...")
    ref_sizes, ref_total_pts, ref_bg = get_instance_sizes(args.ref)
    
    print(f"Loading {args.test_name}...")
    test_sizes, test_total_pts, test_bg = get_instance_sizes(args.test)
    
    print(f"\n{args.ref_name}:")
    print(f"  Total points: {ref_total_pts:,}")
    print(f"  Background points: {ref_bg:,} ({100*ref_bg/ref_total_pts:.1f}%)")
    print(f"  Tree instances: {len(ref_sizes)}")
    print(f"  Min size: {min(ref_sizes)}")
    print(f"  Max size: {max(ref_sizes):,}")
    print(f"  Median size: {np.median(ref_sizes):.0f}")
    print(f"  Mean size: {np.mean(ref_sizes):.0f}")
    
    print(f"\n{args.test_name}:")
    print(f"  Total points: {test_total_pts:,}")
    print(f"  Background points: {test_bg:,} ({100*test_bg/test_total_pts:.1f}%)")
    print(f"  Tree instances: {len(test_sizes)}")
    print(f"  Min size: {min(test_sizes)}")
    print(f"  Max size: {max(test_sizes):,}")
    print(f"  Median size: {np.median(test_sizes):.0f}")
    print(f"  Mean size: {np.mean(test_sizes):.0f}")
    
    # Size thresholds
    thresholds = [10, 50, 100, 300, 500, 1000, 5000, 10000]
    print(f"\n{'Size threshold':<20} {args.ref_name:<20} {args.test_name:<20}")
    print("-" * 60)
    for thresh in thresholds:
        ref_count = sum(1 for s in ref_sizes if s < thresh)
        test_count = sum(1 for s in test_sizes if s < thresh)
        print(f"< {thresh:>5} points{'':<10} {ref_count:>6} ({100*ref_count/len(ref_sizes):>5.1f}%){'':<8} {test_count:>6} ({100*test_count/len(test_sizes):>5.1f}%)")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram - linear scale
    ax = axes[0, 0]
    ax.hist(ref_sizes, bins=50, alpha=0.7, label=args.ref_name, edgecolor='black')
    ax.hist(test_sizes, bins=50, alpha=0.7, label=args.test_name, edgecolor='black')
    ax.set_xlabel('Instance Size (points)')
    ax.set_ylabel('Count')
    ax.set_title('Instance Size Distribution (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Histogram - log scale
    ax = axes[0, 1]
    ax.hist(ref_sizes, bins=50, alpha=0.7, label=args.ref_name, edgecolor='black')
    ax.hist(test_sizes, bins=50, alpha=0.7, label=args.test_name, edgecolor='black')
    ax.set_xlabel('Instance Size (points)')
    ax.set_ylabel('Count')
    ax.set_title('Instance Size Distribution (Log Scale)')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # CDF
    ax = axes[1, 0]
    ref_sorted = np.sort(ref_sizes)
    test_sorted = np.sort(test_sizes)
    ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    test_cdf = np.arange(1, len(test_sorted) + 1) / len(test_sorted)
    ax.plot(ref_sorted, ref_cdf, label=args.ref_name, linewidth=2)
    ax.plot(test_sorted, test_cdf, label=args.test_name, linewidth=2)
    ax.set_xlabel('Instance Size (points)')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Box plot
    ax = axes[1, 1]
    ax.boxplot([ref_sizes, test_sizes], labels=[args.ref_name, args.test_name])
    ax.set_ylabel('Instance Size (points)')
    ax.set_title('Box Plot Comparison')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nSaved plot: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()

