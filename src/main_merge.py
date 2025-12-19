#!/usr/bin/env python3
"""
Main merge script: Merge segmented tiles with instance matching.

This script wraps the merge_tiles.py functionality to provide a clean interface
for the pipeline orchestrator.

Pipeline:
1. Load and filter (centroid-based buffer zone filtering)
2. Assign global IDs
3. Cross-tile instance matching
4. Merge and deduplicate
5. Small volume merging
6. Optionally retile to original files

Usage:
    python main_merge.py --segmented_folder /path/to/segmented_remapped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Import parameters
try:
    from parameters import MERGE_PARAMS
except ImportError:
    MERGE_PARAMS = {
        'buffer': 10.0,
        'overlap_threshold': 0.3,
        'max_centroid_distance': 3.0,
        'correspondence_tolerance': 0.05,
        'max_volume_for_merge': 4.0,
        'workers': 4,
    }

# Import the core merge function
try:
    from merge_tiles import merge_tiles as core_merge_tiles
except ImportError:
    core_merge_tiles = None


def run_merge(
    segmented_dir: Path,
    output_merged: Optional[Path] = None,
    output_tiles_dir: Optional[Path] = None,
    original_tiles_dir: Optional[Path] = None,
    buffer: float = 10.0,
    overlap_threshold: float = 0.3,
    max_centroid_distance: float = 3.0,
    correspondence_tolerance: float = 0.05,
    max_volume_for_merge: float = 4.0,
    num_threads: int = 4,
    enable_matching: bool = True,
    require_overlap: bool = True,
    enable_volume_merge: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Run the tile merge pipeline.
    
    Args:
        segmented_dir: Directory containing segmented LAZ tiles
        output_merged: Output path for merged LAZ file (auto-derived if None)
        output_tiles_dir: Output directory for retiled files (optional)
        original_tiles_dir: Directory with original tile files for retiling (optional)
        buffer: Buffer zone distance in meters
        overlap_threshold: Overlap ratio threshold for instance matching
        max_centroid_distance: Max distance between centroids to merge instances
        correspondence_tolerance: Max distance for point correspondence
        max_volume_for_merge: Max convex hull volume for small instance merging
        num_threads: Number of workers for parallel processing
        enable_matching: Enable cross-tile instance matching
        require_overlap: Require overlap ratio check (vs centroid distance only)
        enable_volume_merge: Enable small volume instance merging
        verbose: Print detailed merge decisions
    
    Returns:
        Path to merged output file
    """
    print("=" * 60)
    print("3DTrees Merge Pipeline")
    print("=" * 60)
    
    # Validate input
    if not segmented_dir.exists():
        raise ValueError(f"Segmented directory not found: {segmented_dir}")
    
    # Auto-derive output path if not provided
    if output_merged is None:
        output_merged = segmented_dir.parent / "merged.laz"
    
    print(f"Input: {segmented_dir}")
    print(f"Output: {output_merged}")
    print(f"Buffer: {buffer}m")
    print(f"Instance matching: {'ENABLED' if enable_matching else 'DISABLED'}")
    if enable_matching:
        print(f"  Overlap threshold: {overlap_threshold}")
        print(f"  Max centroid distance: {max_centroid_distance}m")
    print(f"Volume merge: {'ENABLED' if enable_volume_merge else 'DISABLED'}")
    if enable_volume_merge:
        print(f"  Max volume: {max_volume_for_merge} m³")
    print(f"Workers: {num_threads}")
    print()
    
    # Check if core merge function is available
    if core_merge_tiles is None:
        raise ImportError(
            "merge_tiles.py not found. Make sure it's in the same directory."
        )
    
    # Run the core merge function
    core_merge_tiles(
        input_dir=segmented_dir,
        original_tiles_dir=original_tiles_dir,
        output_merged=output_merged,
        output_tiles_dir=output_tiles_dir,
        buffer=buffer,
        overlap_threshold=overlap_threshold,
        max_centroid_distance=max_centroid_distance,
        correspondence_tolerance=correspondence_tolerance,
        max_volume_for_merge=max_volume_for_merge,
        num_threads=num_threads,
        enable_matching=enable_matching,
        require_overlap=require_overlap,
        enable_volume_merge=enable_volume_merge,
        verbose=verbose,
    )
    
    return output_merged


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Merge Pipeline - Merge segmented tiles with instance matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--segmented_folder", "-i",
        type=Path,
        required=True,
        help="Directory containing segmented LAZ tiles"
    )
    
    parser.add_argument(
        "--output_merged", "-o",
        type=Path,
        default=None,
        help="Output path for merged LAZ file (auto-derived if not specified)"
    )
    
    parser.add_argument(
        "--output_tiles_dir",
        type=Path,
        default=None,
        help="Output directory for retiled files (optional)"
    )
    
    parser.add_argument(
        "--original_tiles_dir",
        type=Path,
        default=None,
        help="Directory with original tile files for retiling (optional)"
    )
    
    parser.add_argument(
        "--buffer",
        type=float,
        default=MERGE_PARAMS.get('buffer', 10.0),
        help=f"Buffer zone distance in meters (default: {MERGE_PARAMS.get('buffer', 10.0)})"
    )
    
    parser.add_argument(
        "--overlap_threshold",
        type=float,
        default=MERGE_PARAMS.get('overlap_threshold', 0.3),
        help=f"Overlap ratio threshold (default: {MERGE_PARAMS.get('overlap_threshold', 0.3)})"
    )
    
    parser.add_argument(
        "--max_centroid_distance",
        type=float,
        default=MERGE_PARAMS.get('max_centroid_distance', 3.0),
        help=f"Max centroid distance (default: {MERGE_PARAMS.get('max_centroid_distance', 3.0)})"
    )
    
    parser.add_argument(
        "--correspondence_tolerance",
        type=float,
        default=MERGE_PARAMS.get('correspondence_tolerance', 0.05),
        help=f"Point correspondence tolerance (default: {MERGE_PARAMS.get('correspondence_tolerance', 0.05)})"
    )
    
    parser.add_argument(
        "--max_volume_for_merge",
        type=float,
        default=MERGE_PARAMS.get('max_volume_for_merge', 4.0),
        help=f"Max volume for small instance merge (default: {MERGE_PARAMS.get('max_volume_for_merge', 4.0)})"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=MERGE_PARAMS.get('workers', 4),
        help=f"Number of workers (default: {MERGE_PARAMS.get('workers', 4)})"
    )
    
    parser.add_argument(
        "--disable_matching",
        action="store_true",
        help="Disable cross-tile instance matching"
    )
    
    parser.add_argument(
        "--disable_overlap_check",
        action="store_true",
        help="Disable overlap ratio check (centroid distance only)"
    )
    
    parser.add_argument(
        "--disable_volume_merge",
        action="store_true",
        help="Disable small volume instance merging"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed merge decisions"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        output_file = run_merge(
            segmented_dir=args.segmented_folder,
            output_merged=args.output_merged,
            output_tiles_dir=args.output_tiles_dir,
            original_tiles_dir=args.original_tiles_dir,
            buffer=args.buffer,
            overlap_threshold=args.overlap_threshold,
            max_centroid_distance=args.max_centroid_distance,
            correspondence_tolerance=args.correspondence_tolerance,
            max_volume_for_merge=args.max_volume_for_merge,
            num_threads=args.workers,
            enable_matching=not args.disable_matching,
            require_overlap=not args.disable_overlap_check,
            enable_volume_merge=not args.disable_volume_merge,
            verbose=args.verbose,
        )
        print(f"\nMerged output: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

