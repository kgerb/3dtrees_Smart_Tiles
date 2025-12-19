#!/usr/bin/env python3
"""
Main tiling script: XYZ reduction, COPC conversion, index building, and tiling.

This script handles the first phase of the 3DTrees pipeline:
1. Convert input LAZ files to XYZ-only COPC (reduced file size)
2. Build spatial index (tindex)
3. Calculate tile bounds
4. Create overlapping tiles

Usage:
    python main_tile.py --input_dir /path/to/input --output_dir /path/to/output
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plot_tiles_and_copc

# Import parameters
try:
    from parameters import TILE_PARAMS
except ImportError:
    TILE_PARAMS = {
        'tile_length': 100,
        'tile_buffer': 5,
        'threads': 5,
        'workers': 4,
        'grid_offset': 1.0,
    }


def get_pdal_path() -> str:
    """Get the path to pdal executable."""
    import shutil
    # Use shutil.which to find pdal in PATH
    pdal_path = shutil.which("pdal")
    return pdal_path if pdal_path else "pdal"


def get_pdal_wrench_path() -> str:
    """Get the path to pdal_wrench executable."""
    import shutil
    # Use shutil.which to find pdal_wrench in PATH
    wrench_path = shutil.which("pdal_wrench")
    return wrench_path if wrench_path else "pdal_wrench"


def create_minimal_laz(input_file: Path, output_file: Path, chunk_size: int = 1_000_000) -> bool:
    """
    Create a minimal LAZ file with only XYZ coordinates using laspy chunked reading.
    
    This creates the smallest possible LAZ file by:
    1. Using point format 0 (minimal format)
    2. Keeping only X, Y, Z data
    3. Zeroing out all other mandatory fields
    4. Processing in chunks for memory efficiency with large files
    
    Args:
        input_file: Path to input LAZ file
        output_file: Path to output minimal LAZ file
        chunk_size: Number of points to process per chunk (default: 1M points)
    
    Returns:
        True if successful, False otherwise
    """
    import laspy
    
    try:
        with laspy.open(str(input_file)) as reader:
            # Create output header with point format 0 (minimal: 20 bytes/point)
            out_header = laspy.LasHeader(point_format=0, version="1.2")
            out_header.offsets = reader.header.offsets
            out_header.scales = reader.header.scales
            
            # Determine best available LAZ backend
            laz_backend = None
            for backend in [laspy.LazBackend.LazrsParallel, laspy.LazBackend.Lazrs, laspy.LazBackend.Laszip]:
                try:
                    laz_backend = backend
                    break
                except Exception:
                    continue
            
            # Open writer with compression (auto-detect backend if needed)
            writer_kwargs = {'mode': 'w', 'header': out_header}
            if laz_backend:
                writer_kwargs['laz_backend'] = laz_backend
            
            with laspy.open(str(output_file), **writer_kwargs) as writer:
                # Process in chunks for memory efficiency
                for chunk in reader.chunk_iterator(chunk_size):
                    # Create minimal point record with only XYZ
                    out_points = laspy.ScaleAwarePointRecord.zeros(len(chunk), header=out_header)
                    out_points.x = chunk.x
                    out_points.y = chunk.y
                    out_points.z = chunk.z
                    
                    writer.write_points(out_points)
        
        return True
    except Exception as e:
        print(f"    Error creating minimal LAZ: {e}")
        return False


def convert_single_file(args: Tuple[Path, Path, Path, bool]) -> Tuple[str, bool, str]:
    """
    Convert a single LAZ file to COPC.
    
    Two modes:
    1. With dimension reduction (default): 
       - Use laspy to create minimal LAZ (XYZ only, format 0)
       - Convert to COPC using pdal_wrench
       - Achieves ~37% size reduction
    2. Without dimension reduction (skip_dimension_reduction=True):
       - Direct conversion to COPC preserving all dimensions
    
    Args:
        args: Tuple of (input_file, output_file, log_dir, skip_dimension_reduction)
    
    Returns:
        Tuple of (filename, success, message)
    """
    input_file, output_file, log_dir, skip_dimension_reduction = args
    
    try:
        log_file = log_dir / f"{input_file.stem}_convert.log"
        
        if skip_dimension_reduction:
            # Use PDAL pipeline to preserve ALL dimensions including extra dimensions
            import json
            
            pipeline_file = log_dir / f"{input_file.stem}_copc_pipeline.json"
            pipeline = {
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": str(input_file)
                    },
                    {
                        "type": "writers.copc",
                        "filename": str(output_file),
                        "forward": "all",  # Forward all dimensions including extra
                        "extra_dims": "all"  # Explicitly preserve extra dimensions
                    }
                ]
            }
            
            with open(pipeline_file, 'w') as f:
                json.dump(pipeline, f, indent=2)
            
            pdal_cmd = get_pdal_path()
            result = subprocess.run(
                [pdal_cmd, "pipeline", str(pipeline_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Clean up pipeline file on success
            if result.returncode == 0 and pipeline_file.exists():
                pipeline_file.unlink()
            
            # Save log on error
            if result.returncode != 0:
                with open(log_file, 'w') as f:
                    f.write(f"Command: pdal pipeline {pipeline_file}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Stderr:\n{result.stderr}\n")
        else:
            # Two-step process with dimension reduction
            # Step 1: Create minimal LAZ with laspy (XYZ only)
            temp_laz = log_dir / f"{input_file.stem}_minimal.laz"
            
            if not create_minimal_laz(input_file, temp_laz):
                return (input_file.name, False, "Failed to create minimal LAZ")
            
            # Step 2: Convert minimal LAZ to COPC using pdal_wrench (fast)
            result = subprocess.run(
                [
                    pdal_wrench, "translate",
                    f"--input={temp_laz}",
                    f"--output={output_file}"
                ],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Save log on error
            if result.returncode != 0:
                with open(log_file, 'w') as f:
                    f.write(f"Command: pdal_wrench translate --input={temp_laz} --output={output_file}\n")
                    f.write(f"Return code: {result.returncode}\n")
                    f.write(f"Stderr:\n{result.stderr}\n")
            
            # Clean up temp LAZ
            if temp_laz.exists():
                temp_laz.unlink()
        
        if result.returncode != 0:
            return (input_file.name, False, result.stderr[:200])
        
        # Verify output exists and has content
        if not output_file.exists() or output_file.stat().st_size == 0:
            return (input_file.name, False, "Output file empty or not created")
        
        return (input_file.name, True, "Success")
        
    except Exception as e:
        return (input_file.name, False, str(e))


def convert_to_xyz_copc(
    input_dir: Path,
    output_dir: Path,
    num_workers: int,
    log_dir: Optional[Path] = None,
    skip_dimension_reduction: bool = False
) -> List[Path]:
    """
    Convert LAZ files to COPC format using pdal_wrench.
    
    Uses pdal_wrench translate for efficient COPC conversion with spatial indexing.
    Parallelizes across num_workers.
    
    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Directory for output COPC files
        num_workers: Number of parallel workers
        log_dir: Directory for log files (optional)
        skip_dimension_reduction: Skip XYZ-only reduction and keep all dimensions
    
    Returns:
        List of created COPC file paths
    """
    print("=" * 60)
    if skip_dimension_reduction:
        print("Step 1: Converting LAZ to COPC (preserving all dimensions)")
    else:
        print("Step 1: Converting LAZ to COPC (XYZ-only)")
    print("=" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    if log_dir is None:
        log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all LAZ files
    input_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
    # Exclude already converted COPC files
    input_files = [f for f in input_files if not f.name.endswith('.copc.laz')]
    
    if not input_files:
        print(f"  No LAZ/LAS files found in {input_dir}")
        # Check if COPC files already exist
        existing_copc = list(output_dir.glob("*.copc.laz"))
        if existing_copc:
            print(f"  Found {len(existing_copc)} existing COPC files")
            return existing_copc
        return []
    
    print(f"  Found {len(input_files)} files to convert")
    print(f"  Using {num_workers} parallel workers")
    print(f"  Output: {output_dir}")
    print()
    
    # Prepare conversion tasks
    tasks = []
    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}.copc.laz"
        
        # Skip if already converted
        if output_file.exists() and output_file.stat().st_size > 0:
            print(f"  Skipping {input_file.name} (already converted)")
            continue
        
        tasks.append((input_file, output_file, log_dir, skip_dimension_reduction))
    
    if not tasks:
        print("  All files already converted")
        return list(output_dir.glob("*.copc.laz"))
    
    # Process files in parallel
    successful = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_single_file, task): task[0] for task in tasks}
        
        for future in as_completed(futures):
            filename, success, message = future.result()
            if success:
                successful += 1
                print(f"  ✓ Converted: {filename}")
            else:
                failed += 1
                print(f"  ✗ Failed: {filename} - {message}")
    
    print()
    print(f"  Conversion complete: {successful} successful, {failed} failed")
    
    return list(output_dir.glob("*.copc.laz"))


def build_tindex(copc_dir: Path, output_gpkg: Path) -> Path:
    """
    Build spatial index (tindex) from COPC files.
    
    Uses pdal tindex to create a GeoPackage containing the spatial
    extents of all COPC files for efficient spatial queries.
    
    Args:
        copc_dir: Directory containing COPC files
        output_gpkg: Output path for tindex GeoPackage
    
    Returns:
        Path to created tindex file
    """
    print()
    print("=" * 60)
    print("Step 2: Building spatial index (tindex)")
    print("=" * 60)
    
    # Check if tindex already exists
    if output_gpkg.exists():
        print(f"  Using existing tindex: {output_gpkg}")
        return output_gpkg
    
    # Create output directory
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    
    # Find COPC files
    copc_files = list(copc_dir.glob("*.copc.laz"))
    if not copc_files:
        raise ValueError(f"No COPC files found in {copc_dir}")
    
    print(f"  Found {len(copc_files)} COPC files")
    print(f"  Output: {output_gpkg}")
    
    # Create file list for pdal tindex
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for copc_file in sorted(copc_files):
            f.write(f"{copc_file}\n")
        file_list_path = f.name
    
    try:
        # Build tindex using pdal tindex
        pdal_cmd = get_pdal_path()
        cmd = [
            pdal_cmd, "tindex", "create",
            f"--tindex={output_gpkg}",
            "--stdin",
            "--tindex_name=Location",
            "--ogrdriver=GPKG",
            "--fast_boundary",
            "--write_absolute_path"
        ]
        
        with open(file_list_path, 'r') as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
                check=False
            )
        
        if result.returncode != 0:
            raise RuntimeError(f"pdal tindex failed: {result.stderr}")
        
        print(f"  ✓ Tindex created: {output_gpkg}")
        
    finally:
        # Clean up temp file
        os.unlink(file_list_path)
    
    return output_gpkg


def calculate_tile_bounds(
    tindex_file: Path,
    tile_length: float,
    tile_buffer: float,
    grid_offset: float,
    output_dir: Path
) -> Tuple[Path, Path, dict]:
    """
    Calculate tile bounds from tindex.
    
    Uses prepare_tile_jobs.py to compute tile grid based on the
    spatial extent of input files.
    
    Args:
        tindex_file: Path to tindex GeoPackage
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        output_dir: Directory for output files
    
    Returns:
        Tuple of (tile_jobs_file, tile_bounds_json, env_dict)
    """
    print()
    print("=" * 60)
    print("Step 3: Calculating tile bounds")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    prepare_jobs_script = script_dir / "prepare_tile_jobs.py"
    
    jobs_file = output_dir / f"tile_jobs_{int(tile_length)}m.txt"
    bounds_json = output_dir / "tile_bounds_tindex.json"
    
    cmd = [
        sys.executable,
        str(prepare_jobs_script),
        str(tindex_file),
        f"--tile-length={tile_length}",
        f"--tile-buffer={tile_buffer}",
        f"--jobs-out={jobs_file}",
        f"--bounds-out={bounds_json}",
        f"--grid-offset={grid_offset}"
    ]
    
    print(f"  Tile length: {tile_length}m")
    print(f"  Tile buffer: {tile_buffer}m")
    print(f"  Grid offset: {grid_offset}m")
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"prepare_tile_jobs.py failed: {result.stderr}")
    
    # Parse environment variables from output
    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"')
    
    tile_count = env.get('tile_count', 'unknown')
    print(f"  ✓ Calculated {tile_count} tiles")
    print(f"  Jobs file: {jobs_file}")
    print(f"  Bounds file: {bounds_json}")
    
    return jobs_file, bounds_json, env


def get_copc_files_from_tindex(tindex_file: Path) -> List[str]:
    """Get list of COPC files from tindex database."""
    import sqlite3
    
    conn = sqlite3.connect(str(tindex_file))
    cursor = conn.cursor()
    
    # Get table name from gpkg_contents
    cursor.execute('SELECT table_name FROM gpkg_contents WHERE data_type = "features" LIMIT 1')
    result = cursor.fetchone()
    
    if not result:
        conn.close()
        return []
    
    table_name = result[0]
    cursor.execute(f'SELECT DISTINCT Location FROM "{table_name}"')
    files = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return files


def process_single_tile(args: Tuple[str, str, str, List[str], Path, Path, int]) -> Tuple[str, bool, str]:
    """
    Process a single tile: extract points from COPC files within bounds.
    
    Args:
        args: Tuple of (label, proj_bounds, geo_bounds, copc_files, tiles_dir, log_dir, threads)
    
    Returns:
        Tuple of (label, success, message)
    """
    label, proj_bounds, geo_bounds, copc_files, tiles_dir, log_dir, threads = args
    
    final_tile = tiles_dir / f"{label}.copc.laz"
    
    # Skip if already exists
    if final_tile.exists() and final_tile.stat().st_size > 0:
        return (label, True, "Already exists")
    
    try:
        # Create temporary directory for parts
        tile_dir = tiles_dir / label
        tile_dir.mkdir(exist_ok=True)
        
        parts_created = 0
        
        # Extract from each COPC file with bounds filtering
        for part_num, copc_file in enumerate(copc_files):
            if not os.path.isfile(copc_file):
                continue
            
            part_file = tile_dir / f"part_{part_num}.copc.laz"
            
            # Create pipeline for extraction
            pipeline = {
                "pipeline": [
                    {
                        "type": "readers.copc",
                        "filename": copc_file,
                        "bounds": proj_bounds
                    },
                    {
                        "type": "writers.copc",
                        "filename": str(part_file),
                        "threads": threads,
                        "forward": "all",
                        "extra_dims": "all"  # Preserve extra dimensions like PredInstance
                    }
                ]
            }
            
            pipeline_file = log_dir / f"{label}_part{part_num}_pipeline.json"
            with open(pipeline_file, 'w') as f:
                json.dump(pipeline, f)
            
            pdal_cmd = get_pdal_path()
            result = subprocess.run(
                [pdal_cmd, "pipeline", str(pipeline_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            # Clean up pipeline file
            if pipeline_file.exists():
                pipeline_file.unlink()
            
            if result.returncode == 0 and part_file.exists() and part_file.stat().st_size > 0:
                parts_created += 1
            elif part_file.exists():
                part_file.unlink()
        
        if parts_created == 0:
            if tile_dir.exists():
                tile_dir.rmdir()
            return (label, True, "No data in bounds")
        
        # Merge parts into final tile
        parts = list(tile_dir.glob("part_*.copc.laz"))
        
        if len(parts) == 1:
            # Just rename single part
            parts[0].rename(final_tile)
        else:
            # Create merge pipeline
            readers = [{"type": "readers.copc", "filename": str(p)} for p in parts]
            merge_pipeline = {
                "pipeline": readers + [
                    {"type": "filters.merge"},
                    {
                        "type": "writers.copc",
                        "filename": str(final_tile),
                        "threads": threads,
                        "forward": "all",
                        "extra_dims": "all",  # Preserve extra dimensions like PredInstance
                        "scale_x": 0.01,
                        "scale_y": 0.01,
                        "scale_z": 0.01
                    }
                ]
            }
            
            merge_pipeline_file = log_dir / f"{label}_merge_pipeline.json"
            with open(merge_pipeline_file, 'w') as f:
                json.dump(merge_pipeline, f)
            
            pdal_cmd = get_pdal_path()
            result = subprocess.run(
                [pdal_cmd, "pipeline", str(merge_pipeline_file)],
                capture_output=True,
                text=True,
                check=False
            )
            
            if merge_pipeline_file.exists():
                merge_pipeline_file.unlink()
            
            if result.returncode != 0:
                return (label, False, f"Merge failed: {result.stderr[:200]}")
        
        # Clean up parts
        for part in parts:
            if part.exists():
                part.unlink()
        if tile_dir.exists() and not any(tile_dir.iterdir()):
            tile_dir.rmdir()
        
        return (label, True, f"{parts_created} parts merged")
        
    except Exception as e:
        return (label, False, str(e))


def create_tiles(
    tindex_file: Path,
    tile_jobs_file: Path,
    tiles_dir: Path,
    log_dir: Path,
    threads: int = 5,
    max_parallel: int = 5
) -> List[Path]:
    """
    Create overlapping tiles from COPC files.
    
    Reads tile jobs from the job file and creates tiles by extracting
    points from COPC files using spatial bounds filtering.
    
    Args:
        tindex_file: Path to tindex GeoPackage
        tile_jobs_file: Path to tile jobs file
        tiles_dir: Output directory for tiles
        log_dir: Directory for log files
        threads: Threads per COPC writer
        max_parallel: Maximum parallel tile processes
    
    Returns:
        List of created tile paths
    """
    print()
    print("=" * 60)
    print("Step 4: Creating tiles")
    print("=" * 60)
    
    # Create directories
    tiles_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get COPC files from tindex
    copc_files = get_copc_files_from_tindex(tindex_file)
    if not copc_files:
        raise ValueError("No COPC files found in tindex")
    
    print(f"  Source files: {len(copc_files)}")
    print(f"  Output: {tiles_dir}")
    print(f"  Parallel tiles: {max_parallel}")
    
    # Read tile jobs
    jobs = []
    with open(tile_jobs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            if len(parts) >= 2:
                label = parts[0]
                proj_bounds = parts[1]
                geo_bounds = parts[2] if len(parts) > 2 else ""
                jobs.append((label, proj_bounds, geo_bounds, copc_files, tiles_dir, log_dir, threads))
    
    print(f"  Tiles to create: {len(jobs)}")
    print()
    
    # Process tiles in parallel
    successful = 0
    failed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(process_single_tile, job): job[0] for job in jobs}
        
        for future in as_completed(futures):
            label, success, message = future.result()
            if success:
                if "Already exists" in message or "No data" in message:
                    skipped += 1
                    print(f"  - {label}: {message}")
                else:
                    successful += 1
                    print(f"  ✓ {label}: {message}")
            else:
                failed += 1
                print(f"  ✗ {label}: {message}")
    
    print()
    print(f"  Tiling complete: {successful} created, {skipped} skipped, {failed} failed")
    
    return list(tiles_dir.glob("*.copc.laz"))


def run_tiling_pipeline(
    input_dir: Path,
    output_dir: Path,
    tile_length: float = 100,
    tile_buffer: float = 5,
    grid_offset: float = 1.0,
    num_workers: int = 4,
    threads: int = 5,
    max_tile_procs: int = 5,
    skip_dimension_reduction: bool = False
) -> Path:
    """
    Run the complete tiling pipeline.
    
    Steps:
    1. Convert LAZ to COPC (with or without dimension reduction)
    2. Build spatial index
    3. Calculate tile bounds
    4. Create overlapping tiles
    
    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Base output directory
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        num_workers: Number of parallel conversion workers
        threads: Threads per COPC writer
        max_tile_procs: Maximum parallel tile processes
        skip_dimension_reduction: Skip XYZ-only reduction and keep all dimensions
    
    Returns:
        Path to tiles directory
    """
    print("=" * 60)
    print("3DTrees Tiling Pipeline")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {tile_length}m with {tile_buffer}m buffer")
    print()
    
    # Define paths
    copc_dir = output_dir / ("copc_full" if skip_dimension_reduction else "copc_xyz")
    tiles_dir = output_dir / f"tiles_{int(tile_length)}m"
    log_dir = output_dir / "logs"
    tindex_file = output_dir / f"tindex_{int(tile_length)}m.gpkg"
    
    # Step 1: Convert to COPC
    copc_files = convert_to_xyz_copc(input_dir, copc_dir, num_workers, log_dir, skip_dimension_reduction)
    
    if not copc_files:
        raise ValueError("No COPC files created or found")
    
    # Step 2: Build tindex
    tindex_file = build_tindex(copc_dir, tindex_file)
    
    # Step 3: Calculate tile bounds
    jobs_file, bounds_json, env = calculate_tile_bounds(
        tindex_file, tile_length, tile_buffer, grid_offset, output_dir
    )

    # Step 4: Plot the tiles
    plot_tiles_and_copc.plot_extents(tindex_file, bounds_json, output_dir/"overview_copc_tiles.png")
    
    # Step 5: Create tiles
    tile_files = create_tiles(
        tindex_file, jobs_file, tiles_dir, log_dir, threads, max_tile_procs
    )
    
    print()
    print("=" * 60)
    print("Tiling Pipeline Complete")
    print("=" * 60)
    print(f"  COPC files: {len(copc_files)}")
    print(f"  Tiles created: {len(tile_files)}")
    print(f"  Tiles directory: {tiles_dir}")
    
    return tiles_dir


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="3DTrees Tiling Pipeline - XYZ reduction, COPC conversion, and tiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=Path,
        required=True,
        help="Input directory containing LAZ files"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
        help="Output directory for all stages"
    )
    
    parser.add_argument(
        "--tile_length",
        type=float,
        default=TILE_PARAMS.get('tile_length', 100),
        help=f"Tile size in meters (default: {TILE_PARAMS.get('tile_length', 100)})"
    )
    
    parser.add_argument(
        "--tile_buffer",
        type=float,
        default=TILE_PARAMS.get('tile_buffer', 5),
        help=f"Buffer overlap in meters (default: {TILE_PARAMS.get('tile_buffer', 5)})"
    )
    
    parser.add_argument(
        "--grid_offset",
        type=float,
        default=TILE_PARAMS.get('grid_offset', 1.0),
        help=f"Grid offset in meters (default: {TILE_PARAMS.get('grid_offset', 1.0)})"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=TILE_PARAMS.get('workers', 4),
        help=f"Number of parallel workers (default: {TILE_PARAMS.get('workers', 4)})"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=TILE_PARAMS.get('threads', 5),
        help=f"Threads per COPC writer (default: {TILE_PARAMS.get('threads', 5)})"
    )
    
    parser.add_argument(
        "--max_tile_procs",
        type=int,
        default=5,
        help="Maximum parallel tile processes (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        tiles_dir = run_tiling_pipeline(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tile_length=args.tile_length,
            tile_buffer=args.tile_buffer,
            grid_offset=args.grid_offset,
            num_workers=args.num_workers,
            threads=args.threads,
            max_tile_procs=args.max_tile_procs
        )
        print(f"\nTiles ready for subsampling: {tiles_dir}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

