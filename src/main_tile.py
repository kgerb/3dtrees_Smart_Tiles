#!/usr/bin/env python3
"""
Main tiling script: COPC conversion, index building, and tiling.

This script handles the first phase of the 3DTrees pipeline:
1. Convert input LAZ files to COPC (with optional XYZ-only reduction using untwine --dims)
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
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plot_tiles_and_copc

from parameters import TILE_PARAMS



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


def get_untwine_path() -> str:
    """Get the path to untwine executable."""
    import shutil
    # Use shutil.which to find untwine in PATH
    untwine_path = shutil.which("untwine")
    return untwine_path if untwine_path else "untwine"


def _laspy_laz_backend():
    """Return the LAZ backend to use for laspy (Lazrs or LazrsParallel when available)."""
    try:
        import laspy
        if hasattr(laspy.LazBackend, "LazrsParallel"):
            return laspy.LazBackend.LazrsParallel
        if hasattr(laspy.LazBackend, "Lazrs"):
            return laspy.LazBackend.Lazrs
    except Exception:
        pass
    return None


def reduce_to_xyz_las(input_path: Path, output_path: Path) -> None:
    """
    Reduce a LAZ/LAS file to XYZ-only using laspy and write as uncompressed LAS.
    Used as a workaround when untwine --dims X,Y,Z triggers bugs (e.g. free(): invalid pointer).

    Root cause (upstream): In untwine < 1.4, when --dims X,Y,Z is used the internal
    "untwine bits" offset can be -1 (no such dimension). The code did an unguarded
    memcpy(dst + offset, ...), corrupting the heap and leading to free(): invalid pointer
    or segfault. Fixed in hobuinc/untwine#151 (Don't crash when there are no "untwine bits"),
    merged Jan 2024, included in untwine 1.4+. Conda-forge may still ship 1.3.x.

    Preserves header scale, offset, and CRS so coordinates are unchanged.
    """
    import numpy as np
    import laspy
    laz_backend = _laspy_laz_backend()
    kwargs = {}
    if input_path.suffix.lower() == ".laz" and laz_backend is not None:
        kwargs["laz_backend"] = laz_backend

    with laspy.open(str(input_path), **kwargs) as src:
        header = src.header
        # Point format 0 = only X, Y, Z
        new_header = laspy.LasHeader(point_format=0, version=header.version)
        new_header.offsets = np.array([header.x_offset, header.y_offset, header.z_offset])
        new_header.scales = np.array([header.x_scale, header.y_scale, header.z_scale])
        # Copy CRS if present
        try:
            crs = header.parse_crs()
            if crs is not None:
                new_header.add_crs(crs)
        except Exception:
            pass
        # Read all points (chunked to limit memory for huge files)
        x = np.empty(header.point_count, dtype=np.float64)
        y = np.empty(header.point_count, dtype=np.float64)
        z = np.empty(header.point_count, dtype=np.float64)
        done = 0
        for chunk in src.chunk_iterator(2_000_000):
            n = len(chunk.x)
            x[done : done + n] = chunk.x
            y[done : done + n] = chunk.y
            z[done : done + n] = chunk.z
            done += n
    out = laspy.LasData(new_header)
    out.x = x
    out.y = y
    out.z = z
    out.write(str(output_path))


def check_crs(laz_file: Path) -> Tuple[bool, str]:
    """
    Check if LAZ/LAS file has a CRS.
    
    Args:
        laz_file: Path to LAZ/LAS file

    Returns:
        Tuple of (has_crs, crs_description)
    """
    try:
        import laspy
        laz_backend = _laspy_laz_backend()
        kwargs = {}
        if laz_file.suffix.lower() == ".laz" and laz_backend is not None:
            kwargs["laz_backend"] = laz_backend

        # Read only the header to be fast
        with laspy.open(str(laz_file), **kwargs) as las:
            # Check if CRS exists
            has_crs = False
            try:
                crs = las.header.parse_crs()
                if crs is not None:
                    wkt = (crs.to_wkt() or "").strip()
                    if wkt and wkt.upper() not in ("", "UNKNOWN", "LOCAL_CS[]"):
                        has_crs = True
                        return (True, str(crs))
            except Exception:
                pass
            
            return (False, "Missing")

    except Exception as e:
        return (False, f"Error checking: {e}")


def convert_single_file(args: Tuple[Path, Path, Path, bool, Optional[str]]) -> Tuple[str, bool, str]:
    """
    Convert a single LAZ file to COPC using untwine.
    
    Two modes:
    1. With dimension reduction (dimension_reduction=True, default):
       - Prefer: untwine with --dims X,Y,Z on original file (untwine >= 1.4).
       - Fallback: if that fails, reduce to XYZ with laspy then untwine without --dims.
    2. Without dimension reduction (dimension_reduction=False):
       - Direct conversion to COPC using untwine preserving all dimensions
    
    Args:
        args: Tuple of (input_file, output_file, log_dir, dimension_reduction, original_filename)
    
    Returns:
        Tuple of (filename, success, message)
    """
    input_file, output_file, log_dir, dimension_reduction, original_filename = args
    # Use original filename for reporting if provided, otherwise use input_file.name
    display_name = original_filename if original_filename else input_file.name
    temp_xyz_las = None  # set early so except block can clean up

    def _cleanup_temp_xyz():
        if temp_xyz_las is not None and temp_xyz_las.exists():
            try:
                temp_xyz_las.unlink()
            except Exception:
                pass

    try:
        log_file = log_dir / f"{input_file.stem}_convert.log"
        untwine_cmd = get_untwine_path()
        temp_untwine_dir = log_dir / f"{input_file.stem}_untwine_temp"
        temp_untwine_dir.mkdir(parents=True, exist_ok=True)

        # When dimension_reduction: prefer untwine --dims; on failure fall back to laspy + untwine
        untwine_input = input_file
        used_laspy_fallback = False
        if dimension_reduction:
            # Prefer: untwine with --dims X,Y,Z on original file (works with untwine >= 1.4)
            args_prefer = [
                untwine_cmd, "-i", str(input_file), "-o", str(output_file),
                "--temp_dir", str(temp_untwine_dir), "--dims", "X,Y,Z"
            ]
            r = subprocess.run(args_prefer, capture_output=True, text=True, check=False)
            if r.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
                try:
                    shutil.rmtree(temp_untwine_dir)
                except Exception:
                    pass
                return (display_name, True, "Success")
            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception:
                    pass
            # Fallback: laspy reduce to XYZ then untwine without --dims (for untwine < 1.4 or buggy inputs)
            temp_xyz_las = temp_untwine_dir / f"{input_file.stem}_xyz.las"
            try:
                reduce_to_xyz_las(input_file, temp_xyz_las)
            except Exception as e:
                _cleanup_temp_xyz()
                return (display_name, False, f"untwine --dims failed; laspy fallback failed: {e}")
            untwine_input = temp_xyz_las
            used_laspy_fallback = True

        # Run untwine (no --dims: either no dimension_reduction or input is already XYZ from laspy)
        untwine_args = [
            untwine_cmd, "-i", str(untwine_input), "-o", str(output_file),
            "--temp_dir", str(temp_untwine_dir)
        ]
        result = subprocess.run(
            untwine_args,
            capture_output=True,
            text=True,
            check=False
        )

        _cleanup_temp_xyz()
        if result.returncode == 0 and temp_untwine_dir.exists():
            try:
                shutil.rmtree(temp_untwine_dir)
            except Exception:
                pass

        if result.returncode != 0:
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(untwine_args)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Stdout:\n{result.stdout}\n")
                f.write(f"Stderr:\n{result.stderr}\n")
            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception:
                    pass
            return (display_name, False, result.stderr[:200])

        if not output_file.exists() or output_file.stat().st_size == 0:
            if output_file.exists():
                try:
                    output_file.unlink()
                except Exception:
                    pass
            return (display_name, False, "Output file empty or not created")

        return (display_name, True, "Success (laspy fallback)" if used_laspy_fallback else "Success")

    except Exception as e:
        _cleanup_temp_xyz()
        return (display_name, False, str(e))


def check_file_not_empty(laz_file: Path) -> Tuple[bool, int]:
    """
    Check if LAZ/LAS file has points.
    
    Args:
        laz_file: Path to LAZ/LAS file

    Returns:
        Tuple of (is_not_empty, point_count)
    """
    try:
        import laspy
        laz_backend = _laspy_laz_backend()
        kwargs = {}
        if laz_file.suffix.lower() == ".laz" and laz_backend is not None:
            kwargs["laz_backend"] = laz_backend

        # Read only the header to be fast
        with laspy.open(str(laz_file), **kwargs) as las:
            count = las.header.point_count
            return (count > 0, count)

    except Exception:
        return (False, 0)


def convert_to_xyz_copc(
    input_dir: Path,
    output_dir: Path,
    num_workers: int,
    log_dir: Optional[Path] = None,
    dimension_reduction: bool = True
) -> List[Path]:
    """
    Convert LAZ files to COPC format using untwine.
    
    Uses untwine for efficient COPC conversion with spatial indexing.
    If dimension_reduction=True, uses --dims X,Y,Z to limit to XYZ only.
    Parallelizes across num_workers.
    
    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Directory for output COPC files
        num_workers: Number of parallel workers
        log_dir: Directory for log files (optional)
        dimension_reduction: Enable XYZ-only reduction using untwine --dims (default: True)
    
    Returns:
        List of created COPC file paths
    """
    print("=" * 60)
    if dimension_reduction:
        print("Step 1: Converting LAZ to COPC (XYZ-only: untwine --dims, laspy fallback)")
    else:
        print("Step 1: Converting LAZ to COPC (preserving all dimensions)")
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

    # Check CRS and emptiness for all input files
    print("  Checking input files...")
    files_without_crs = []
    empty_files = []
    valid_input_files = []
    
    for input_file in input_files:
        # Check if empty
        is_not_empty, count = check_file_not_empty(input_file)
        if not is_not_empty:
            empty_files.append(input_file.name)
            continue
            
        valid_input_files.append(input_file)
        
        # Check CRS
        has_crs, description = check_crs(input_file)
        if not has_crs:
            files_without_crs.append(input_file.name)

    if empty_files:
        print(f"  ⚠ Warning: {len(empty_files)} file(s) are empty (0 points) and will be skipped.")
        if len(empty_files) <= 5:
            print(f"    Empty: {', '.join(empty_files)}")
            
    if files_without_crs:
        print(f"  Note: {len(files_without_crs)} file(s) have no CRS. Pipeline will treat coordinates as local metric units.")
    elif valid_input_files:
        print(f"  ✓ All valid files have CRS defined")
    
    input_files = valid_input_files

    if not input_files:
        print(f"  No valid (non-empty) LAZ/LAS files found in {input_dir}")
        # Check if COPC files already exist
        existing_copc = list(output_dir.glob("*.copc.laz"))
        if existing_copc:
            print(f"  Found {len(existing_copc)} existing COPC files")
            return existing_copc
        return []

    print(f"  Found {len(input_files)} valid files to convert")
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
        
        # Use original file directly - untwine will handle dimension reduction with --dims
        tasks.append((input_file, output_file, log_dir, dimension_reduction, input_file.name))
    
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
    
    # Try to get SRS from the first COPC file to avoid default EPSG:4326 in tindex
    tindex_srs = None
    try:
        pdal_cmd = get_pdal_path()
        info_cmd = [pdal_cmd, "info", "--metadata", str(copc_files[0])]
        info_result = subprocess.run(info_cmd, capture_output=True, text=True, check=False)
        if info_result.returncode == 0:
            meta = json.loads(info_result.stdout)
            # Try various places where SRS might be stored
            tindex_srs = meta.get("metadata", {}).get("srs", {}).get("compoundwkt") or \
                         meta.get("metadata", {}).get("spatialreference")
    except Exception as e:
        print(f"  Warning: Could not extract SRS for tindex: {e}")

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
            "--write_absolute_path",
        ]
        
        if tindex_srs:
            cmd.append(f"--t_srs={tindex_srs}")
        
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
    output_dir: Path,
    grid_offset: float = 1.0
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


def update_tile_bounds_json_from_files(
    tile_bounds_json: Path,
    files_dir: Path,
    file_glob: str = "*.laz",
) -> int:
    """
    Update tile_bounds_tindex.json so each tile's bounds match the actual
    file header bounds from the created tiles (e.g. subsampled LAZ).
    This keeps the JSON in sync with real data extent for remap/merge matching.

    Matches tiles by label c{col:02d}_r{row:02d} (e.g. c00_r00) to filenames
    that start with that label (e.g. c00_r00_subsampled_1cm.laz).

    Returns:
        Number of tiles whose bounds were updated.
    """
    from merge_tiles import get_tile_bounds_from_header

    if not tile_bounds_json.exists():
        return 0
    with tile_bounds_json.open() as f:
        data = json.load(f)
    tiles = data.get("tiles", [])
    if not tiles:
        return 0

    # Build label -> file path from files_dir
    label_to_path: Dict[str, Path] = {}
    for f in files_dir.glob(file_glob):
        stem = f.stem
        # Match c00_r00 (prefix before _subsampled or similar)
        for sep in ("_subsampled", "_chunk", "."):
            if sep in stem:
                stem = stem.split(sep)[0]
                break
        if stem and stem not in label_to_path:
            label_to_path[stem] = f

    updated = 0
    for tile in tiles:
        col, row = tile["col"], tile["row"]
        label = f"c{col:02d}_r{row:02d}"
        path = label_to_path.get(label)
        if path is None:
            continue
        bounds = get_tile_bounds_from_header(path)
        if bounds is None:
            continue
        minx, maxx, miny, maxy = bounds
        tile["bounds"] = [[minx, maxx], [miny, maxy]]
        updated += 1

    if updated > 0:
        with tile_bounds_json.open("w") as f:
            json.dump(data, f, indent=2)
    return updated


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


def get_copc_bounds_from_tindex(tindex_file: Path) -> Dict[str, Tuple[float, float, float, float]]:
    """Get spatial bounds for each COPC file from tindex GeoPackage geometry.
    
    Returns dict mapping file path -> (minx, miny, maxx, maxy).
    """
    import sqlite3
    import struct
    
    conn = sqlite3.connect(str(tindex_file))
    cursor = conn.cursor()
    
    cursor.execute('SELECT table_name, column_name FROM gpkg_geometry_columns LIMIT 1')
    row = cursor.fetchone()
    if not row:
        conn.close()
        return {}
    table_name, geom_col = row
    
    cursor.execute(f'SELECT Location, "{geom_col}" FROM "{table_name}"')
    bounds_map = {}
    for filepath, geom_blob in cursor.fetchall():
        if not geom_blob or not filepath:
            continue
        try:
            # GeoPackage geometry binary: header (magic GP, version, flags, srs_id, envelope)
            # flags byte at offset 3 tells envelope type
            flags = geom_blob[3]
            envelope_type = (flags >> 1) & 0x07
            header_size = 8  # magic(2) + version(1) + flags(1) + srs_id(4)
            if envelope_type == 1:  # [minx, maxx, miny, maxy]
                minx, maxx, miny, maxy = struct.unpack_from('<dddd', geom_blob, header_size)
                bounds_map[filepath] = (minx, miny, maxx, maxy)
            elif envelope_type == 2:  # [minx, maxx, miny, maxy, minz, maxz]
                minx, maxx, miny, maxy = struct.unpack_from('<dddd', geom_blob, header_size)
                bounds_map[filepath] = (minx, miny, maxx, maxy)
        except (struct.error, IndexError):
            continue
    
    conn.close()
    return bounds_map


def _parse_proj_bounds(proj_bounds: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse '([xmin,xmax],[ymin,ymax])' into (xmin, ymin, xmax, ymax)."""
    try:
        s = proj_bounds.strip().strip("()")
        parts = s.split("],[")
        xpart = parts[0].strip("([])").split(",")
        ypart = parts[1].strip("([])").split(",")
        xmin, xmax = float(xpart[0]), float(xpart[1])
        ymin, ymax = float(ypart[0]), float(ypart[1])
        return (xmin, ymin, xmax, ymax)
    except (ValueError, IndexError):
        return None


def _bounds_overlap(a: Tuple[float, float, float, float],
                    b: Tuple[float, float, float, float]) -> bool:
    """Check if two (minx, miny, maxx, maxy) boxes overlap."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def filter_copc_files_for_tile(
    copc_files: List[str],
    copc_bounds: Dict[str, Tuple[float, float, float, float]],
    tile_bounds: Tuple[float, float, float, float]
) -> List[str]:
    """Return only COPC files whose bounds overlap the tile bounds."""
    result = []
    for f in copc_files:
        fb = copc_bounds.get(f)
        if fb is None:
            result.append(f)  # no bounds info, keep as candidate
        elif _bounds_overlap(fb, tile_bounds):
            result.append(f)
    return result


def _crop_with_laspy(
    input_file: str,
    output_file: Path,
    bounds: Tuple[float, float, float, float],
) -> Tuple[bool, int, str]:
    """Crop a LAZ/COPC file to bounds using laspy + numpy.

    Reads the full file, applies a bounding box mask, and writes the
    cropped points as compressed LAZ.  This bypasses PDAL's readers.copc
    which hangs on large selections (>50M points).

    Args:
        input_file: Path to input LAZ/COPC file.
        output_file: Path for the cropped output LAZ file.
        bounds: (xmin, ymin, xmax, ymax) bounding box.

    Returns:
        (success, point_count, message)
    """
    import laspy
    import numpy as np

    xmin, ymin, xmax, ymax = bounds

    try:
        laz_backend = _laspy_laz_backend()
        kwargs = {}
        if input_file.lower().endswith(".laz") and laz_backend is not None:
            kwargs["laz_backend"] = laz_backend

        las = laspy.read(input_file, **kwargs)

        mask = (
            (np.asarray(las.x) >= xmin)
            & (np.asarray(las.x) <= xmax)
            & (np.asarray(las.y) >= ymin)
            & (np.asarray(las.y) <= ymax)
        )

        count = int(mask.sum())
        if count == 0:
            return (True, 0, "No points in bounds")

        cropped = las.points[mask]

        new_header = laspy.LasHeader(
            point_format=las.header.point_format,
            version=las.header.version,
        )
        new_header.offsets = las.header.offsets
        new_header.scales = las.header.scales
        for vlr in las.header.vlrs:
            if vlr.record_id in (1, 2) and vlr.user_id == "copc":
                continue
            new_header.vlrs.append(vlr)
        for dim in las.point_format.extra_dims:
            if dim.name not in [d.name for d in new_header.point_format.extra_dims]:
                new_header.add_extra_dim(laspy.ExtraBytesParams(
                    name=dim.name, type=dim.dtype, description=dim.description or "",
                ))

        new_las = laspy.LasData(new_header)
        new_las.points = cropped

        write_kwargs = {}
        if laz_backend is not None:
            write_kwargs["laz_backend"] = laz_backend
        new_las.write(str(output_file), **write_kwargs)

        return (True, count, "OK")
    except Exception as e:
        return (False, 0, str(e))


def process_single_tile(args: Tuple) -> Tuple[str, bool, str]:
    """
    Process a single tile: extract points from COPC files within bounds.

    Uses laspy + numpy for the crop step (PDAL readers.copc hangs on
    large subsets >50M points).  Part files are written as LAZ, then
    merged into a final COPC tile with PDAL.
    
    Args:
        args: Tuple of (label, proj_bounds, geo_bounds, copc_files, copc_bounds, tiles_dir, log_dir, threads)
              copc_bounds: dict mapping file path -> (minx, miny, maxx, maxy)
    
    Returns:
        Tuple of (label, success, message)
    """
    label, proj_bounds, geo_bounds, copc_files, copc_bounds, tiles_dir, log_dir, threads = args
    
    final_tile = tiles_dir / f"{label}.copc.laz"
    
    # Skip if already exists
    if final_tile.exists() and final_tile.stat().st_size > 0:
        return (label, True, "Already exists")
    
    # Pre-filter: only process COPC files that spatially overlap this tile
    tile_bounds = _parse_proj_bounds(proj_bounds)
    if tile_bounds and copc_bounds:
        relevant_files = filter_copc_files_for_tile(copc_files, copc_bounds, tile_bounds)
    else:
        relevant_files = copc_files
    
    if not tile_bounds:
        return (label, False, "Could not parse tile bounds")

    try:
        tile_dir = tiles_dir / label
        tile_dir.mkdir(exist_ok=True)
        
        parts_created = 0
        total_points = 0
        
        for part_num, copc_file in enumerate(relevant_files):
            if not os.path.isfile(copc_file):
                continue
            
            part_file = tile_dir / f"part_{part_num}.laz"
            
            ok, npts, msg = _crop_with_laspy(copc_file, part_file, tile_bounds)
            
            if ok and npts > 0 and part_file.exists() and part_file.stat().st_size > 0:
                parts_created += 1
                total_points += npts
            elif part_file.exists():
                part_file.unlink()
        
        if parts_created == 0:
            if tile_dir.exists() and not any(tile_dir.iterdir()):
                tile_dir.rmdir()
            return (label, True, "No data in bounds")
        
        # Merge parts into final COPC tile
        parts = sorted(tile_dir.glob("part_*.laz"))
        
        if len(parts) == 1:
            # Convert single LAZ part → COPC using PDAL
            convert_pipeline = {
                "pipeline": [
                    {"type": "readers.las", "filename": str(parts[0])},
                    {
                        "type": "writers.copc",
                        "filename": str(final_tile),
                        "forward": "all",
                        "extra_dims": "all",
                    }
                ]
            }
            pipeline_file = log_dir / f"{label}_convert_pipeline.json"
            with open(pipeline_file, 'w') as f:
                json.dump(convert_pipeline, f)

            pdal_cmd = get_pdal_path()
            result = subprocess.run(
                [pdal_cmd, "pipeline", str(pipeline_file)],
                capture_output=True, text=True, check=False,
            )
            if pipeline_file.exists():
                pipeline_file.unlink()
            if result.returncode != 0:
                return (label, False, f"COPC convert failed: {result.stderr[:200]}")
        else:
            readers = [{"type": "readers.las", "filename": str(p)} for p in parts]
            merge_pipeline = {
                "pipeline": readers + [
                    {"type": "filters.merge"},
                    {
                        "type": "writers.copc",
                        "filename": str(final_tile),
                        "forward": "all",
                        "extra_dims": "all",
                        "scale_x": 0.01,
                        "scale_y": 0.01,
                        "scale_z": 0.01,
                    }
                ]
            }
            
            merge_pipeline_file = log_dir / f"{label}_merge_pipeline.json"
            with open(merge_pipeline_file, 'w') as f:
                json.dump(merge_pipeline, f)
            
            pdal_cmd = get_pdal_path()
            result = subprocess.run(
                [pdal_cmd, "pipeline", str(merge_pipeline_file)],
                capture_output=True, text=True, check=False,
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
        
        return (label, True, f"{parts_created} parts, {total_points:,} pts")
        
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
    
    # Load spatial bounds for pre-filtering (skip files with no overlap)
    copc_bounds = get_copc_bounds_from_tindex(tindex_file)
    if copc_bounds:
        print(f"  Source files: {len(copc_files)} ({len(copc_bounds)} with spatial bounds for pre-filtering)")
    else:
        print(f"  Source files: {len(copc_files)} (no spatial bounds; will process all files per tile)")
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
                jobs.append((label, proj_bounds, geo_bounds, copc_files, copc_bounds, tiles_dir, log_dir, threads))
    
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
    dimension_reduction: bool = True,
    tiling_threshold: float = None
) -> Path:
    """
    Run the complete tiling pipeline.

    Steps:
    1. Convert LAZ to COPC (with or without dimension reduction using untwine --dims)
    2. Build spatial index
    3. Calculate tile bounds
    4. Create overlapping tiles

    If input folder contains a single file below tiling_threshold, skips steps 2-4.

    Args:
        input_dir: Directory containing input LAZ files
        output_dir: Base output directory
        tile_length: Tile size in meters
        tile_buffer: Buffer overlap in meters
        grid_offset: Offset from min coordinates
        num_workers: Number of parallel conversion workers
        threads: Threads per COPC writer
        max_tile_procs: Maximum parallel tile processes
        dimension_reduction: Enable XYZ-only reduction using untwine --dims (default: True)
        tiling_threshold: File size threshold in MB. If single file below this size, skip tiling

    Returns:
        Path to tiles directory (or COPC directory if tiling was skipped)
    """
    print("=" * 60)
    print("3DTrees Tiling Pipeline")
    print("=" * 60)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Tile size: {tile_length}m with {tile_buffer}m buffer")
    print()
    
    # Define paths
    copc_dir = output_dir / ("copc_full" if not dimension_reduction else "copc_xyz")
    tiles_dir = output_dir / f"tiles_{int(tile_length)}m"
    log_dir = output_dir / "logs"
    tindex_file = output_dir / f"tindex_{int(tile_length)}m.gpkg"
    
    # Check if we should skip tiling based on original input file size
    # This check happens BEFORE COPC conversion to use original file size
    should_skip_tiling = False
    if tiling_threshold is not None:
        input_files = list(input_dir.glob("*.laz")) + list(input_dir.glob("*.las"))
        input_files = [f for f in input_files if not f.name.endswith('.copc.laz')]
        
        if len(input_files) == 1:
            original_size_mb = input_files[0].stat().st_size / (1024 * 1024)
            if original_size_mb < tiling_threshold:
                should_skip_tiling = True
                print("=" * 60)
                print("Tiling Threshold Check")
                print("=" * 60)
                print(f"  Single file detected: {input_files[0].name}")
                print(f"  Original file size: {original_size_mb:.2f} MB")
                print(f"  Threshold: {tiling_threshold} MB")
                print(f"  Decision: Will skip tiling after COPC conversion")
                print("=" * 60)
                print()
    
    # Step 1: Convert to COPC (untwine handles dimension reduction with --dims if enabled)
    copc_files = convert_to_xyz_copc(input_dir, copc_dir, num_workers, log_dir, dimension_reduction)

    if not copc_files:
        raise ValueError("No COPC files created or found")

    # Step 2: Build tindex
    tindex_file = build_tindex(copc_dir, tindex_file)
    
    # Step 3: Calculate tile bounds
    jobs_file, bounds_json, env = calculate_tile_bounds(
        tindex_file, tile_length, tile_buffer, output_dir, grid_offset
    )

    # Symlink tindex to fixed name for Galaxy if needed
    fixed_tindex = output_dir / "tindex.gpkg"
    if not fixed_tindex.exists() and tindex_file.exists():
        if fixed_tindex.is_symlink():
            fixed_tindex.unlink()
        fixed_tindex.symlink_to(tindex_file.name)

    # Step 4: Plot the tiles
    plot_tiles_and_copc.plot_extents(tindex_file, bounds_json, output_dir/"overview_copc_tiles.png")

    # Check if we should skip tiling based on original input file size
    # We do this AFTER tindex/plotting so we still get the overview outputs
    if should_skip_tiling:
        print()
        print("=" * 60)
        print("Skipping Tiling (Single Small File)")
        print("=" * 60)
        print(f"  Threshold was based on original input file size")
        print(f"  Returning COPC directory for direct subsampling")
        print("=" * 60)
        return copc_dir
    
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

