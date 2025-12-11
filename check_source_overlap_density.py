#!/usr/bin/env python3
"""
Check if overlapping source COPC files result in higher point density in merged tiles.
This verifies that points from both source files are included in the overlap region.
"""

import sys
import json
import pdal
import sqlite3
from pathlib import Path
from collections import defaultdict

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

def get_tile_bounds(tile_jobs_file):
    """Read tile bounds from tile_jobs file."""
    tiles = {}
    with open(tile_jobs_file, 'r') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                label = parts[0].strip()
                proj_bounds = parts[1].strip()
                # Parse bounds: ([xmin, xmax], [ymin, ymax])
                bounds_str = proj_bounds.strip('()')
                x_part, y_part = bounds_str.split('],[')
                xmin, xmax = map(float, x_part.strip('[').split(','))
                ymin, ymax = map(float, y_part.strip(']').split(','))
                tiles[label] = {
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax
                }
    return tiles

def get_copc_bounds_from_header(copc_file):
    """Get bounds from COPC file header without reading all points."""
    import subprocess
    
    # Method 1: Use laspy to read header only (fastest for COPC files)
    if HAS_LASPY:
        try:
            las = laspy.open(str(copc_file))
            header = las.header
            return {
                'xmin': header.x_min,
                'xmax': header.x_max,
                'ymin': header.y_min,
                'ymax': header.y_max
            }
        except Exception:
            pass
    
    # Method 2: Use PDAL info command to get metadata only (fast)
    try:
        result = subprocess.run(
            ['pdal', 'info', '--metadata', str(copc_file)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            import json as json_lib
            info_data = json_lib.loads(result.stdout)
            metadata = info_data.get('metadata', {})
            copc_info = metadata.get('readers.copc', {})
            if 'minx' in copc_info and 'maxx' in copc_info:
                return {
                    'xmin': copc_info['minx'],
                    'xmax': copc_info['maxx'],
                    'ymin': copc_info['miny'],
                    'ymax': copc_info['maxy']
                }
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass
    
    # Method 3: Fallback - use PDAL Python API (slower, reads all points)
    try:
        pipeline_json = [{"type": "readers.copc", "filename": str(copc_file)}]
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        metadata = pipeline.metadata.get('metadata', {})
        copc_info = metadata.get('readers.copc', {})
        if 'minx' in copc_info and 'maxx' in copc_info:
            return {
                'xmin': copc_info['minx'],
                'xmax': copc_info['maxx'],
                'ymin': copc_info['miny'],
                'ymax': copc_info['maxy']
            }
    except Exception:
        pass
    
    return None

def get_copc_files_from_tindex(tindex_file):
    """Get all COPC files and their bounds from tindex using geometry column."""
    copc_files = {}
    try:
        conn = sqlite3.connect(tindex_file)
        # Try to load SpatiaLite extension for spatial functions
        try:
            conn.enable_load_extension(True)
            conn.load_extension('mod_spatialite')
        except Exception:
            # Extension loading failed, will use fallback
            pass
        
        cursor = conn.cursor()
        
        # Get all feature tables
        cursor.execute("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'")
        tables = cursor.fetchall()
        
        for (table_name,) in tables:
            # Get files - we'll read bounds from COPC headers to get correct CRS
            # (tindex geometry is in lat/lon but we need projected coords)
            cursor.execute(f"SELECT Location FROM \"{table_name}\"")
            for (location,) in cursor.fetchall():
                file_path = Path(location)
                if file_path.exists():
                    # Read bounds from COPC file header (fast with laspy, correct CRS)
                    bounds = get_copc_bounds_from_header(file_path)
                    if bounds:
                        copc_files[str(file_path)] = {
                            'xmin': bounds['xmin'],
                            'xmax': bounds['xmax'],
                            'ymin': bounds['ymin'],
                            'ymax': bounds['ymax'],
                            'file': file_path
                        }
        
        conn.close()
    except Exception as e:
        print(f"Error reading tindex: {e}", file=sys.stderr)
    
    return copc_files

def find_overlapping_copc_files(copc_files):
    """Find pairs of COPC files that overlap."""
    overlaps = []
    file_list = list(copc_files.items())
    
    for i in range(len(file_list)):
        for j in range(i + 1, len(file_list)):
            file1_path, file1_bounds = file_list[i]
            file2_path, file2_bounds = file_list[j]
            
            # Check if they overlap
            overlap_xmin = max(file1_bounds['xmin'], file2_bounds['xmin'])
            overlap_xmax = min(file1_bounds['xmax'], file2_bounds['xmax'])
            overlap_ymin = max(file1_bounds['ymin'], file2_bounds['ymin'])
            overlap_ymax = min(file1_bounds['ymax'], file2_bounds['ymax'])
            
            if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
                overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
                overlaps.append({
                    'file1': file1_path,
                    'file2': file2_path,
                    'file1_bounds': file1_bounds,
                    'file2_bounds': file2_bounds,
                    'xmin': overlap_xmin,
                    'xmax': overlap_xmax,
                    'ymin': overlap_ymin,
                    'ymax': overlap_ymax,
                    'area': overlap_area
                })
    
    return overlaps

def count_points_in_bounds(copc_file, xmin, xmax, ymin, ymax):
    """Count points in a bounded region using Python PDAL."""
    if not Path(copc_file).exists():
        return None
    
    try:
        pipeline_json = [
            {
                "type": "readers.copc",
                "filename": str(copc_file),
                "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])"
            }
        ]
        
        pipeline = pdal.Pipeline(json.dumps(pipeline_json))
        pipeline.execute()
        
        arrays = pipeline.arrays
        if len(arrays) > 0:
            return len(arrays[0])
        return 0
    except Exception as e:
        print(f"    Error reading {copc_file}: {e}", file=sys.stderr)
        return None

def find_tiles_covering_overlap(overlap, tiles, tiles_dir):
    """Find which tiles cover this overlap region."""
    covering_tiles = []
    
    for tile_label, tile_bounds in tiles.items():
        # Check if tile covers the overlap
        if (tile_bounds['xmin'] <= overlap['xmin'] and 
            tile_bounds['xmax'] >= overlap['xmax'] and
            tile_bounds['ymin'] <= overlap['ymin'] and
            tile_bounds['ymax'] >= overlap['ymax']):
            
            tile_file = tiles_dir / f"{tile_label}.copc.laz"
            if tile_file.exists():
                covering_tiles.append({
                    'label': tile_label,
                    'file': tile_file,
                    'bounds': tile_bounds
                })
    
    return covering_tiles

def main():
    if len(sys.argv) != 4:
        print("Usage: check_source_overlap_density.py <tiles_dir> <tile_jobs_file> <tindex_file>")
        sys.exit(1)
    
    tiles_dir = Path(sys.argv[1])
    tile_jobs_file = sys.argv[2]
    tindex_file = sys.argv[3]
    
    print("=" * 80)
    print("Source File Overlap Density Analysis")
    print("=" * 80)
    print()
    print("This test verifies that overlapping source COPC files result in")
    print("higher point density in merged tiles (points from both files combined).")
    print()
    
    # Load data
    print("Loading tile information...")
    tiles = get_tile_bounds(tile_jobs_file)
    print(f"  Found {len(tiles)} tiles")
    
    print("Loading source COPC file information...")
    print("  (This may take a moment as we read bounds from each COPC file...)")
    copc_files = get_copc_files_from_tindex(tindex_file)
    print(f"  Found {len(copc_files)} source COPC files with bounds")
    
    print("Finding overlapping source files...")
    overlaps = find_overlapping_copc_files(copc_files)
    print(f"  Found {len(overlaps)} overlapping pairs")
    print()
    
    if len(overlaps) == 0:
        print("No overlapping source files found.")
        return
    
    # Analyze each overlap
    results = []
    
    # Just analyze first 2 overlapping pairs for quick testing
    for i, overlap in enumerate(overlaps[:2]):
        print(f"Analyzing overlap {i+1}/{min(len(overlaps), 10)}:")
        print(f"  File 1: {Path(overlap['file1']).name}")
        print(f"  File 2: {Path(overlap['file2']).name}")
        print(f"  Overlap bounds: X=[{overlap['xmin']:.2f}, {overlap['xmax']:.2f}], "
              f"Y=[{overlap['ymin']:.2f}, {overlap['ymax']:.2f}]")
        print(f"  Overlap area: {overlap['area']:.2f} m²")
        
        # Count points from each source file
        print(f"  Counting points from file 1...")
        count1 = count_points_in_bounds(
            overlap['file1'],
            overlap['xmin'], overlap['xmax'],
            overlap['ymin'], overlap['ymax']
        )
        
        print(f"  Counting points from file 2...")
        count2 = count_points_in_bounds(
            overlap['file2'],
            overlap['xmin'], overlap['xmax'],
            overlap['ymin'], overlap['ymax']
        )
        
        if count1 is None or count2 is None:
            print(f"  ⚠️  Error counting points from source files")
            print()
            continue
        
        density1 = count1 / overlap['area'] if overlap['area'] > 0 else 0
        density2 = count2 / overlap['area'] if overlap['area'] > 0 else 0
        expected_total = count1 + count2
        expected_density = expected_total / overlap['area'] if overlap['area'] > 0 else 0
        
        print(f"  Points from file 1: {count1:,} (density: {density1:.2f} pts/m²)")
        print(f"  Points from file 2: {count2:,} (density: {density2:.2f} pts/m²)")
        print(f"  Expected total (if merged): {expected_total:,} (density: {expected_density:.2f} pts/m²)")
        
        # Find tiles covering this overlap
        covering_tiles = find_tiles_covering_overlap(overlap, tiles, tiles_dir)
        
        if len(covering_tiles) == 0:
            print(f"  ⚠️  No tiles found covering this overlap region")
            print()
            continue
        
        # Check each covering tile
        for tile_info in covering_tiles:
            print(f"  Checking merged tile: {tile_info['label']}")
            merged_count = count_points_in_bounds(
                tile_info['file'],
                overlap['xmin'], overlap['xmax'],
                overlap['ymin'], overlap['ymax']
            )
            
            if merged_count is None:
                print(f"    ⚠️  Error counting points in merged tile")
                continue
            
            merged_density = merged_count / overlap['area'] if overlap['area'] > 0 else 0
            
            print(f"    Points in merged tile: {merged_count:,} (density: {merged_density:.2f} pts/m²)")
            
            # Compare
            if merged_count >= expected_total * 0.95:  # Allow 5% tolerance
                print(f"    ✓ EXCELLENT: Merged tile has ~{merged_count/expected_total*100:.1f}% of expected points")
                print(f"      Density is {merged_density/expected_density*100:.1f}% of expected")
                status = "PASS"
            elif merged_count >= max(count1, count2) * 1.1:  # At least 10% more than single file
                print(f"    ✓ GOOD: Merged tile has more points than either single file")
                print(f"      {merged_count:,} vs max({count1:,}, {count2:,})")
                status = "PASS"
            elif merged_count >= max(count1, count2):
                print(f"    ⚠️  CAUTION: Merged tile has similar points to single file")
                print(f"      May not be merging correctly")
                status = "WARNING"
            else:
                print(f"    ❌ FAIL: Merged tile has fewer points than single file!")
                print(f"      {merged_count:,} < max({count1:,}, {count2:,})")
                status = "FAIL"
            
            results.append({
                'file1': Path(overlap['file1']).name,
                'file2': Path(overlap['file2']).name,
                'tile': tile_info['label'],
                'count1': count1,
                'count2': count2,
                'merged_count': merged_count,
                'expected_total': expected_total,
                'density1': density1,
                'density2': density2,
                'merged_density': merged_density,
                'expected_density': expected_density,
                'status': status
            })
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if results:
        pass_count = sum(1 for r in results if r['status'] == 'PASS')
        warn_count = sum(1 for r in results if r['status'] == 'WARNING')
        fail_count = sum(1 for r in results if r['status'] == 'FAIL')
        
        print(f"Total overlap regions analyzed: {len(results)}")
        print(f"  ✓ PASS: {pass_count}")
        print(f"  ⚠️  WARNING: {warn_count}")
        print(f"  ❌ FAIL: {fail_count}")
        print()
        
        if fail_count > 0:
            print("Failed cases:")
            for r in results:
                if r['status'] == 'FAIL':
                    print(f"  {r['file1']} + {r['file2']} in tile {r['tile']}: "
                          f"{r['merged_count']:,} < expected {r['expected_total']:,}")
    else:
        print("No results to summarize.")

if __name__ == "__main__":
    main()

