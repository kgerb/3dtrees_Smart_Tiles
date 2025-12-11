#!/usr/bin/env python3
"""
Visualize file extents and generated tile extents.
Shows input LAZ/LAS files and output tiles on the same plot.
"""

import json
import sys
import subprocess
import re
from pathlib import Path
import argparse
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    from pyproj import Transformer
except ImportError as e:
    print(f"ERROR: Required packages missing. Install with: pip install matplotlib pyproj")
    print(f"Error: {e}")
    sys.exit(1)

# Note: We use 'pdal info' command line tool, not Python bindings
# This is more efficient as it only reads file headers


def get_file_bounds(file_path: Path, target_crs: str = "EPSG:32632"):
    """Get bounds of a LAZ/LAS file using PDAL info (header only, no point loading)."""
    try:
        # Use pdal info command to get metadata without loading points
        # This is much faster for large files - only reads the header
        cmd = ["pdal", "info", "--metadata", str(file_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            print(f"WARNING: pdal info failed for {file_path.name}: {error_msg}", file=sys.stderr)
            return None
        
        # Parse JSON output - ignore PROJ warnings in stderr
        try:
            info_data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            print(f"WARNING: Could not parse JSON from pdal info for {file_path.name}: {e}", file=sys.stderr)
            return None
        
        # Extract bounds - they're directly in metadata, not nested under readers.las
        metadata = info_data.get('metadata', {})
        
        # Get bounds from metadata (top level in metadata dict)
        minx = metadata.get('minx')
        miny = metadata.get('miny')
        maxx = metadata.get('maxx')
        maxy = metadata.get('maxy')
        
        if minx is None or miny is None or maxx is None or maxy is None:
            print(f"WARNING: Could not extract bounds from {file_path.name} metadata", file=sys.stderr)
            return None
        
        # Get CRS from metadata - try multiple paths
        srs = None
        
        # First, try to extract EPSG code from spatialreference string
        spatialref = metadata.get('spatialreference', '')
        if spatialref:
            # Look for EPSG code in the string
            epsg_match = re.search(r'EPSG["\']?\s*[:\s]+(\d+)', spatialref, re.IGNORECASE)
            if epsg_match:
                srs = f"EPSG:{epsg_match.group(1)}"
            else:
                # Try to identify UTM zone from the string
                utm_match = re.search(r'UTM\s+zone\s+(\d+)([NS])?', spatialref, re.IGNORECASE)
                if utm_match:
                    zone = int(utm_match.group(1))
                    hemisphere = utm_match.group(2) if utm_match.group(2) else 'N'
                    # UTM zones: 32601-32660 (N), 32701-32760 (S)
                    if hemisphere.upper() == 'N':
                        epsg_code = 32600 + zone
                    else:
                        epsg_code = 32700 + zone
                    srs = f"EPSG:{epsg_code}"
        
        # If no EPSG found, try srs dict
        if not srs:
            srs_dict = metadata.get('srs', {})
            if isinstance(srs_dict, dict):
                # Try to get from compoundwkt or horizontal
                srs_str = srs_dict.get('compoundwkt', '') or srs_dict.get('horizontal', '')
                if srs_str:
                    epsg_match = re.search(r'EPSG["\']?\s*[:\s]+(\d+)', srs_str, re.IGNORECASE)
                    if epsg_match:
                        srs = f"EPSG:{epsg_match.group(1)}"
                    else:
                        # Try UTM zone detection here too
                        utm_match = re.search(r'UTM\s+zone\s+(\d+)([NS])?', srs_str, re.IGNORECASE)
                        if utm_match:
                            zone = int(utm_match.group(1))
                            hemisphere = utm_match.group(2) if utm_match.group(2) else 'N'
                            if hemisphere.upper() == 'N':
                                epsg_code = 32600 + zone
                            else:
                                epsg_code = 32700 + zone
                            srs = f"EPSG:{epsg_code}"
        
        # Default if still not found - check if bounds look like they're already projected
        if not srs:
            # If bounds are large numbers (likely projected), assume they're already in target CRS
            if abs(minx) > 180 or abs(miny) > 90:
                # Likely already in a projected CRS
                # Check if target_crs is a valid EPSG code
                if target_crs.startswith('EPSG:'):
                    srs = target_crs  # Assume already in target CRS
                else:
                    srs = 'EPSG:4326'  # Fallback
            else:
                srs = 'EPSG:4326'  # Geographic coordinates
        
        # Transform to target CRS if needed
        if srs != target_crs:
            try:
                transformer = Transformer.from_crs(srs, target_crs, always_xy=True)
                # Transform all four corners
                corners = [
                    transformer.transform(minx, miny),
                    transformer.transform(minx, maxy),
                    transformer.transform(maxx, miny),
                    transformer.transform(maxx, maxy),
                ]
                proj_xs, proj_ys = zip(*corners)
                minx, maxx = min(proj_xs), max(proj_xs)
                miny, maxy = min(proj_ys), max(proj_ys)
            except Exception as e:
                print(f"WARNING: Could not transform {file_path.name} from {srs} to {target_crs}: {e}", file=sys.stderr)
                return None
        
        # Validate bounds - check for NaN or invalid values
        import math
        if math.isnan(minx) or math.isnan(miny) or math.isnan(maxx) or math.isnan(maxy):
            print(f"WARNING: Invalid bounds (NaN) for {file_path.name}", file=sys.stderr)
            return None
        
        if minx >= maxx or miny >= maxy:
            print(f"WARNING: Invalid bounds (min >= max) for {file_path.name}: ({minx}, {miny}, {maxx}, {maxy})", file=sys.stderr)
            return None
        
        return (minx, miny, maxx, maxy), srs
        
    except Exception as e:
        print(f"WARNING: Could not read bounds from {file_path.name}: {e}", file=sys.stderr)
        return None


def load_file_extents(folder_path: Path, target_crs: str = "EPSG:32632"):
    """Load extents of all LAZ/LAS files in folder."""
    extents = []
    filenames = []
    crs_list = []
    
    # Find all LAZ/LAS files
    files = list(folder_path.glob("*.laz")) + list(folder_path.glob("*.las"))
    
    if not files:
        print(f"ERROR: No .laz or .las files found in {folder_path}")
        sys.exit(1)
    
    print(f"Found {len(files)} files, reading extents...")
    
    successful = 0
    for file_path in sorted(files):
        result = get_file_bounds(file_path, target_crs)
        if result:
            bounds, crs = result
            extents.append(bounds)
            filenames.append(file_path.name)
            crs_list.append(crs)
            successful += 1
        else:
            print(f"  Failed to read bounds from: {file_path.name}")
    
    if not extents:
        print(f"\nERROR: Could not read bounds from any files ({successful}/{len(files)} successful)")
        print("  Make sure:")
        print("    1. PDAL is installed and in PATH")
        print("    2. Files are valid LAZ/LAS files")
        print("    3. Files have valid headers")
        sys.exit(1)
    
    print(f"  Successfully read bounds from {successful}/{len(files)} files")
    
    # Use most common CRS or target CRS
    if crs_list:
        most_common_crs = max(set(crs_list), key=crs_list.count)
        if most_common_crs != target_crs:
            print(f"Note: Most files are in {most_common_crs}, but plotting in {target_crs}")
    
    return extents, filenames


def calculate_tiles(extents, tile_length: float, tile_buffer: float, proj_crs: str = "EPSG:32632"):
    """Calculate tile grid from file extents."""
    # Get overall extent
    all_xs = []
    all_ys = []
    
    for xmin, ymin, xmax, ymax in extents:
        # Validate each extent
        if math.isnan(xmin) or math.isnan(ymin) or math.isnan(xmax) or math.isnan(ymax):
            continue
        all_xs.extend([xmin, xmax])
        all_ys.extend([ymin, ymax])
    
    if not all_xs or not all_ys:
        raise ValueError("No valid extents found - all extents contain NaN values")
    
    overall_xmin, overall_xmax = min(all_xs), max(all_xs)
    overall_ymin, overall_ymax = min(all_ys), max(all_ys)
    
    # Validate overall extent
    if math.isnan(overall_xmin) or math.isnan(overall_ymin) or math.isnan(overall_xmax) or math.isnan(overall_ymax):
        raise ValueError(f"Invalid overall extent: ({overall_xmin}, {overall_ymin}, {overall_xmax}, {overall_ymax})")
    
    if overall_xmin >= overall_xmax or overall_ymin >= overall_ymax:
        raise ValueError(f"Invalid overall extent: min >= max: ({overall_xmin}, {overall_ymin}, {overall_xmax}, {overall_ymax})")
    
    # Calculate number of tiles
    width = overall_xmax - overall_xmin
    height = overall_ymax - overall_ymin
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid extent dimensions: width={width}, height={height}")
    
    num_cols = math.ceil(width / tile_length)
    num_rows = math.ceil(height / tile_length)
    
    tiles = []
    
    for row in range(num_rows):
        for col in range(num_cols):
            # Tile bounds with buffer
            tile_xmin = overall_xmin + col * tile_length - tile_buffer
            tile_ymin = overall_ymin + row * tile_length - tile_buffer
            tile_xmax = overall_xmin + (col + 1) * tile_length + tile_buffer
            tile_ymax = overall_ymin + (row + 1) * tile_length + tile_buffer
            
            # Core bounds (without buffer)
            core_xmin = overall_xmin + col * tile_length
            core_ymin = overall_ymin + row * tile_length
            core_xmax = overall_xmin + (col + 1) * tile_length
            core_ymax = overall_ymin + (row + 1) * tile_length
            
            # Only include tiles that intersect with data
            if tile_xmax < overall_xmin or tile_xmin > overall_xmax or \
               tile_ymax < overall_ymin or tile_ymin > overall_ymax:
                continue
            
            tiles.append({
                'label': f"c{col:02d}_r{row:02d}",
                'bounds': (tile_xmin, tile_ymin, tile_xmax, tile_ymax),
                'core': (core_xmin, core_ymin, core_xmax, core_ymax)
            })
    
    return tiles, (overall_xmin, overall_ymin, overall_xmax, overall_ymax)


def plot_extents(folder_path: Path, tile_length: float, tile_buffer: float, 
                 output_png: Path, proj_crs: str = "EPSG:32632"):
    """Create visualization of files and tiles."""
    print("Loading file extents...")
    file_extents, file_names = load_file_extents(folder_path, target_crs=proj_crs)
    print(f"Found {len(file_extents)} files")
    
    print("Calculating tile grid...")
    tiles, overall_extent = calculate_tiles(file_extents, tile_length, tile_buffer, proj_crs)
    print(f"Generated {len(tiles)} tiles")
    
    overall_xmin, overall_ymin, overall_xmax, overall_ymax = overall_extent
    
    # Add padding
    x_padding = (overall_xmax - overall_xmin) * 0.05
    y_padding = (overall_ymax - overall_ymin) * 0.05
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot file extents
    file_patches = []
    for xmin, ymin, xmax, ymax in file_extents:
        width = xmax - xmin
        height = ymax - ymin
        rect = mpatches.Rectangle((xmin, ymin), width, height, 
                                  edgecolor='blue', facecolor='lightblue', 
                                  alpha=0.5, linewidth=1.5)
        file_patches.append(rect)
    
    file_collection = PatchCollection(file_patches, match_original=True)
    ax.add_collection(file_collection)
    
    # Add file labels (only for smaller number of files to avoid clutter)
    if len(file_extents) <= 20:
        for (xmin, ymin, xmax, ymax), name in zip(file_extents, file_names):
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            # Shorten filename for display
            short_name = name[:20] + "..." if len(name) > 20 else name
            ax.text(center_x, center_y, short_name, 
                    ha='center', va='center', 
                    fontsize=7, color='darkblue', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot tile extents
    tile_patches = []
    for tile in tiles:
        xmin, ymin, xmax, ymax = tile['bounds']
        width = xmax - xmin
        height = ymax - ymin
        rect = mpatches.Rectangle((xmin, ymin), width, height,
                                  edgecolor='red', facecolor='none',
                                  linewidth=2, linestyle='--')
        tile_patches.append(rect)
        
        # Add tile label
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        ax.text(center_x, center_y, tile['label'],
                ha='center', va='center',
                fontsize=9, color='red', weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8))
    
    tile_collection = PatchCollection(tile_patches, match_original=True)
    ax.add_collection(tile_collection)
    
    # Plot tile cores
    core_patches = []
    for tile in tiles:
        if tile.get('core'):
            core = tile['core']
            xmin, ymin, xmax, ymax = core
            width = xmax - xmin
            height = ymax - ymin
            rect = mpatches.Rectangle((xmin, ymin), width, height,
                                      edgecolor='darkred', facecolor='pink',
                                      alpha=0.3, linewidth=1)
            core_patches.append(rect)
    
    if core_patches:
        core_collection = PatchCollection(core_patches, match_original=True)
        ax.add_collection(core_collection)
    
    # Set limits and labels
    ax.set_xlim(overall_xmin - x_padding, overall_xmax + x_padding)
    ax.set_ylim(overall_ymin - y_padding, overall_ymax + y_padding)
    ax.set_aspect('equal')
    ax.set_xlabel(f'X (Projected CRS: {proj_crs})', fontsize=12)
    ax.set_ylabel(f'Y (Projected CRS: {proj_crs})', fontsize=12)
    ax.set_title(f'File Extents and Generated Tiles\n(Blue = Input files, Red dashed = Tile bounds with {tile_buffer}m buffer, Pink = Tile cores)', 
                 fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    file_legend = mpatches.Patch(color='lightblue', alpha=0.5, label='File extent')
    tile_legend = mpatches.Patch(facecolor='none', edgecolor='red', linestyle='--', 
                                 linewidth=2, label=f'Tile extent (with {tile_buffer}m buffer)')
    core_legend = mpatches.Patch(color='pink', alpha=0.3, label='Tile core (no buffer)')
    ax.legend(handles=[file_legend, tile_legend, core_legend], loc='upper right', fontsize=10)
    
    # Add statistics text box
    stats_text = f'Files: {len(file_extents)}\nTiles: {len(tiles)}\nTile size: {tile_length}m\nBuffer: {tile_buffer}m'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_png}")
    print(f"  - {len(file_extents)} files (blue)")
    print(f"  - {len(tiles)} tiles (red dashed)")
    
    # Optionally show plot
    # plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize file extents and generated tile extents"
    )
    parser.add_argument(
        "folder_path",
        type=Path,
        help="Path to folder containing LAZ/LAS files"
    )
    parser.add_argument(
        "--tile-length",
        type=float,
        default=100.0,
        help="Tile length in meters (default: 100)"
    )
    parser.add_argument(
        "--tile-buffer",
        type=float,
        default=5.0,
        help="Tile buffer in meters (default: 5)"
    )
    parser.add_argument(
        "--proj-crs",
        type=str,
        default="EPSG:32632",
        help="Projected CRS for tiling (default: EPSG:32632)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG file path (default: <folder>/tiles_visualization.png)"
    )
    args = parser.parse_args()
    
    if not args.folder_path.exists():
        print(f"ERROR: Folder not found: {args.folder_path}")
        sys.exit(1)
    
    if not args.folder_path.is_dir():
        print(f"ERROR: Path is not a directory: {args.folder_path}")
        sys.exit(1)
    
    if args.output is None:
        args.output = args.folder_path / "tiles_visualization.png"
    
    plot_extents(args.folder_path, args.tile_length, args.tile_buffer, 
                 args.output, args.proj_crs)


if __name__ == "__main__":
    main()

