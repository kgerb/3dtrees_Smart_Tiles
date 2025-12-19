#!/usr/bin/env python3
"""
Get extent from tindex shapefile and compute tile bounds.
Similar to get_bounds.py but uses tindex instead of VPC.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import fiona
    from pyproj import Transformer
except ImportError as e:
    print(f"ERROR: Required package missing. Install with: pip install fiona pyproj")
    print(f"Error: {e}")
    exit(1)


def load_extent_from_tindex(tindex_path: Path):
    """Load extent from tindex shapefile.
    
    Returns:
        Tuple of (minx, miny, maxx, maxy), crs_string, is_projected
        
    The coordinates are returned in the native CRS of the data.
    is_projected indicates if coordinates are already in a projected CRS.
    """
    with fiona.open(tindex_path) as src:
        # Get CRS from file
        src_crs = str(src.crs) if src.crs else "EPSG:32632"
        
        # Get bounds of all features
        minx = miny = math.inf
        maxx = maxy = -math.inf
        
        feature_count = 0
        for feature in src:
            feature_count += 1
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = geom['coordinates'][0]
            elif geom['type'] == 'MultiPolygon':
                coords = [c for poly in geom['coordinates'] for c in poly[0]]
            else:
                continue
            
            xs, ys = zip(*coords)
            minx = min(minx, min(xs))
            miny = min(miny, min(ys))
            maxx = max(maxx, max(xs))
            maxy = max(maxy, max(ys))
        
        if feature_count == 0:
            bounds = src.bounds
            if bounds and bounds != (0.0, 0.0, 0.0, 0.0):
                minx, miny, maxx, maxy = bounds
            else:
                raise ValueError(f"No features found in tindex: {tindex_path}")
        
        # Detect if coordinates are already projected (values > 360 are clearly not lat/lon)
        is_projected = abs(minx) > 360 or abs(maxx) > 360 or abs(miny) > 360 or abs(maxy) > 360
        
        if is_projected and "4326" in src_crs:
            # CRS is misreported as WGS84, assume EPSG:32632 (UTM 32N)
            print(f"  Note: CRS reported as {src_crs} but coordinates appear projected", file=sys.stderr)
            print(f"  Assuming EPSG:32632 (UTM 32N)", file=sys.stderr)
            src_crs = "EPSG:32632"
        
        return (minx, miny, maxx, maxy), src_crs, is_projected


def build_tiles(minx, miny, maxx, maxy, length, buffer, align_to_grid=False, grid_offset=0.0):
    """Build tile grid.
    
    Args:
        minx, miny, maxx, maxy: Data extent bounds
        length: Tile size in meters
        buffer: Buffer size in meters
        align_to_grid: If True, snap to grid (floor to nearest tile_length multiple).
                       If False, start from actual data extent (more efficient coverage).
        grid_offset: Offset in meters to add to minx and miny before starting grid (default: 0.0)
    """
    if align_to_grid:
        # Grid-aligned: snap to multiples of tile_length (ensures consistent grid across datasets)
        start_x = math.floor((minx + grid_offset) / length) * length
        start_y = math.floor((miny + grid_offset) / length) * length
        end_x = math.ceil(maxx / length) * length
        end_y = math.ceil(maxy / length) * length
    else:
        # Data-aligned: start from actual data extent with offset (minimizes tiles, better coverage)
        start_x = minx + grid_offset
        start_y = miny + grid_offset
        end_x = math.ceil((maxx - minx) / length) * length + start_x
        end_y = math.ceil((maxy - miny) / length) * length + start_y

    tiles = []
    col = 0
    x = start_x
    while x < end_x:
        row = 0
        y = start_y
        while y < end_y:
            core_x = [x, x + length]
            core_y = [y, y + length]
            buffered_x = [core_x[0] - buffer, core_x[1] + buffer]
            buffered_y = [core_y[0] - buffer, core_y[1] + buffer]
            tiles.append(
                {
                    "col": col,
                    "row": row,
                    "core": [core_x, core_y],
                    "bounds": [buffered_x, buffered_y],
                }
            )
            row += 1
            y += length
        col += 1
        x += length

    grid_bounds = (
        start_x - buffer,
        end_x + buffer,
        start_y - buffer,
        end_y + buffer,
    )
    tiles.sort(key=lambda t: (t["col"], t["row"]))
    return tiles, grid_bounds


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute tile bounds for the tindex extent and emit helper values "
            "for PDAL pipelines."
        )
    )
    parser.add_argument("tindex_path", type=Path, help="Path to the tindex shapefile")
    parser.add_argument(
        "--tile-length", type=float, default=40.0, help="Tile core length (default: 40 m)"
    )
    parser.add_argument(
        "--tile-buffer", type=float, default=5.0, help="Tile buffer distance (default: 5 m)"
    )
    parser.add_argument(
        "--proj-crs",
        type=str,
        default="EPSG:32632",
        help="Projected CRS for tiling (default: EPSG:32632)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/home/kg281/data/output/pdal_experiments/tile_bounds_tindex.json"),
        help="Where to write the tile bounds JSON summary",
    )
    parser.add_argument(
        "--grid-offset",
        type=float,
        default=0.0,
        help="Offset in meters to add to minx and miny before starting grid (default: 0.0)",
    )
    args = parser.parse_args()

    (minx, miny, maxx, maxy), srs, is_projected = load_extent_from_tindex(args.tindex_path)
    
    # Determine the working CRS and coordinates
    if is_projected:
        # Coordinates are already projected, use directly
        proj_minx, proj_miny, proj_maxx, proj_maxy = minx, miny, maxx, maxy
        proj_crs = srs if srs != "EPSG:4326" else args.proj_crs
        
        # Calculate geographic bounds for info
        try:
            geo_transformer = Transformer.from_crs(proj_crs, "EPSG:4326", always_xy=True)
            corners = [
                geo_transformer.transform(minx, miny),
                geo_transformer.transform(maxx, maxy),
            ]
            geo_minx, geo_miny = corners[0]
            geo_maxx, geo_maxy = corners[1]
        except Exception:
            geo_minx, geo_miny, geo_maxx, geo_maxy = minx, miny, maxx, maxy
    else:
        # Coordinates are geographic, transform to projected
        geo_minx, geo_miny, geo_maxx, geo_maxy = minx, miny, maxx, maxy
        proj_crs = args.proj_crs
        
        proj_transformer = Transformer.from_crs("EPSG:4326", proj_crs, always_xy=True)
        corners_proj = [
            proj_transformer.transform(minx, miny),
            proj_transformer.transform(minx, maxy),
            proj_transformer.transform(maxx, miny),
            proj_transformer.transform(maxx, maxy),
        ]
        proj_xs, proj_ys = zip(*corners_proj)
        proj_minx, proj_maxx = min(proj_xs), max(proj_xs)
        proj_miny, proj_maxy = min(proj_ys), max(proj_ys)
    
    # Use data-aligned tiling
    tiles, grid_bounds = build_tiles(
        proj_minx, proj_miny, proj_maxx, proj_maxy, 
        args.tile_length, args.tile_buffer, 
        align_to_grid=False,
        grid_offset=args.grid_offset
    )

    summary = {
        "tindex": str(args.tindex_path),
        "tindex_srs": srs,
        "proj_srs": proj_crs,
        "proj_extent": {"minx": proj_minx, "miny": proj_miny, "maxx": proj_maxx, "maxy": proj_maxy},
        "geo_extent": {"minx": geo_minx, "miny": geo_miny, "maxx": geo_maxx, "maxy": geo_maxy},
        "tile_length": args.tile_length,
        "tile_buffer": args.tile_buffer,
        "grid_bounds": {
            "xmin": grid_bounds[0],
            "xmax": grid_bounds[1],
            "ymin": grid_bounds[2],
            "ymax": grid_bounds[3],
        },
        "tiles": tiles,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)

    crop_bounds = f"([{grid_bounds[0]},{grid_bounds[1]}],[{grid_bounds[2]},{grid_bounds[3]}])"
    reader_bounds = f"([{geo_minx},{geo_maxx}],[{geo_miny},{geo_maxy}])"

    print(f"tile_bounds_file={args.out}")
    print(f"tile_count={len(tiles)}")
    print(f"crop_bounds=\"{crop_bounds}\"")
    print(f"reader_bounds=\"{reader_bounds}\"")


if __name__ == "__main__":
    main()

