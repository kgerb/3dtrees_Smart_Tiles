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
        # Get CRS from file, default to EPSG:32630 if missing
        if src.crs:
            src_crs = str(src.crs)
        else:
            src_crs = "EPSG:32630"
            print(f"  ⚠ Warning: No CRS found in tindex, defaulting to {src_crs}", file=sys.stderr)
        
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

        # Check if CRS is projected or geographic based on EPSG code
        # EPSG:4326 is geographic (WGS84 lat/lon)
        # EPSG:326XX or EPSG:327XX are UTM projected CRS
        is_geographic = src_crs == "EPSG:4326"
        is_projected = not is_geographic and ("EPSG:32" in src_crs or "UTM" in src_crs.upper())

        # Also check if coordinate values are clearly projected (> 360 degrees)
        if not is_projected:
            is_projected = abs(minx) > 360 or abs(maxx) > 360 or abs(miny) > 360 or abs(maxy) > 360

        # Only check for coordinate swapping if CRS is geographic (EPSG:4326)
        # For projected CRS (UTM), coordinates are in meters and don't need swapping
        if is_geographic and not is_projected:
            # Check if coordinates might be swapped (lat/lon instead of lon/lat)
            # Valid latitude range is -90 to 90, valid longitude range is -180 to 180
            # If we see values outside these ranges, coordinates might be swapped
            x_in_lat_range = -90 <= minx <= 90 and -90 <= maxx <= 90
            y_in_lon_range = -180 <= miny <= 180 and -180 <= maxy <= 180
            x_in_lon_range = -180 <= minx <= 180 and -180 <= maxx <= 180
            y_in_lat_range = -90 <= miny <= 90 and -90 <= maxy <= 90

            # If x values look like lat and y values look like lon, swap them
            if x_in_lat_range and y_in_lon_range and not (x_in_lon_range and y_in_lat_range):
                print(
                    f"  Note: Coordinates appear to be in (lat, lon) order instead of (lon, lat). "
                    f"Swapping coordinates.",
                    file=sys.stderr
                )
                minx, miny, maxx, maxy = miny, minx, maxy, maxx

        # Detect if coordinates are suspiciously small for geographic coordinates
        # Very small coordinates (0-10 range) labeled as EPSG:4326 are likely test/placeholder data
        # or already projected (meters) but mislabeled
        x_span = abs(maxx - minx)
        y_span = abs(maxy - miny)
        very_small_range = (x_span <= 10.0 and y_span <= 10.0 and
                          abs(minx) <= 10.0 and abs(miny) <= 10.0 and
                          abs(maxx) <= 10.0 and abs(maxy) <= 10.0)
        
        # If coordinates are very small and CRS is 4326, they're likely test/placeholder data
        if very_small_range and "4326" in src_crs and not is_projected:
            raise ValueError(
                f"Coordinates appear to be test/placeholder data: "
                f"bounds=({minx}, {miny}, {maxx}, {maxy}), span=({x_span:.2f}°, {y_span:.2f}°). "
                f"CRS is reported as {src_crs} but coordinates are in the 0-10 range, "
                f"which would create an unreasonably large tiling area (~1000km x 1000km) when transformed. "
                f"Please check the tindex file and COPC files - they may contain invalid test data. "
                f"If these are real coordinates, they may already be in a projected CRS (meters) "
                f"and the CRS needs to be corrected in the tindex file."
            )
        
        if is_projected and "4326" in src_crs:
            # CRS is misreported as WGS84, assume EPSG:32630 (UTM 30N)
            print(f"  ⚠ Warning: CRS reported as {src_crs} but coordinates appear projected", file=sys.stderr)
            print(f"  Assuming EPSG:32630 (UTM 30N)", file=sys.stderr)
            src_crs = "EPSG:32630"
        
        return (minx, miny, maxx, maxy), src_crs, is_projected


def get_utm_zone(lon, lat):
    """Get UTM zone EPSG code for a given longitude and latitude.
    
    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees
        
    Returns:
        EPSG code string (e.g., 'EPSG:32632' for UTM 32N)
    """
    # UTM zones are 6 degrees wide, numbered 1-60
    # Zone number = floor((lon + 180) / 6) + 1
    zone = int(math.floor((lon + 180) / 6)) + 1
    
    # Northern hemisphere (lat >= 0) uses 32600 + zone
    # Southern hemisphere (lat < 0) uses 32700 + zone
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    
    return f"EPSG:{epsg}"


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
    # Validate inputs - check for infinity or NaN
    if not all(math.isfinite(v) for v in [minx, miny, maxx, maxy]):
        raise ValueError(
            f"Invalid bounds detected (infinity or NaN): "
            f"minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}. "
            f"This usually indicates the data is outside the projection zone."
        )
    
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
        # Check for overflow before calling math.ceil
        x_range = maxx - minx
        y_range = maxy - miny
        if not math.isfinite(x_range) or not math.isfinite(y_range):
            raise ValueError(
                f"Invalid range calculated: x_range={x_range}, y_range={y_range}. "
                f"Bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"
            )
        end_x = math.ceil(x_range / length) * length + start_x
        end_y = math.ceil(y_range / length) * length + start_y
    
    # Estimate number of tiles and warn if excessive
    num_tiles_x = int(math.ceil((end_x - start_x) / length))
    num_tiles_y = int(math.ceil((end_y - start_y) / length))
    total_tiles = num_tiles_x * num_tiles_y
    
    # Warn if creating too many tiles (more than 1 million)
    MAX_TILES = 1000000
    if total_tiles > MAX_TILES:
        raise ValueError(
            f"Would create {total_tiles:,} tiles ({num_tiles_x} x {num_tiles_y}), "
            f"which exceeds the maximum of {MAX_TILES:,}. "
            f"This usually indicates the data extent is too large or the tile size is too small. "
            f"Bounds: minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}, "
            f"tile_length={length}m. "
            f"Consider using a larger tile size or splitting the data into smaller regions."
        )

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
        default="EPSG:32630",
        help="Projected CRS for tiling (default: EPSG:32630 - WGS 84 / UTM zone 30N)",
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
        # Use detected CRS if valid, otherwise fallback to default EPSG:32630
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
        
        # Calculate centroid to determine appropriate UTM zone
        centroid_lon = (minx + maxx) / 2.0
        centroid_lat = (miny + maxy) / 2.0
        
        # Try the specified CRS first, but auto-detect UTM zone if it fails
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
        
        # Check if transformation produced invalid values
        if not all(math.isfinite(v) for v in [proj_minx, proj_miny, proj_maxx, proj_maxy]):
            # Check if data spans multiple UTM zones (more than ~6 degrees longitude)
            lon_range = abs(maxx - minx)
            if lon_range > 60.0:
                # Data spans more than 60 degrees - too wide for any single projection
                raise ValueError(
                    f"Data spans {lon_range:.1f} degrees longitude, which is too wide for tiling. "
                    f"Geographic bounds: ({minx}, {miny}, {maxx}, {maxy}). "
                    f"Please split the data into smaller regions (e.g., by longitude) before tiling, "
                    f"or use a much larger tile size."
                )
            elif lon_range > 10.0:
                # Data spans multiple zones, use World Mercator (EPSG:3857) but warn about tile count
                print(
                    f"  Warning: Data spans {lon_range:.1f} degrees longitude. "
                    f"Transformation to {proj_crs} produced invalid values. "
                    f"Using World Mercator (EPSG:3857) instead.",
                    file=sys.stderr
                )
                print(
                    f"  Note: Wide extents may produce many tiles. Consider using a larger tile size.",
                    file=sys.stderr
                )
                proj_crs = "EPSG:3857"
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
            else:
                # Try auto-detecting UTM zone based on centroid
                auto_utm = get_utm_zone(centroid_lon, centroid_lat)
                if auto_utm != proj_crs:
                    print(
                        f"  Warning: Transformation to {proj_crs} produced invalid values. "
                        f"Trying auto-detected UTM zone: {auto_utm}",
                        file=sys.stderr
                    )
                    proj_crs = auto_utm
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
            
            # If still invalid, raise an error
            if not all(math.isfinite(v) for v in [proj_minx, proj_miny, proj_maxx, proj_maxy]):
                raise ValueError(
                    f"Could not transform coordinates to projected CRS. "
                    f"Geographic bounds: ({minx}, {miny}, {maxx}, {maxy}), "
                    f"Centroid: ({centroid_lon}, {centroid_lat}), "
                    f"Longitude range: {lon_range:.1f} degrees. "
                    f"Tried CRS: {args.proj_crs} and {proj_crs}. "
                    f"Projected bounds: ({proj_minx}, {proj_miny}, {proj_maxx}, {proj_maxy})"
                )
    
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
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

