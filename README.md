# 3DTrees Smart Tiling Pipeline

Point cloud processing pipeline for 3D tree segmentation: tiling, subsampling, remapping, and merging with instance matching.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TILE TASK                                         │
│  Input LAZ → COPC Conversion → Spatial Index → Tiling → Subsampling        │
│                                                    ↓                        │
│                                          2cm and 10cm outputs               │
└─────────────────────────────────────────────────────────────────────────────┘
                                         ↓
                              [External Segmentation]
                                         ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MERGE TASK                                        │
│  Segmented 10cm → Remap to 2cm → Merge Tiles → Instance Matching           │
│                                                    ↓                        │
│                                          Unified point cloud                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start with Docker

### Build the Image

```bash
docker build -t 3dtrees-smart-tile .
```

### Tile Task

Convert, tile, and subsample point clouds:

```bash
docker run -v /path/to/data:/data 3dtrees-smart-tile \
    --task tile \
    --input_dir /data/input \
    --output_dir /data/output
```

### Merge Task

Remap predictions and merge tiles:

```bash
docker run -v /path/to/data:/data 3dtrees-smart-tile \
    --task merge \
    --subsampled_10cm_folder /data/output/tiles_100m/subsampled_10cm
```

### View Parameters

```bash
docker run 3dtrees-smart-tile --show-params
```

### Interactive Shell

```bash
docker run -it --entrypoint /bin/bash 3dtrees-smart-tile
```

## Pipeline Details

### Task: `tile`

1. **COPC Conversion**: Convert to COPC format (optionally strip dimensions to XYZ-only for size reduction)
2. **Build Spatial Index**: Create tindex for efficient querying
3. **Tiling**: Create overlapping tiles with configurable buffer
4. **Subsample Resolution 1**: Default 2cm
5. **Subsample Resolution 2**: Default 10cm

**Note:** By default, dimensions are reduced to XYZ-only for ~37% size reduction. Use `--skip_dimension_reduction` to preserve all point attributes.

### Task: `merge`

1. **Remap**: Transfer predictions from 10cm back to 2cm resolution
2. **Buffer Filtering**: Remove instances in overlap zones
3. **Cross-tile Matching**: Merge instances across tile boundaries
4. **Deduplication**: Remove duplicate points from overlaps
5. **Small Volume Merging**: Merge small fragments to nearby trees

## Parameters

Override defaults via command line:

```bash
docker run -v /data:/data 3dtrees-smart-tile \
    --task tile \
    --input_dir /data/input \
    --output_dir /data/output \
    --tile_length 150 \
    --tile_buffer 10 \
    --resolution_1 0.03 \
    --resolution_2 0.1 \
    --workers 8
```

See [README_PARAMETERS.md](README_PARAMETERS.md) for full parameter documentation.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tile_length` | 100 | Tile size in meters |
| `--tile_buffer` | 5 | Buffer overlap in meters |
| `--resolution_1` | 0.02 | First subsampling resolution (2cm) |
| `--resolution_2` | 0.1 | Second subsampling resolution (10cm) |
| `--skip_dimension_reduction` | False | Preserve all point dimensions (disable XYZ-only reduction) |
| `--workers` | 4 | Number of parallel workers |
| `--buffer` | 10.0 | Merge buffer distance (meters) |
| `--overlap_threshold` | 0.3 | Instance matching threshold |

## Local Installation (without Docker)

### Requirements

- Python 3.10+
- PDAL >= 2.5
- pdal_wrench (from [PDAL/wrench](https://github.com/PDAL/wrench))
- GDAL >= 3.0

### Conda Environment

```bash
# Create environment
mamba create -n 3dtrees -c conda-forge \
    python=3.10 \
    pdal=2.6 \
    python-pdal \
    laspy \
    lazrs-python \
    numpy \
    scipy \
    matplotlib \
    fiona \
    pyproj \
    geopandas

# Activate
conda activate 3dtrees

# Build pdal_wrench from source
git clone https://github.com/PDAL/wrench.git
cd wrench
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo cp pdal_wrench /usr/local/bin/
```

### Run Locally

```bash
# Tile task
python src/run.py --task tile --input_dir /path/to/input --output_dir /path/to/output

# Merge task
python src/run.py --task merge --subsampled_10cm_folder /path/to/10cm

# Show parameters
python src/run.py --show-params
```

## Input Requirements

### Tile Task

- LAZ or LAS files in the input directory
- Files should be in a projected CRS (e.g., UTM)

### Merge Task

- Segmented tiles with `PredInstance` attribute
- Tile naming convention: `c{col}_r{row}*.laz`

## Output Structure

```
output_dir/
├── copc_xyz/              # XYZ-only COPC files (or copc_full/ if --skip_dimension_reduction)
├── tiles_100m/            # Tiled point clouds
│   ├── c00_r00.copc.laz
│   ├── c00_r01.copc.laz
│   ├── subsampled_2cm/    # 2cm resolution
│   └── subsampled_10cm/   # 10cm resolution
├── tindex_100m.gpkg       # Spatial index
└── overview_copc_tiles.png # Visualization
```

## Project Structure

```
3dtrees_smart_tile/
├── src/                          # Python source code
│   ├── run.py                    # Main orchestrator
│   ├── main_tile.py              # Tiling pipeline
│   ├── main_subsample.py         # Subsampling pipeline
│   ├── main_remap.py             # Prediction remapping
│   ├── main_merge.py             # Merge wrapper
│   ├── merge_tiles.py            # Core merge implementation
│   ├── parameters.py             # Parameter configuration
│   ├── filter_buffer_instances.py
│   ├── prepare_tile_jobs.py
│   ├── get_bounds_from_tindex.py
│   ├── plot_tiles_and_copc.py
│   └── example_config.py         # Example parameter config
├── Dockerfile
├── docker-compose.yml
├── README.md
└── README_PARAMETERS.md
```

## Dependencies

Core dependencies (installed via Dockerfile or conda):

- **PDAL** >= 2.5 - Point cloud processing
- **pdal_wrench** - Parallel PDAL operations ([GitHub](https://github.com/PDAL/wrench))
- **laspy** - LAS/LAZ file handling
- **scipy** - KDTree for spatial queries
- **numpy** - Array operations
- **fiona** - Vector file handling
- **pyproj** - CRS transformations
- **matplotlib** - Visualization

## License

MIT License
