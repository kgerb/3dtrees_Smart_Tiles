# Parameter Configuration Guide

The 3DTrees pipeline supports flexible parameter configuration via CLI, config files, and environment variables.

## Quick Start

### View Current Parameters

```bash
python run.py --show-params
```

### Override Individual Parameters

```bash
python run.py --task tile \
  --input_dir /path/to/input \
  --output_dir /path/to/output \
  --tile_length 150 \
  --resolution_1 0.03 \
  --workers 8
```

### Use Custom Config File

Create a custom config file (e.g., `my_config.py`):

```python
TILE_PARAMS = {
    'tile_length': 150,
    'tile_buffer': 10,
    'resolution_1': 0.03,
    'resolution_2': 0.15,
    'workers': 8,
}

MERGE_PARAMS = {
    'buffer': 15.0,
    'overlap_threshold': 0.4,
}
```

Then use it:

```bash
python run.py --task tile \
  --input_dir /path/to/input \
  --output_dir /path/to/output \
  --config my_config.py
```

## Priority Order

Parameters are applied in this order (highest priority last):

1. **Default values** (in `parameters.py`)
2. **Environment variables** (e.g., `TILE_LENGTH=150`)
3. **Config file** (`--config my_config.py`)
4. **CLI overrides** (`--param tile_length=150`)

## Parameter Reference

### TILE_PARAMS

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tile_length` | 100 | Tile size in meters |
| `tile_buffer` | 5 | Buffer overlap in meters |
| `threads` | 5 | Threads per COPC writer |
| `workers` | 4 | Number of parallel workers |
| `resolution_1` | 0.02 | First subsampling resolution (2cm) |
| `resolution_2` | 0.1 | Second subsampling resolution (10cm) |
| `grid_offset` | 1.0 | Grid offset from min coordinates |
| `skip_dimension_reduction` | False | Preserve all point dimensions (default: reduce to XYZ-only) |

### REMAP_PARAMS

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_resolution_cm` | 2 | Target resolution in cm |
| `workers` | 4 | Workers for KDTree queries |

### MERGE_PARAMS

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer` | 10.0 | Buffer distance for filtering (meters) |
| `overlap_threshold` | 0.3 | Overlap ratio for instance matching |
| `max_centroid_distance` | 3.0 | Max centroid distance to merge (meters) |
| `correspondence_tolerance` | 0.05 | Point correspondence tolerance (meters) |
| `max_volume_for_merge` | 4.0 | Max volume for small instance merge (m³) |
| `workers` | 4 | Number of parallel workers |

## Usage Examples

### Example 1: Large Tiles with More Workers

```bash
python run.py --task tile \
  --input_dir /data/input \
  --output_dir /data/output \
  --tile_length 200 \
  --tile_buffer 10 \
  --workers 16
```

### Example 2: Custom Resolutions

```bash
python run.py --task tile \
  --input_dir /data/input \
  --output_dir /data/output \
  --resolution_1 0.01 \
  --resolution_2 0.05
```

### Example 2b: Preserve All Point Dimensions

```bash
python run.py --task tile \
  --input_dir /data/input \
  --output_dir /data/output \
  --skip_dimension_reduction
```

Note: By default, the pipeline reduces point clouds to XYZ-only for ~37% size reduction. Use `--skip_dimension_reduction` to keep all attributes (intensity, classification, RGB, etc.).

### Example 3: Using Environment Variables

```bash
export TILE_LENGTH=150
export TILE_BUFFER=8
export TILE_WORKERS=12

python run.py --task tile \
  --input_dir /data/input \
  --output_dir /data/output
```

### Example 4: Multiple Overrides

```bash
python run.py --task merge \
  --subsampled_10cm_folder /data/10cm \
  --buffer 15.0 \
  --overlap_threshold 0.4 \
  --workers 16
```

### Example 5: Config File with Overrides

Create `production_params.py`:

```python
TILE_PARAMS = {
    'tile_length': 200,
    'tile_buffer': 10,
    'workers': 16,
}

MERGE_PARAMS = {
    'buffer': 15.0,
    'workers': 16,
}
```

Use it with additional overrides:

```bash
python run.py --task tile \
  --input_dir /data/input \
  --output_dir /data/output \
  --config production_params.py \
  --resolution_1 0.015
```

## Testing Parameters

Test your parameter configuration without running the pipeline:

```bash
# View defaults
python run.py --show-params

# View with config file
python run.py --show-params --config my_config.py

# View with overrides
python run.py --show-params --tile_length 150 --workers 8

# View with config + overrides
python run.py --show-params --config my_config.py --resolution_1 0.03
```

## Parameter Format

CLI parameters are passed directly as arguments:

```bash
# Direct parameter syntax
--tile_length 150
--resolution_1 0.03
--workers 8

# Multiple parameters
python run.py --task tile --input_dir /data --output_dir /out \
  --tile_length 150 --tile_buffer 10 --workers 8
```

## Tips

1. **Start with defaults**: Use `python run.py --show-params` to see current values
2. **Use config files for projects**: Create project-specific config files for consistent runs
3. **Use CLI for experimentation**: Quick parameter tweaks via direct CLI arguments
4. **Use environment variables for deployment**: Set system-wide defaults via env vars
5. **Combine approaches**: Config file for base settings + CLI overrides for experiments

## Available Parameters

All parameters can be passed as CLI arguments:

**Tile Parameters:** `--tile_length`, `--tile_buffer`, `--grid_offset`, `--threads`, `--workers`, `--resolution_1`, `--resolution_2`, `--skip_dimension_reduction`

**Remap Parameters:** `--target_resolution`, `--workers`

**Merge Parameters:** `--buffer`, `--overlap_threshold`, `--max_centroid_distance`, `--correspondence_tolerance`, `--max_volume_for_merge`, `--workers`



