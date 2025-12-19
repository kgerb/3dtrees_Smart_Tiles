"""
Example configuration file for 3DTrees pipeline.

Copy this file and modify parameters as needed, then use:
    python run.py --task tile --config my_config.py --input_dir ... --output_dir ...
"""

# Tile task parameters - for large areas with higher resolution
TILE_PARAMS = {
    'tile_length': 150,           # Larger tiles (150m instead of default 100m)
    'tile_buffer': 8,             # Larger buffer (8m instead of 5m)
    'threads': 8,                 # More threads for COPC writing
    'workers': 8,                 # More parallel workers
    'resolution_1': 0.01,         # Finer resolution 1 (1cm instead of 2cm)
    'resolution_2': 0.05,         # Resolution 2 (5cm instead of 10cm)
    'grid_offset': 1.0,           # Grid offset (standard)
}

# Remap task parameters - for higher precision
REMAP_PARAMS = {
    'target_resolution_cm': 1,    # Target 1cm instead of 2cm
    'workers': 16,                # More workers for KDTree
}

# Merge task parameters - more conservative merging
MERGE_PARAMS = {
    'buffer': 12.0,               # Larger buffer zone (12m instead of 10m)
    'overlap_threshold': 0.4,     # Higher threshold (40% instead of 30%)
    'max_centroid_distance': 2.5, # Stricter distance requirement
    'correspondence_tolerance': 0.03, # Tighter correspondence
    'max_volume_for_merge': 3.0,  # Smaller volume threshold
    'workers': 16,                # More parallel workers
}

