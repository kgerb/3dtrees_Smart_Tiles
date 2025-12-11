---
name: Remap Predictions to Input
overview: Modify run_merge_species.sh to call the source remapping script after merging tiles, ensuring predictions are mapped back to original input files rather than tiles.
todos:
  - id: update-remap-default
    content: Update default source folder in run_remap_to_source.sh to point to input folder
    status: pending
  - id: add-remap-call
    content: Add call to run_remap_to_source.sh at end of run_merge_species.sh
    status: pending
---

# Remap Merged Predictions to Original Input Files

## Current Issue

The script [`run_merge_species.sh`](/home/kg281/projects/3dtrees_smart_tile/run_merge_species.sh) only merges tiles but does not call the remapping step to map predictions back to the original input files.

## Proposed Changes

### 1. Update [`run_remap_to_source.sh`](/home/kg281/projects/3dtrees_smart_tile/run_remap_to_source.sh)

Change the default source folder from:

```bash
DEFAULT_SOURCE_FOLDER="/home/kg281/data/kaltenborn25_lidar/bechstedt"
```

to:

```bash
DEFAULT_SOURCE_FOLDER="/home/kg281/data/kaltenborn25_lidar/bechstedt/input"
```

### 2. Update [`run_merge_species.sh`](/home/kg281/projects/3dtrees_smart_tile/run_merge_species.sh)

Add a call to `run_remap_to_source.sh` at the end of the script, passing the merged LAZ file and input folder:

```bash
# After merge_species_tiles.py call, add:
INPUT_SOURCE_FOLDER="/home/kg281/data/kaltenborn25_lidar/bechstedt/input"
OUTPUT_WITH_PREDICTIONS="/home/kg281/data/kaltenborn25_lidar/bechstedt/input_with_predictions"

"$SCRIPT_DIR/run_remap_to_source.sh" \
    "$OUTPUT_LAZ" \
    "$INPUT_SOURCE_FOLDER" \
    "$OUTPUT_WITH_PREDICTIONS" \
    "$NUM_THREADS"
```

## Result

After running `run_merge_species.sh`, the predictions will be:

1. Merged into `merged_detailview.laz` (as before)
2. Remapped to each original input file, saved to `input_with_predictions/` folder with files like:

   - `10-019_cloud0_with_predictions.laz`
   - `9-018_cloudb1146f2d8aac575d_with_predictions.laz`