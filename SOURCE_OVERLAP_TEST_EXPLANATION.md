# Source File Overlap Density Test Explanation

## Purpose

This test verifies that when **two source COPC files overlap**, the merged tile contains **points from both files**, resulting in **higher point density** in the overlap region compared to each individual source file.

## How It Works

### Step 1: Identify Overlapping Source Files
- Read all source COPC files from the tindex
- Get bounds for each COPC file
- Find pairs of source files that have overlapping extents

### Step 2: For Each Overlapping Pair
1. **Calculate overlap region**: The area where both source files cover the same extent
2. **Count points from File 1** in the overlap region
3. **Count points from File 2** in the overlap region  
4. **Count points in merged tile** that covers this overlap region

### Step 3: Compare Densities
- **Expected total**: `count1 + count2` (if both files are merged)
- **Expected density**: `(count1 + count2) / overlap_area`
- **Actual merged count**: Points in the merged tile's overlap region
- **Actual merged density**: `merged_count / overlap_area`

### Step 4: Evaluate Results
- ✓ **PASS**: Merged tile has ≥95% of expected total points
- ✓ **GOOD**: Merged tile has ≥10% more points than either single file
- ⚠️ **WARNING**: Merged tile has similar points to single file (may not be merging)
- ❌ **FAIL**: Merged tile has fewer points than single file

## Visual Example

```
Source Files:
┌─────────────┐     ┌─────────────┐
│  File A     │     │  File B     │
│             │     │             │
│  [Points]   │     │  [Points]   │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └───────┬───────────┘
               │
         ┌─────▼─────┐
         │  OVERLAP  │  ← Both files contribute here
         │  REGION   │
         └─────┬─────┘
               │
               ▼
         ┌───────────┐
         │ Merged    │
         │ Tile      │  ← Should have points from A + B
         │           │
         └───────────┘
```

## Expected Behavior

### ✓ Correct (Merge Filter Working)
```
Overlap region: 1000 m²
- File 1 points: 500,000 (density: 500 pts/m²)
- File 2 points: 300,000 (density: 300 pts/m²)
- Expected total: 800,000 (density: 800 pts/m²)
- Merged tile: 795,000 (density: 795 pts/m²)
✓ EXCELLENT: Merged tile has ~99% of expected points
```

### ❌ Incorrect (Merge Filter Not Working)
```
Overlap region: 1000 m²
- File 1 points: 500,000 (density: 500 pts/m²)
- File 2 points: 300,000 (density: 300 pts/m²)
- Expected total: 800,000 (density: 800 pts/m²)
- Merged tile: 300,000 (density: 300 pts/m²)
❌ FAIL: Only File 2's points are present (last file overwrote first)
```

## Running the Test

```bash
conda activate retile_env

python3 check_source_overlap_density.py \
    <tiles_dir> \
    <tile_jobs_file> \
    <tindex_file>
```

Example:
```bash
python3 check_source_overlap_density.py \
    /home/kg281/data/kaltenborn25_lidar/ULS_tiles_test/tiles_100m \
    /home/kg281/data/kaltenborn25_lidar/ULS_tiles_test/tile_jobs_100m.txt \
    /home/kg281/data/kaltenborn25_lidar/ULS_tiles_test/ULS_tiles_100m_tindex.gpkg
```

## Key Takeaway

This test ensures that **overlapping source files are properly merged**, not overwritten. When two source files overlap, the merged tile should contain **more points** (ideally the sum) than either individual file in that overlap region.






