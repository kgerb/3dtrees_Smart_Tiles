# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# Single-stage build with all dependencies
# Builds pdal_wrench from source (requires PDAL >= 2.5, GDAL >= 3.0)
# =============================================================================

FROM condaforge/miniforge3:latest

# Install system dependencies using mamba (PDAL, untwine, GDAL, build tools)
# PDAL: Required for pdal command-line tool (used in main_tile.py, main_subsample.py)
# untwine: Command-line tool for COPC conversion (used in main_tile.py)
# gdal: Required for fiona (used in get_bounds_from_tindex.py, plot_tiles_and_copc.py)
# cmake, make, cxx-compiler: Required for building python-pdal (if needed)
RUN mamba install -n base -c conda-forge \
    python=3.10 \
    pdal \
    untwine \
    gdal \
    cmake \
    make \
    cxx-compiler \
    -y && \
    mamba clean --all -y

# Install uv
RUN pip install --no-cache-dir uv

# Install Python dependencies using uv
# lazrs: LAZ compression backend for laspy (laspy.LazBackend.Lazrs / LazrsParallel)
# laspy: Used extensively for reading/writing LAZ/LAS files; pass laz_backend=... when reading/writing LAZ
# numpy: Used extensively for array operations
# scipy: Used for cKDTree in merge_tiles.py and main_remap.py
# matplotlib: Used only in plot_tiles_and_copc.py (visualization) - PyPI name is 'matplotlib', not 'matplotlib-base'
# fiona: Used in get_bounds_from_tindex.py and plot_tiles_and_copc.py for reading shapefiles
# pyproj: Used in get_bounds_from_tindex.py, prepare_tile_jobs.py, plot_tiles_and_copc.py for CRS transformations
# pydantic, pydantic-settings: Used in parameters.py for configuration
# Note: geopandas removed - not used anywhere
# Note: python-pdal removed - PDAL is only called via subprocess, not imported
# Note: untwine is installed via mamba (conda-forge), not a Python package
RUN uv pip install --system \
    laspy \
    lazrs \
    numpy \
    scipy \
    matplotlib \
    fiona \
    pyproj \
    pydantic \
    pydantic-settings

# Verify installations
RUN pdal --version && \
    untwine --version

# ===========================================
# Setup project
# ===========================================
WORKDIR /src

# Copy Python scripts from src/ folder
COPY src/ /src/

# Create a non-root user for running the application
# Create data directories with proper permissions (owned by appuser)
RUN mkdir -p /in /out /src/out
RUN chmod -R 755 /in /out /src/out

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/src
ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib"
# Fix PROJ database path (conda location)
ENV PROJ_DATA="/opt/conda/share/proj"
ENV PROJ_LIB="/opt/conda/share/proj"
# Fix matplotlib config directory (writable location)
ENV MPLCONFIGDIR="/tmp/matplotlib"



# Set entrypoint to python run.py
# ===========================================
# Usage Examples:
# ===========================================
# Build:
#   docker build -t 3dtrees-smart-tile .
#
# Tile task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task tile --input_dir /data/input --output_dir /data/output
#
# Merge task:
#   docker run -v /path/to/data:/data 3dtrees-smart-tile \
#       --task merge --subsampled_10cm_folder /data/10cm
#
# Show parameters:
#   docker run 3dtrees-smart-tile --show-params
#
# Interactive shell:
#   docker run -it --entrypoint /bin/bash 3dtrees-smart-tile
# ===========================================
