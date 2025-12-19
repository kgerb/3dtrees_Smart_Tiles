# =============================================================================
# 3DTrees Smart Tile Pipeline - Dockerfile
# =============================================================================
# Single-stage build with all dependencies
# Builds pdal_wrench from source (requires PDAL >= 2.5, GDAL >= 3.0)
# =============================================================================

FROM condaforge/miniforge3:latest

# Install all dependencies (runtime + build tools)
# Per https://github.com/PDAL/wrench: needs PDAL >= 2.5 and GDAL >= 3.0
# Let mamba resolve compatible versions (don't pin specific versions)
RUN mamba install -n base -c conda-forge \
    python=3.10 \
    pdal \
    python-pdal \
    gdal \
    laspy \
    lazrs-python \
    numpy \
    scipy \
    matplotlib-base \
    fiona \
    pyproj \
    geopandas \
    cmake \
    make \
    cxx-compiler \
    git \
    -y && \
    mamba clean --all -y

# Build pdal_wrench from source
WORKDIR /tmp
RUN git clone --depth 1 https://github.com/PDAL/wrench.git && \
    cd wrench && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    cp pdal_wrench /usr/local/bin/ && \
    chmod +x /usr/local/bin/pdal_wrench && \
    cd /tmp && rm -rf wrench

# Verify installations
RUN pdal --version && pdal_wrench --help | head -5

# ===========================================
# Setup project
# ===========================================
WORKDIR /src

# Copy Python scripts from src/ folder
COPY src/ /src/

# Create data directories
RUN mkdir -p -m 777 /in /out

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/src
ENV PATH="/usr/local/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib"
# Fix PROJ database path
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
