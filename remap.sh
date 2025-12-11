#! /bin/bash

#input file
input_file=$1

pdal-parallelizer process-pipelines \
-c /home/kg281/projects/3dtrees_smart_tile/remap_config.json \
-it single \
-ts 100 100 \
-nw 2 