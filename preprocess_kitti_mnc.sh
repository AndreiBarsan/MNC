#!/usr/bin/env bash
# Credits:
#   * Nikolaus Mayer from Uni Freiburg for inspiring me to Dockerize my code
#     after seeing just how amazingly easy it was to work with the dockerized
#     versions of DispNet an FlowNet developed by him and the vision lab at
#     the University of Freiburg.

# TODO(andreib): Separate script for downloading and unzipping data. Make this script call if if 'dataset-root' not given.

# TODO-LOW(andreib): Check for minimum driver version since that's something
# which can't be abstracted by Docker for now.
# TODO-LOW(andreib): Support CPU-only inference. May not be supported by the
# custom MNC layers, which may only have CUDA implementations IIRC.
# TODO-LOW(andreib): Makefile to make things even easier.

set -eu

usage() {
    cat >&2 <<EOF
Usage: $0 <dataset> <dataset-root> <split> <sequence-id>

A helper script for preprocessing KITTI data for DynSLAM.

Arguments:
   dataset      "kitti-odometry" or "kitti-tracking"
   dataset-root The root directory of the dataset
   split        training or testing
   sequence-id  The ID of the sequence to process.

Example:
    $0 kitti-tracking training ~/my-data/my-kitti/tracking 7
EOF
}

fail() {
  ret="$?"
  # echo first argument in red
  printf "\e[31m✘ ${1}"
  # reset colours back to normal
  printf "\033[0m"
  echo
  exit "$ret"
}

if [[ "$#" -lt 4 ]]; then
    usage;
    exit 1
fi

if ! hash nvidia-docker >/dev/null 2>&1; then
    echo >&2 "nvidia-docker is required"
    exit 2
fi

if ! [[ -f Dockerfile ]]; then
    echo >&2 "Please run this script from the same directory as the MNC"
    echo >&2 "Dockerfile."
    exit 3
fi

# TODO(andreib): Validate
DATASET="$1"
DATASET_ROOT="$2"
SPLIT="$3"
SEQUENCE_ID="$4"

# TODO(andrei): Support both tracking and odometry.

# TODO pass kitti dir location to image as mount
nvidia-docker build -t mnc-docker-img . || fail "Could not build Docker image."
nvidia-docker run -t mnc-docker-img \
    bash -c \
    'cd /opt/mnc/caffe-mnc && ls -lah && nvidia-smi' ||
    fail "Docker sanity check failed"


#for id in $(seq -f '%04g' 0 20); do
id="$(printf '%04d' $SEQUENCE_ID)"
printf "\n\t%s\n\n" "Processing sequence [$id]..."

SEQUENCE_IN="training/image_02/$id"
SEQUENCE_OUT="training/seg_image_02/$id"

# TODO(andrei): Check if SEQUENCE_OUT exists and has first and last segmented frames and skip.

# Note: the tool is called 'demo.py' for historical reasons.
nvidia-docker run \
    --mount src="$DATASET_ROOT",target=/data/kitti,type=bind \
    -t mnc-docker-img \
    bash -c "cd /opt/mnc && mkdir -p /data/kitti/$SEQUENCE_OUT && CUDA_VISIBLE_DEVICES=0 tools/demo.py --input /data/kitti/$SEQUENCE_IN --output /data/kitti/$SEQUENCE_OUT"
#done
