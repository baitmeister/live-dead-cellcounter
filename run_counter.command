#!/bin/bash
# Double-click to review the TIFFs in the folder containing this program folder.
#
# Expected layout:
#     image-folder/
#       *.tif
#       cellcounter/
#         run_counter.command
#
# An explicit image-folder argument overrides the parent-folder default.
PROGRAM_DIR="$(cd "$(dirname "$0")" && pwd)" || exit 1
IMAGE_DIR="$(cd "$PROGRAM_DIR/.." && pwd)" || exit 1

if [ "$#" -gt 0 ]; then
    TARGET_DIR="$(cd "$1" 2>/dev/null && pwd)" || {
        echo "Image folder not found: $1" >&2
        exit 1
    }
else
    TARGET_DIR="$IMAGE_DIR"
fi

# Import the cellcounter package from its parent, but pass the image folder to
# the app so review_state.json and both viability CSVs stay beside the TIFFs.
cd "$IMAGE_DIR" || exit 1
exec python3 -m cellcounter.app "$TARGET_DIR"
