#!/bin/bash

# Ensure two arguments are passed
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source-dir> <dest-dir>"
  exit 1
fi

SOURCE_DIR=$1
DEST_DIR=$2

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all PNG files in the source directory
for file in "$SOURCE_DIR"/*; do
  # Extract filename without extension
  filename=$(basename -- "$file")
  filename="${filename%.*}"

  # Quality between 0 and 31, good between 2-5, we are sticking with 3 
  ffmpeg -i "$file" -q:v 3 "$DEST_DIR/${filename}.jpg"
done

echo "Conversion complete."
