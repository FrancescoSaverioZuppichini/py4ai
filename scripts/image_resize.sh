#!/bin/bash

# Ensure five arguments are passed
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <source-dir> <dest-dir> <target-width> <target-height> <format>"
  echo "<format> should be 'jpg' for JPEG output or 'png' for uncompressed PNG output."
  exit 1
fi

SOURCE_DIR=$1
DEST_DIR=$2
TARGET_WIDTH=$3
TARGET_HEIGHT=$4
FORMAT=$5

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist."
  exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all files in the source directory
for file in "$SOURCE_DIR"/*; do
  # Extract filename without extension
  filename=$(basename -- "$file")
  filename="${filename%.*}"

  if [ "$FORMAT" = "jpg" ]; then
    # Compress and save as JPEG
    ffmpeg -i "$file" -vf "scale=$TARGET_WIDTH:$TARGET_HEIGHT" "$DEST_DIR/${filename}.jpg"
  elif [ "$FORMAT" = "png" ]; then
    # Save as uncompressed PNG
    ffmpeg -i "$file" -vf "scale=$TARGET_WIDTH:$TARGET_HEIGHT" "$DEST_DIR/${filename}.png"
  else
    echo "Unsupported format: $FORMAT. Please choose 'jpg' or 'png'."
    exit 1
  fi
done

echo "Conversion complete."
