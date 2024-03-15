import shutil
import os
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(
    description="Copy an image 1000 times to a destination folder."
)

# Add arguments for source image path and destination folder
parser.add_argument("source_image_path", type=str, help="Path to the source image.")
parser.add_argument(
    "destination_folder",
    type=str,
    help="Path to the destination folder where images will be copied.",
)

# Parse the command-line arguments
args = parser.parse_args()

# Use the provided arguments for source image path and destination folder
source_image_path = args.source_image_path
destination_folder = args.destination_folder

# Copy the source image 1000 times to the destination folder
for i in range(1000):
    destination_image_path = os.path.join(destination_folder, f"image_copy_{i+1}.jpg")
    shutil.copy2(source_image_path, destination_image_path)
