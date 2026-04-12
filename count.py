import os
import sys

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.svg', '.ico', '.heic', '.heif', '.raw', '.cr2',
    '.nef', '.arw', '.dng', '.psd', '.avif'
}

def count_images(folder_path):
    if not os.path.exists(folder_path):
        print(f"Error: folder '{folder_path}' does not exist.")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a folder.")
        return

    count = 0
    for file in os.listdir(folder_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            count += 1

    print(f"Folder : {folder_path}")
    print(f"Images found : {count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        folder = input("Enter folder path: ")
    else:
        folder = sys.argv[1]

    count_images(folder)