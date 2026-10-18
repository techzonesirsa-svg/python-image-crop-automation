import os
import cv2
import numpy as np

def crop_exact_yellow_bottom(input_folder, output_folder):
    # Ensure output folder path is created
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"No valid images found in '{input_folder}'.")
        return

    print(f"Scanning and processing {len(files)} files...")

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Error reading {file_name}. Skipping.")
            continue

        height, width, _ = img.shape
        
        # 1. Convert to HSV for accurate yellow color isolation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define tighter/vibrant bounds targeting the bright yellow banner
        lower_yellow = np.array([15, 120, 120])
        upper_yellow = np.array([35, 255, 255])
        
        # Create a binary map where yellow pixels are white (255) and others black (0)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 2. Analyze the bottom 20% profile to protect any yellow objects in the top area
        start_scan_y = int(height * 0.8)
        bottom_mask_profile = yellow_mask[start_scan_y:height, :]
        
        # Calculate how many yellow pixels exist across each horizontal row
        row_counts = np.sum(bottom_mask_profile > 0, axis=1)
        
        # 3. Scan upward from the very bottom row to locate where the block of yellow ends
        crop_line_relative = len(row_counts) - 1
        found_edge = False
        
        # Expect at least 30% of the row width to be yellow to qualify as the strip line
        min_yellow_pixels_per_row = int(width * 0.3)

        for i in range(len(row_counts) - 1, -1, -1):
            if row_counts[i] < min_yellow_pixels_per_row:
                crop_line_relative = i
                found_edge = True
                break
                
        # Calculate absolute crop y-coordinate
        if found_edge:
            crop_y = start_scan_y + crop_line_relative
        else:
            # Fallback safe estimate if an image completely lacks a strip
            crop_y = int(height * 0.915)

        # 4. Final safety clamp and cut execution
        crop_y = max(int(height * 0.75), min(crop_y, height))
        cropped_img = img[0:crop_y, 0:width]

        # Save keeping matching input naming parameters
        cv2.imwrite(output_path, cropped_img)
        print(f"Cropped at row {crop_y}: {file_name}")

    print("\nSmart color scanning batch job successfully finalized!")

# --- Configurations ---
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"

crop_exact_yellow_bottom(INPUT_DIR, OUTPUT_DIR)
