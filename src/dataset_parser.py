import os
import json
from PIL import Image, ImageFile
from pathlib import Path

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Configuration Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Pointing to the validation files now
RAW_IMG_DIR = PROJECT_ROOT / "data" / "raw" / "vehiDE-Data" / "validation"
JSON_PATH = PROJECT_ROOT / "data" / "raw" / "vehiDE-Data" / "0Val_via_annos.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "val"

def parse_vehide_json():
    """Parses VehiDE JSON format, crops damages, and sorts them by class."""
    
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    processed_count = 0
    error_count = 0
    class_counts = {}

    print(f"Found {len(data)} images in JSON. Starting processing...")

    for img_key, img_data in data.items():
        filename = img_data.get("name")
        regions = img_data.get("regions", [])
        
        if not filename or not regions:
            continue
            
        img_path = RAW_IMG_DIR / filename
        
        if not img_path.exists():
            error_count += 1
            continue
            
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                width, height = img.size
                
                for idx, region in enumerate(regions):
                    damage_class = region.get("class", "unknown_damage")
                    x_points = region.get("all_x", [])
                    y_points = region.get("all_y", [])
                    
                    if not x_points or not y_points:
                        continue
                        
                    # Calculate bounding box
                    min_x, max_x = min(x_points), max(x_points)
                    min_y, max_y = min(y_points), max(y_points)
                    
                    # Add 10% padding so the model sees the surrounding context
                    pad_x = int((max_x - min_x) * 0.10)
                    pad_y = int((max_y - min_y) * 0.10)
                    
                    left = max(0, min_x - pad_x)
                    upper = max(0, min_y - pad_y)
                    right = min(width, max_x + pad_x)
                    lower = min(height, max_y + pad_y)

                    if left >= right or upper >= lower:
                        continue

                    crop_box = (left, upper, right, lower)
                    
                    # Create class-specific directory if it doesn't exist
                    class_dir = PROCESSED_DIR / damage_class
                    class_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Crop and save
                    cropped_img = img.crop(crop_box)
                    out_name = f"{filename.split('.')[0]}_{idx}.jpg"
                    cropped_img.save(class_dir / out_name)
                    
                    # Track statistics
                    processed_count += 1
                    class_counts[damage_class] = class_counts.get(damage_class, 0) + 1
                        
        except Exception as e:
            print(f"Error on {filename}: {e}")
            error_count += 1

    print("-" * 30)
    print("Processing Complete!")
    print(f"Successfully extracted {processed_count} damage crops.")
    print("Class Distribution:")
    for cls_name, count in class_counts.items():
        print(f" - {cls_name}: {count} images")

if __name__ == "__main__":
    parse_vehide_json()