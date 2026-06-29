"""
SROIE 2019 Organizer - With encoding handling
Handles special characters (£, ¥, ©, etc.) gracefully
"""

import os
import shutil
import json
from pathlib import Path

# ============================================
# CONFIGURATION - MATCH YOUR STRUCTURE
# ============================================

# Get the current script's directory
SCRIPT_DIR = Path(__file__).parent.absolute()

# Your SROIE2019 folder (same level as this script)
SROIE_ROOT = SCRIPT_DIR / "SROIE2019"

# Where to save organized data (inside invoice-project)
OUTPUT_DIR = SCRIPT_DIR / "invoice_data" / "raw" / "sroie"

# ============================================
# HELPER: Read file with fallback encodings
# ============================================

def read_file_with_fallback(file_path):
    """Try multiple encodings to read a file"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'cp437']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read(), encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Last resort: read as binary and decode ignoring errors
    with open(file_path, 'rb') as f:
        raw = f.read()
        return raw.decode('utf-8', errors='ignore'), 'utf-8-ignore'
    
def write_file_safe(file_path, content):
    """Write content with utf-8 encoding"""
    with open(file_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(content)

# ============================================
# ORGANIZE FUNCTION
# ============================================

def organize_sroie():
    print("="*60)
    print("📦 SROIE 2019 ORGANIZER (with encoding fix)")
    print("="*60)
    print(f"\n📁 Script location: {SCRIPT_DIR}")
    print(f"📁 SROIE root: {SROIE_ROOT}")
    
    # Check if SROIE2019 exists
    if not SROIE_ROOT.exists():
        print(f"\n❌ SROIE2019 folder not found at: {SROIE_ROOT}")
        return False
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {"train": {"images": 0, "annotations": 0, "errors": 0}, 
             "test": {"images": 0, "annotations": 0, "errors": 0}}
    
    # Process train and test splits
    for split in ["train", "test"]:
        split_path = SROIE_ROOT / split
        
        if not split_path.exists():
            print(f"\n⚠️ {split} folder not found, skipping...")
            continue
        
        print(f"\n📂 Processing {split}...")
        
        # Source directories
        img_src = split_path / "img"
        box_src = split_path / "box"
        entities_src = split_path / "entities"
        
        # Destination directories
        img_dst = OUTPUT_DIR / split / "images"
        ann_dst = OUTPUT_DIR / split / "annotations"
        img_dst.mkdir(parents=True, exist_ok=True)
        ann_dst.mkdir(parents=True, exist_ok=True)
        
        # Copy all images
        if img_src.exists():
            for img_file in img_src.glob("*.*"):
                dest = img_dst / img_file.name
                shutil.copy2(img_file, dest)
                stats[split]["images"] += 1
            print(f"  ✅ Copied {stats[split]['images']} images to {img_dst}")
        else:
            print(f"  ⚠️ No img folder found at {img_src}")
        
        # Merge box and entities into one annotation file
        if box_src.exists() and entities_src.exists():
            box_files = list(box_src.glob("*.txt"))
            print(f"  📝 Found {len(box_files)} box files")
            
            for box_file in box_files:
                base_name = box_file.stem
                entities_file = entities_src / f"{base_name}.txt"
                
                try:
                    if entities_file.exists():
                        # Read both files with proper encoding handling
                        box_content, box_enc = read_file_with_fallback(box_file)
                        entities_content, ent_enc = read_file_with_fallback(entities_file)
                        
                        ann_dest_path = ann_dst / f"{base_name}.txt"
                        
                        # Combine content
                        combined = "="*60 + "\n"
                        combined += "ENTITIES (Ground Truth)\n"
                        combined += "="*60 + "\n\n"
                        combined += entities_content
                        combined += "\n\n" + "="*60 + "\n"
                        combined += "OCR BOXES\n"
                        combined += "="*60 + "\n\n"
                        combined += box_content
                        
                        # Write combined file
                        write_file_safe(ann_dest_path, combined)
                        stats[split]["annotations"] += 1
                    else:
                        # No matching entities, just copy box file
                        box_content, _ = read_file_with_fallback(box_file)
                        ann_dest_path = ann_dst / f"{base_name}.txt"
                        write_file_safe(ann_dest_path, box_content)
                        stats[split]["annotations"] += 1
                        
                except Exception as e:
                    stats[split]["errors"] += 1
                    print(f"  ⚠️ Error processing {base_name}: {str(e)[:50]}")
                    continue
            
            print(f"  ✅ Created {stats[split]['annotations']} annotation files")
            if stats[split]["errors"] > 0:
                print(f"  ⚠️ {stats[split]['errors']} files had errors (skipped)")
        else:
            print(f"  ⚠️ Missing box or entities folder")
    
    # Print summary
    print("\n" + "="*60)
    print("✅ ORGANIZATION COMPLETE!")
    print("="*60)
    
    print(f"\n📊 TRAIN split:")
    print(f"   Images: {stats['train']['images']}")
    print(f"   Annotations: {stats['train']['annotations']}")
    if stats['train']['errors'] > 0:
        print(f"   Errors: {stats['train']['errors']}")
    
    print(f"\n📊 TEST split:")
    print(f"   Images: {stats['test']['images']}")
    print(f"   Annotations: {stats['test']['annotations']}")
    if stats['test']['errors'] > 0:
        print(f"   Errors: {stats['test']['errors']}")
    
    print(f"\n📁 Output location:")
    print(f"   {OUTPUT_DIR}")
    
    return stats

# ============================================
# EXPLORE FUNCTION - See what we have
# ============================================

def explore_sroie():
    """Display sample files and annotation format"""
    
    print("\n" + "="*60)
    print("🔍 EXPLORING SROIE DATA")
    print("="*60)
    
    for split in ["train", "test"]:
        img_dir = OUTPUT_DIR / split / "images"
        ann_dir = OUTPUT_DIR / split / "annotations"
        
        if not img_dir.exists():
            continue
            
        print(f"\n📁 {split.upper()} split:")
        print(f"   Images: {len(list(img_dir.glob('*.*')))}")
        print(f"   Annotations: {len(list(ann_dir.glob('*.*')))}")
        
        # Show sample image
        images = list(img_dir.glob("*.*"))[:2]
        if images:
            print(f"\n   Sample images:")
            for img in images:
                size = img.stat().st_size / 1024
                print(f"      • {img.name} ({size:.1f} KB)")
        
        # Show sample annotation
        annotations = list(ann_dir.glob("*.txt"))[:1]
        if annotations:
            print(f"\n   Sample annotation: {annotations[0].name}")
            
            # Try to extract entities
            entities = parse_entities_from_annotation(annotations[0])
            print(f"   Extracted fields:")
            for key, value in entities.items():
                if value:
                    print(f"      {key}: {value[:50] if value else 'None'}")

def parse_entities_from_annotation(ann_path):
    """Parse SROIE annotation to extract company, date, address, and total."""
    
    entities = {
        "vendor_name": None,
        "date": None,
        "total_amount": None,
        "address": None
    }
    
    try:
        content, _ = read_file_with_fallback(ann_path)
        
        # Find entities section. SROIE entity files are JSON.
        if "ENTITIES" in content:
            _, rest = content.split("ENTITIES (Ground Truth)", 1)
            json_text = rest.split("OCR BOXES", 1)[0].replace("="*60, "").strip()
            raw_entities = json.loads(json_text)

            entities["vendor_name"] = raw_entities.get("company")
            entities["date"] = raw_entities.get("date")
            entities["address"] = raw_entities.get("address")
            entities["total_amount"] = raw_entities.get("total")
    except Exception as e:
        print(f"Error parsing: {e}")
    
    return entities

# ============================================
# VERIFY FUNCTION - Check annotation quality
# ============================================

def verify_annotations():
    """Check how many annotations have valid fields"""
    
    print("\n" + "="*60)
    print("🔍 VERIFYING ANNOTATION QUALITY")
    print("="*60)
    
    for split in ["train", "test"]:
        ann_dir = OUTPUT_DIR / split / "annotations"
        
        if not ann_dir.exists():
            continue
        
        stats = {
            "total": 0,
            "has_vendor": 0,
            "has_date": 0,
            "has_total": 0,
            "has_address": 0,
            "all_four": 0
        }
        
        for ann_file in list(ann_dir.glob("*.txt"))[:100]:  # Check first 100
            stats["total"] += 1
            entities = parse_entities_from_annotation(ann_file)
            
            if entities["vendor_name"]:
                stats["has_vendor"] += 1
            if entities["date"]:
                stats["has_date"] += 1
            if entities["total_amount"]:
                stats["has_total"] += 1
            if entities["address"]:
                stats["has_address"] += 1
            if all([entities["vendor_name"], entities["date"], 
                    entities["total_amount"], entities["address"]]):
                stats["all_four"] += 1
        
        if stats["total"] > 0:
            print(f"\n📊 {split.upper()} (sample of {stats['total']} files):")
            print(f"   Has vendor name: {stats['has_vendor']}/{stats['total']} ({100*stats['has_vendor']/stats['total']:.1f}%)")
            print(f"   Has date: {stats['has_date']}/{stats['total']} ({100*stats['has_date']/stats['total']:.1f}%)")
            print(f"   Has total amount: {stats['has_total']}/{stats['total']} ({100*stats['has_total']/stats['total']:.1f}%)")
            print(f"   Has address: {stats['has_address']}/{stats['total']} ({100*stats['has_address']/stats['total']:.1f}%)")
            print(f"   All 4 fields: {stats['all_four']}/{stats['total']} ({100*stats['all_four']/stats['total']:.1f}%)")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # First, organize the data
    stats = organize_sroie()
    
    if stats and (stats['train']['images'] > 0 or stats['test']['images'] > 0):
        # Then explore what we have
        explore_sroie()
        
        # Verify annotation quality
        verify_annotations()
        
        print("\n" + "="*60)
        print("🎯 NEXT STEP")
        print("="*60)
        print("\n✅ SROIE 2019 is organized and ready!")
        print("\nYour data is at:")
        print(f"   {OUTPUT_DIR}/train/images/  ({stats['train']['images']} images)")
        print(f"   {OUTPUT_DIR}/train/annotations/  ({stats['train']['annotations']} files)")
        print(f"   {OUTPUT_DIR}/test/images/  ({stats['test']['images']} images)")
        print(f"   {OUTPUT_DIR}/test/annotations/  ({stats['test']['annotations']} files)")
        print("\n📌 Next: Step 2 - Data Preprocessing")
    else:
        print("\n❌ No data was organized. Please check your SROIE2019 folder.")
