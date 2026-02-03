"""
Extract Class Names from Training Directory
Run this to create a class_names.json file for your Streamlit app
"""

import os
import json

def extract_class_names(train_dir='train'):
    """
    Extract class names from the training directory structure
    
    Args:
        train_dir: Path to your training directory
        
    Returns:
        List of class names
    """
    if not os.path.exists(train_dir):
        print(f"❌ Error: Directory '{train_dir}' not found!")
        print(f"Please provide the correct path to your training directory.")
        return None
    
    # Get all subdirectory names (these are your class names)
    class_names = []
    
    for item in os.listdir(train_dir):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    
    # Sort alphabetically for consistency
    class_names = sorted(class_names)
    
    return class_names


def save_class_names(class_names, output_file='class_names.json'):
    """Save class names to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"✅ Saved {len(class_names)} class names to {output_file}")


def main():
    print("="*60)
    print("Plant Disease Class Names Extractor")
    print("="*60)
    print()
    
    # You can change this path if your training directory is elsewhere
    train_dir = input("Enter path to your 'train' directory (or press Enter for 'train'): ").strip()
    if not train_dir:
        train_dir = 'train'
    
    print(f"\nSearching in: {train_dir}")
    print()
    
    # Extract class names
    class_names = extract_class_names(train_dir)
    
    if class_names is None:
        return
    
    print(f"Found {len(class_names)} classes:")
    print("-"*60)
    
    for i, name in enumerate(class_names):
        print(f"{i:2d}. {name}")
    
    print()
    print("="*60)
    
    # Save to JSON file
    save_choice = input("Save these class names to 'class_names.json'? (y/n): ").strip().lower()
    
    if save_choice == 'y':
        save_class_names(class_names)
        print()
        print("🎉 All done! You can now run your Streamlit app.")
        print("   Run: streamlit run app.py")
    else:
        print("\nClass names not saved. You can copy them manually if needed.")


if __name__ == "__main__":
    main()