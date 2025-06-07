import os
import sys

def explore_directory(path, max_depth=3, current_depth=0):
    """Recursively explore directory structure"""
    if current_depth > max_depth:
        return
    
    try:
        items = os.listdir(path)
        indent = "  " * current_depth
        
        for item in items[:10]:  # Limit to first 10 items per directory
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{indent}📁 {item}/")
                if current_depth < max_depth:
                    explore_directory(item_path, max_depth, current_depth + 1)
            else:
                file_size = os.path.getsize(item_path)
                print(f"{indent}📄 {item} ({file_size} bytes)")
        
        if len(items) > 10:
            print(f"{indent}... and {len(items) - 10} more items")
            
    except PermissionError:
        print(f"{indent}❌ Permission denied")
    except Exception as e:
        print(f"{indent}❌ Error: {e}")

def main():
    try:
        print("Starting dataset exploration...")
        dataset_path = r"C:\Users\hamza\.cache\kagglehub\datasets\kostastokis\simpsons-faces\versions\1"
        
        print(f"Exploring dataset structure: {dataset_path}")
        print("=" * 50)
        
        if os.path.exists(dataset_path):
            print(f"✅ Path exists: {dataset_path}")
            explore_directory(dataset_path)
        else:
            print(f"❌ Dataset path does not exist: {dataset_path}")
            print("Checking parent directories...")
            
            # Check if parent directories exist
            parent_paths = [
                r"C:\Users\hamza\.cache",
                r"C:\Users\hamza\.cache\kagglehub",
                r"C:\Users\hamza\.cache\kagglehub\datasets",
                r"C:\Users\hamza\.cache\kagglehub\datasets\kostastokis",
                r"C:\Users\hamza\.cache\kagglehub\datasets\kostastokis\simpsons-faces"
            ]
            
            for parent_path in parent_paths:
                if os.path.exists(parent_path):
                    print(f"✅ Exists: {parent_path}")
                    # List contents
                    try:
                        contents = os.listdir(parent_path)
                        print(f"   Contents: {contents[:5]}{'...' if len(contents) > 5 else ''}")
                    except Exception as e:
                        print(f"   Error listing contents: {e}")
                else:
                    print(f"❌ Missing: {parent_path}")
                    break
        
        print("\n" + "=" * 50)
        print("Exploration completed!")
        
    except Exception as e:
        print(f"Fatal error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Pause to prevent terminal from closing
        print("\nPress Enter to exit...")
        input()

if __name__ == "__main__":
    main()