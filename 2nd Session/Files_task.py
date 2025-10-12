import os
import random

def folder_files(folder_path,num_files):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist")
    if num_files<1:
        raise ValueError("Number of files to create must be at least 1")

    print(f"Accessing folder:{folder_path}")

    for i in range(num_files):
        file_path=os.path.join(folder_path,f"file_{i + 1}.txt")
        with open(file_path,'w') as f:
            f.write(f"This is file{i + 1}")
    print(f"Created {num_files} files")

    files=[f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    initial_count=len(files)
    print(f"Initial file count:{initial_count}")

    files_to_delete=random.sample(files,len(files) // 2)
    for file_name in files_to_delete:
        os.remove(os.path.join(folder_path,file_name))
    print(f"Deleted{len(files_to_delete)}files")

    remaining_files=[f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    final_count=len(remaining_files)
    print(f"Final file count:{final_count}")
    return {
        "initial_count": initial_count,
        "final_count": final_count,
        "deleted_count": len(files_to_delete)
    }
test_folder = "test_files"
os.makedirs(test_folder, exist_ok=True)

result = folder_files(test_folder, 10)
print("\nSummary:")
print(f"Started with: {result['initial_count']} files")
print(f"Ended with: {result['final_count']} files")
print(f"Deleted: {result['deleted_count']} files")
