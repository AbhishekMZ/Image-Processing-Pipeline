import os

# Create directories
dirs = [
    'Assignment2/data/raw',
    'Assignment2/data/output'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f"Created directory: {dir_path}")
