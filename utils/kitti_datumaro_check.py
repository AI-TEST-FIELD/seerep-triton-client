import sys
from datumaro.components.dataset import Dataset

def load_datumaro_dataset(path):
    # Load Datumaro-format dataset (expects a directory with dataset.json)
    dataset = Dataset.import_from(path, format='datumaro')
    return dataset

def convert_to_kitti_detection(dataset, output_dir):
    # Export dataset to KITTI detection format
    dataset.export(output_dir, format='kitti_detection')
    print(f"Dataset converted and saved to {output_dir} in KITTI detection format.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python kitti_datumaro_check.py <datumaro_project_dir> <output_dir>")
        sys.exit(1)

    datumaro_dir = sys.argv[1]
    output_dir = sys.argv[2]

    dataset = load_datumaro_dataset(datumaro_dir)
    print(f"Loaded Datumaro dataset with {len(dataset)} items.")

    convert_to_kitti_detection(dataset, output_dir)