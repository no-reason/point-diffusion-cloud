import json
from pathlib import Path
import sys

def audit_raw_dataset():
    root = Path("/data/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    split_dir = root / "train_test_split"
    
    with open(split_dir / "shuffled_train_file_list.json", "r") as f:
        train_entries = json.load(f)
    with open(split_dir / "shuffled_val_file_list.json", "r") as f:
        val_entries = json.load(f)
    with open(split_dir / "shuffled_test_file_list.json", "r") as f:
        test_entries = json.load(f)
        
    def count_by_synset(entries):
        counts = {}
        for entry in entries:
            # entry format: shape_data/02691156/10155655850468db78d106ce0a280f87
            parts = Path(entry).parts
            if len(parts) >= 2:
                synset = parts[-2]
                counts[synset] = counts.get(synset, 0) + 1
        return counts

    train_counts = count_by_synset(train_entries)
    val_counts = count_by_synset(val_entries)
    test_counts = count_by_synset(test_entries)
    
    synsetid_to_cate = {
        '02691156': 'airplane', '02773838': 'bag', '02954340': 'cap',
        '02958343': 'car', '03001627': 'chair', '03261776': 'earphone',
        '03467517': 'guitar', '03624134': 'knife', '03636649': 'lamp',
        '03642806': 'laptop', '03790512': 'motorcycle', '03797390': 'mug',
        '03948459': 'pistol', '04099429': 'rocket', '04225987': 'skateboard',
        '04379243': 'table'
    }
    
    print("Name|ID|Train|Val|Test")
    print("---|---|---|---|---")
    
    for synset, name in synsetid_to_cate.items():
        if (root / synset).exists():
            t = train_counts.get(synset, 0)
            v = val_counts.get(synset, 0)
            te = test_counts.get(synset, 0)
            print(f"{name}|{synset}|{t}|{v}|{te}")

if __name__ == "__main__":
    audit_raw_dataset()
