import os
import tarfile
import tempfile
import shutil
from tqdm import tqdm
import argparse
import csv
def extract_tar_to_dir(tar_path, output_dir):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=output_dir)

def get_all_keys(directory):
    keys = set()
    for f in os.listdir(directory):
        keys.add(f.split('.')[0])
    return keys

def merge_tar_files(
    tar1, tar2, tar3, tar4,output_tar, require_full_match=True,
    exts1=["noise.pt","short.pt","long.pt"],exts2 = ['short.txt'],exts3 = ["jpg","txt"]
):
    
    tmp_dir1 = tempfile.mkdtemp(dir="/dev/shm")
    tmp_dir2 = tempfile.mkdtemp(dir="/dev/shm")
    tmp_dir3 = tempfile.mkdtemp(dir="/dev/shm")

    try:
        print("📦 Extracting tar files...")
        extract_tar_to_dir(tar1, tmp_dir1)
        extract_tar_to_dir(tar2, tmp_dir2)
        extract_tar_to_dir(tar3, tmp_dir3)

        keys1 = get_all_keys(tmp_dir1)
        keys2 = get_all_keys(tmp_dir2)
        keys3 = get_all_keys(tmp_dir3)

        if require_full_match:
            common_keys = sorted(keys1 & keys2 & keys3)
        else:
            common_keys = sorted(keys1 | keys2 | keys3 )

        print(f"🔑 Found {len(common_keys)} common keys.")

        with tarfile.open(output_tar, "w") as out_tar:
            for key in tqdm(common_keys):
                for ext in exts1:
                    fpath = os.path.join(tmp_dir1, f"{key}.{ext}")
                    if os.path.exists(fpath):
                        out_tar.add(fpath, arcname=f"{key}.{ext}")
                for ext in exts2:
                    fpath = os.path.join(tmp_dir2, f"{key}.{ext}")
                    if os.path.exists(fpath):
                        out_tar.add(fpath, arcname=f"{key}.{ext}")
                for ext in exts3:
                    fpath = os.path.join(tmp_dir3, f"{key}.{ext}")
                    if os.path.exists(fpath):
                        out_tar.add(fpath, arcname=f"{key}.{ext}")
        print(f"✅ Merged tar saved to: {output_tar}")

    finally:
        shutil.rmtree(tmp_dir1)
        shutil.rmtree(tmp_dir2)
        shutil.rmtree(tmp_dir3)





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder1",default='your_path/cc3m_converted', type=str) #raw caption tar_dir
    parser.add_argument("--folder2",default='your_path/cc3m_short_text', type=str) # DreamLIP_short_text_dir
    parser.add_argument("--folder3",default='your_path/cc3m', type=str) # original cc3m/cc12m/yfcc15m tar_dir


    parser.add_argument("--output_folder",default = 'your_path/cc3m_merged', type=str) #merged_tar_dir
    parser.add_argument("--split_id",default = 0,type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.output_folder, exist_ok=True)

    tars1 = [f for f in os.listdir(args.folder1) if f.endswith(".tar")]
    tars1 = sorted(tars1)
    total_len = len(tars1)
    split = total_len // 16


    if args.split_id == 15:
        tars1 = tars1[args.split_id*split:]
    else:
        tars1 = tars1[args.split_id * split : (args.split_id + 1) * split]
    for tar_name in tqdm(tars1,total=len(tars1), desc="Processing tar files"):
        tar1 = os.path.join(args.folder1, tar_name)
        tar2 = os.path.join(args.folder2, tar_name)
        tar3 = os.path.join(args.folder3, tar_name)
        out_tar = os.path.join(args.output_folder, tar_name)
        merge_tar_files(tar1, tar2, tar3, out_tar)