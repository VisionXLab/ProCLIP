"""
This python script is used to convert the extract text embedding tar files to matched format of original files
"""


import os
import io
import time
import glob
import torch
import tarfile
import logging
from tqdm import tqdm
from typing import List, Tuple, Union
import webdataset as wds
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import argparse

def log_and_continue(exn):
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image

def load_tar_all(tar_path: str) -> dict:
    results = {}
    start = time.time()
    with tarfile.open(tar_path, 'r:*') as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        print(f'✅ 读取文件列表耗时: {time.time() - start:.2f} 秒')
        for member in tqdm(members, desc="Reading .tar contents"):
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()
            if member.name.endswith('.pt'):
                data = torch.load(io.BytesIO(raw), map_location='cpu')
            else:
                data = raw
            key = member.name[:9]
            results[key] = data
    print(f'✅ 总耗时: {time.time() - start:.2f} 秒')
    return results

def load_embedding_dir(embedding_tar_dir):
    noise_key2embedding = {}
    short_key2embedding = {}
    long_key2embedding = {}
    for tar_path in tqdm(glob.glob(os.path.join(embedding_tar_dir, '*.tar'))):
        local_key2embedding = load_tar_all(tar_path)
        if 'short' in tar_path:
            short_key2embedding.update(local_key2embedding)
        elif 'raw_caption' in tar_path:
            noise_key2embedding.update(local_key2embedding)
        else:
            long_key2embedding.update(local_key2embedding)
    return noise_key2embedding,short_key2embedding, long_key2embedding

# === 加载 Embedding ===
def main(args):
    embedding_tar_dir = args.embedding_tar_dir
    noise_key2embedding,short_key2embedding, long_key2embedding = load_embedding_dir(embedding_tar_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    pixel_list = sorted(glob.glob(args.pixel_base_dir))
    for pixel_path in tqdm(pixel_list,total=len(pixel_list)):
        print("processing {}".format(pixel_path))
        dataset = wds.WebDataset(pixel_path).map(lambda sample: sample["__key__"])
        
        save_tar = os.path.join(args.save_dir, os.path.basename(pixel_path))

        with tarfile.open(save_tar, 'w') as tar:
            for key in tqdm(dataset):
                short_tensor = short_key2embedding.get(key,None)
                if short_tensor ==None:
                    continue
                noise_tensor = noise_key2embedding.get(key).clone()
                short_tensor = short_tensor.clone()
                long_tensor = long_key2embedding.get(key).clone()


                buffer = io.BytesIO()
                torch.save(noise_tensor, buffer)
                info = tarfile.TarInfo(name=key + ".noise.pt")
                info.size = buffer.tell()
                buffer.seek(0)
                tar.addfile(tarinfo=info, fileobj=buffer)


                buffer = io.BytesIO()
                torch.save(short_tensor, buffer)
                info = tarfile.TarInfo(name=key + ".short.pt")
                info.size = buffer.tell()
                buffer.seek(0)
                tar.addfile(tarinfo=info, fileobj=buffer)


                buffer = io.BytesIO()
                torch.save(long_tensor, buffer)
                info = tarfile.TarInfo(name=key + ".long.pt")
                info.size = buffer.tell()
                buffer.seek(0)
                tar.addfile(tarinfo=info, fileobj=buffer)
                # print(key)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_tar_dir', type=str, default='your_path/cc3m')
    parser.add_argument('--embedding_tar_dir', type=str, default='your_path/cc3m_embedding')
    parser.add_argument('--save_dir', type=str, default='your_path/cc3m_converted')
    return parser.parse_args()

if __name__ == '__main__':
    args = args()
    main(args)