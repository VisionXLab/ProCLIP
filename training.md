## 1 Data Preparation

### 1.1 Original Data Download
We conduct our experiments on CC3M, CC12M, YFCC15M and the recaption text rephrased by DreamLIP. Please download the data from the following links:

- CC3M: [Link](https://huggingface.co/datasets/pixparse/cc3m-wds  )
- CC12M: [Link](https://huggingface.co/datasets/pixparse/cc12m-wds  )
- YFCC15M: [Link](https://huggingface.co/datasets/dalle-mini/YFCC100M_OpenAI_subset/tree/main  )
- DreamLIP: [Link](https://huggingface.co/datasets/wsdwJohn1231/DreamLIP_capion_csv_w_key  )

It is noted that we only use the train split of the CC3M dataset. In the 1M-scale experiment reported in the paper, we randomly sample 1M samples from CC3M.  
We use captions from DreamLIP augmented with keys. However, we have lost the YFCC15M data with keys. If you intend to use YFCC15M for training, please refer to the original DreamLIP caption file [Link](https://huggingface.co/datasets/qidouxiong619/dreamlip_long_captions/tree/main  ) and perform the necessary mapping.

### 1.2 Text Embedding Generation
There are two approaches to handling text embeddings: one stores embeddings separately, while the other stores them together with the original `.tar` files after extraction. The first method demands more storage space, whereas the second offers relatively faster read speeds. Here, we choose the second approach, but you may select based on your specific circumstances.

We primarily use [LLama-3-8B-cc](https://huggingface.co/microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned  ) as our text embedding model. You should first download it to your local machine.

Check the parameters in `scripts/text_embedding_extract.sh`, especially the file paths, then run the script:

```bash
chmod +x scripts/text_embedding_extract.sh
./scripts/text_embedding_extract.sh
``` 

In the above script, we perform inference using 8 GPUs on a single machine and save the results into individual `.tar` files. You can modify this according to your setup.

After extracting the embeddings, you can convert them into `.tar` files consistent with the original data format. Using our script `scripts/convert_format.py`, you can obtain reformatted `.tar` files—modify the paths as needed:

```bash
chmod +x scripts/convert_format.py
./scripts/convert_format.py
```

Before merging all `.tar` files, you need to download the short captions for DreamLIP rewritten by ShareGPT4V, which will be used in Stage 1 text distillation: [DreamLIP_short_text](https://huggingface.co/datasets/wsdwJohn1231/DreamLIP_short_text  ). After downloading and saving them locally, you can use our pre-defined script `scripts/merge_tar.sh` to merge everything:

```bash
chmod +x scripts/merge_tar.sh
./scripts/merge_tar.sh
```


## 2 Training

Once all preparations are complete, you can begin training.

### 2.1 Stage1 Training

```bash
chmod +x scripts/train_stage1.sh
./scripts/train_stage1.sh
```


### 2.2 Stage2 Training
After completing Stage 1 training, update the checkpoint path in the relevant script and proceed with Stage 2 training:

```bash
chmod +x scripts/train_stage2.sh
./scripts/train_stage2.sh
```