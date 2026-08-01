MODEL=OpenaiCLIP-L-14
# we use the pretrained openai clip as our base model
PRETRAINED=your_path/ViT-L-14.pt
echo "start training"
cd your_path/ProCLIP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 \
        training/main.py \
        --enable-deepspeed \
        --name="cc3m_stage_1_vit-L" \
        --save-frequency 1  \
        --train-data-list "your_path/cc3m/cc3m-train-{0000..0575}.tar
        --dataset-resampled \
        --precision "bf16" \
        --warmup 0 \
        --batch-size=128 \
        --eval-batch-size=1024 \
        --log-every-n-steps 50 \
        --epochs=4 \
        --lr=1e-5 \
        --visual-lr=1e-5 \
        --text-lr=1e-5 \
        --wd=0.05 \
        --visual-wd=0.05 \
        --text-wd=0.05 \
        --ld=1.0 \
        --text-ld=1.01 \
        --visual-ld=0.85 \
        --grad-clip-norm=5.0 \
        --smoothing=0. \
        --workers=16 \
        --model=${MODEL} \
        --seed 4096 \
        --gather-with-grad \
        --force-custom-clip \
        --force-quick-gelu \
        --optimizer="ap_adamw" \
        --zero-stage=1 \
        --dataset-type "webdataset" \
        --distill \
        --teacher OpenaiCLIP-L-14_t\
        --teacher-pretrained ${PRETRAINED} \
        --pretrained=${PRETRAINED} \
        --train-stage 1