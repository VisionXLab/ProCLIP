cd your_path/ProCLIP/scripts

for i in {0..7}
do
    gpu_id=$i
    data_id=$((i+1))
    echo "Launching data_id $data_id on GPU $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python text_embedding_extract.py \
        --data_name cc3m \
        --batch_size 64 \
        --data_id $data_id > log_${data_id}.txt 2>&1 &
done

wait
echo "All processes completed."
