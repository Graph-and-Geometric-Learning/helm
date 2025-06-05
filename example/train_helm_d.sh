export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 29503 \
train.py \
    --model_name HELM_D\
    --find_unused_parameters True\
    --max_batch_size 4\
    --gradient_accumulation_steps 64\
    --lr 4e-4\
    --weight_decay 0.01\
    --find_unused_parameters True\
    --max_seq_len 2048\
    --arch L6_W390_A6\
    --project_emb False\
