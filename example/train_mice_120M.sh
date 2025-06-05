export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  --num_machines=1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --main_process_port 29503 \
train.py \
    --model_name HELM_MiCE\
    --find_unused_parameters True\
    --max_batch_size 4\
    --gradient_accumulation_steps 64\
    --lr 4e-4\
    --weight_decay 0.01\
    --find_unused_parameters True\
    --model_name HELM_MiCE\
    --max_seq_len 2048\
    --dim 390\
    --inter_dim 1560\
    --mice_inter_dim 780\
    --train_curv True\
    --n_layers 6\
    --n_dense_layers 1\
    --n_heads 6\
    --n_routed_experts 4\
    --n_shared_experts 1\
    --n_activated_experts 2\
    --kv_lora_rank 65\
    --qk_nope_head_dim 33\
    --qk_rope_head_dim 17\
    --v_head_dim 33\
    --project_emb False\