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
    --gradient_accumulation_steps 256\
    --lr 4e-4\
    --weight_decay 0.01\
    --gradient_accumulation_steps 256\
    --find_unused_parameters True\
    --model_name HELM_MiCE\
    --max_seq_len 2048\
    --max_batch_size 1\
    --dim 910\
    --inter_dim 3640\
    --mice_inter_dim 1820\
    --train_curv False\
    --n_layers 16\
    --n_dense_layers 1\
    --n_heads 14\
    --n_routed_experts 8\
    --n_shared_experts 1\
    --n_activated_experts 2\
    --kv_lora_rank 257\
    --qk_nope_head_dim 65\
    --qk_rope_head_dim 65\
    --v_head_dim 65\
    --project_emb True\