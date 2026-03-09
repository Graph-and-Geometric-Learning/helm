lm-eval \
  --model helm_mice_1b \
  --model_args ckpt_dir=... \
  --tasks commonsense_qa,openbookqa,hellaswag \
  --num_fewshot 0

lm-eval \
  --model helm_mice_1b \
  --model_args ckpt_dir=... \
  --tasks mmlu,arc_challenge \
  --num_fewshot 5