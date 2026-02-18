lm-eval \
  --model helm_d_115M \
  --tasks commonsense_qa,openbookqa,hellaswag \
  --num_fewshot 0

lm-eval \
  --model helm_d_115M \
  --tasks mmlu,arc_challenge \
  --num_fewshot 5