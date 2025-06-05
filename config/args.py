import argparse

from .utils import add_flags_from_config

config_args = {
    'training_config': {
        'train': (True, 'If true, MiCE model will return information for load balancing'),
        'min_lr_ratio': (0.1, 'ratio between final target learning rate and initial learning rate'),
        'warm_up_ratio': (0.03, 'percent of steps to use as warm up'),
        'seed': (1234, 'random seed'),
        'lr': (4e-4, 'initial learning rate'),
        'weight_decay': (0.01, 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'packing_ratio': (3.0, 'how many samples to pack into one bin for sample packing'),
        'gradient_accumulation_steps': (256, 'how many steps to update gradients for accelerator'),
        'CHECKPOINT_DIR': ('../ckpt', 'where to save the model'),
        'log_dir': ('../log', 'where to log training dynamics'),
        'data_path': ('../data', 'path to data'),
        'model_name': ('HELM_MiCE', 'One of HELM_D or HELM_MiCE'),
        'find_unused_parameters': (True, 'whether the accelerator should find unused parameters'),
        'max_batch_size': (1, 'Maximum batch size'),
        'max_seq_len': (2048, 'Maximum sequence length'),
        'project_emb': (False, 'If true, the model will map tokens to space-like dimension of Lorentz vectors'),
        'vocab_size': (128256, 'Vocabulary size of the tokenizer')
    },
    'model_config':{
        'dim': (910, 'Model dimension'),
        'inter_dim': (3640, 'Intermediate dimension for MLP layers'),
        'mice_inter_dim': (1820, 'Intermediate dimension for MiCE layers'),
        'n_layers': (16, 'Number of transformer layers'),
        'n_dense_layers': (1, 'Number of dense layers in the model'),
        'n_heads': (14, 'Number of attention heads'),
        # mice
        'n_routed_experts':(8, 'Number of routed experts for MiCE layers'),
        'n_shared_experts':(1, 'Number of shared experts for MiCE layers'),
        'n_activated_experts': (2, 'Number of activated experts in MiCE layers'),
        'n_expert_groups': (1, 'Number of expert groups'),
        'n_limited_groups':(1, 'Number of limited groups for MMiCEoE routing'),
        'score_func': ('softmax', 'Scoring function for MiCE routing'),
        'route_scale':(1., 'Scaling factor for routing scores'),
        'bias_update_speed':(0.005, 'How much to update the bias for gating to ensure expert load balancing'),
        'seq_bal_alpha': (1e-4, 'Scaling for sequence load balancing loss'),
        'train_curv': (True, 'If true, sets the curvatures of the experts as trainable'),
        # hmla
        'q_lora_rank': (0, 'LoRA rank for query projections'),
        'kv_lora_rank': (257, 'LoRA rank for key-value projections'),
        'qk_nope_head_dim': (65, 'Dimension for query-key projections without positional embeddings'),
        'qk_rope_head_dim': (65, 'Dimension for query-key projections with rotary embedding'),
        'v_head_dim':(65, 'Dimension for value projections'),
        # yarn
        'original_seq_len': (2048, 'Original sequence length'),
        'rope_theta': (10000, 'Base for rotary positional encoding'),
        'rope_factor': (40, 'Scaling factor for extended sequence length'),
        'beta_fast': (32, 'Fast beta correction factor'),
        'beta_slow': (1, 'Slow beta correction factor'),
        #helm-d
        'arch': ('L6_W390_A6', 'model architecture for HELM-D, given by La_Wb_Ac, where a is number of layers, b is model dimension, and c is number of heads')
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)