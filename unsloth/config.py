from datetime import datetime
# model_name = 'Qwen/Qwen2.5-3B-Instruct'
# model_name = 'meta-llama/Llama-3.1-8B'
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
max_length = int(1024 * 3.5)
max_prompt_length = 256
max_completion_length = max_length - max_prompt_length

time = datetime.now().strftime('%Y%m%d-%H%M%S')
output_dir = f'output/{model_name.split("/")[1]}/{time}'

resume_from_checkpoint = False
checkpoint_load_dir = ''
wandb_project_name=''
wandb_run_name=''
wandb_run_id=''

if resume_from_checkpoint and checkpoint_load_dir:
    output_dir = '/'.join(checkpoint_load_dir.split('/')[:-1])

lora_rank = 8
num_generations = 8
mask_truncated_completions = False
# log_completions = 'rich'
log_completions = None
# per_device_train_batch_size = 64
per_device_train_batch_size = 32
# gradient_accumulation_steps = 1
gradient_accumulation_steps = 2
# learning_rate = 3e-5
learning_rate = 3e-5
max_steps = 200
save_steps=10
report_to='wandb'
# report_to='none'

eval_strategy = 'no'
per_device_eval_batch_size = 64
eval_steps = 0.1 * max_steps
eval_on_start = False

execution_token_entropy_coef = 0.0025
execution_token_entropy_clamp_min = 0.000
planning_token_entropy_coef = 0.0015
planning_token_entropy_clamp_max = 2.0
# advantage_schedule = 'none'
advantage_schedule = 'hicra'
# advantage_schedule = 'execution_token_entropy'
execution_token_entropy_schedule_ceil = 0.4
execution_token_entropy_schedule_ratio = 0.6
initial_normalized_execution_token_entropy = 0.02

gpu_memory_utilization = 0.40

llama_3_base_chat_template = '''<|begin_of_text|>{{- messages[0].content }}'''

chat_template = llama_3_base_chat_template