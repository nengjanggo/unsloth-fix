from datetime import datetime
# model_name = 'Qwen/Qwen3-4B-Instruct-2507'
# model_name = 'Qwen/Qwen3-1.7B-Base'
# model_name = 'Qwen/Qwen2.5-Math-1.5B-Instruct'
model_name = 'Qwen/Qwen2.5-3B-Instruct'
max_length = 1024 * 3
max_prompt_length = 256
max_completion_length = max_length - max_prompt_length

time = datetime.now().strftime('%Y%m%d-%H%M%S')
output_dir = f'output/{model_name.split("/")[1]}/{time}'

resume_from_checkpoint = False
checkpoint_load_dir = ''
wandb_project_name='huggingface'
wandb_run_name=''
wandb_run_id=''

lora_rank = 16
num_generations = 8
per_device_train_batch_size = 64
gradient_accumulation_steps = 2
learning_rate = 3e-5
max_steps = 300
save_steps=50
report_to='wandb'
# report_to='none'

eval_strategy = 'no'
per_device_eval_batch_size = 64
eval_steps = 0.1 * max_steps
eval_on_start = False

execution_token_entropy_coef = 0.0
advantage_schedule = 'none'
# advantage_schedule = 'hicra'
# advantage_schedule = 'execution_token_entropy'

gpu_memory_utilization = 0.3