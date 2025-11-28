import os
os.environ['UNSLOTH_VLLM_STANDBY'] = '1'
from unsloth import FastLanguageModel
from accuracy_rewards import accuracy_reward
from unsloth.config import *
from data_util import get_dapomath17k
from trl import GRPOConfig, GRPOTrainer


def main():
    if resume_from_checkpoint and report_to == 'wandb':
        import wandb
        run = wandb.init(
            project=wandb_project_name,
            name=wandb_run_name,
            id=wandb_run_id,
            resume='must'
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        full_finetuning=False,
        fast_inference=True,
        max_lora_rank=lora_rank
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha = lora_rank * 2,
        use_gradient_checkpointing = 'unsloth'
    )
    
    dataset = get_dapomath17k(10000)
    split_dataset = dataset.train_test_split(test_size=per_device_eval_batch_size * 2, seed=2025)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    config = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        use_vllm=True,
        beta=0.0,
        num_iterations=1,
        epsilon_high=0.28,
        scale_rewards='group',
        loss_type='dapo',
        mask_truncated_completions=True,
        # log_completions='rich',
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        torch_empty_cache_steps=1,
        learning_rate=learning_rate,
        max_steps=max_steps,
        lr_scheduler_type='cosine',
        optim = 'adamw_8bit',
        warmup_ratio=0.05,
        logging_steps=1,
        save_steps=save_steps,
        report_to=report_to,
        eval_strategy=eval_strategy,
        per_device_eval_batch_size=per_device_eval_batch_size,
        bf16_full_eval=True,
        eval_steps=eval_steps,
        eval_on_start=eval_on_start
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint_load_dir)
    else:
        trainer.train(resume_from_checkpoint=False)

    model.save_pretrained_merged(output_dir, tokenizer, save_method='merged_16bit')


if __name__ == '__main__':
    main()