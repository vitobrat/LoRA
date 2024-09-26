from clearml import Task
import subprocess

# Создаём задачу в ClearML
task = Task.init(
    project_name="LoRA Stable Diffusion",
    task_name="Training Naruto Model",
    task_type=Task.TaskTypes.training
)

# Указываем очередь для выполнения на сервере
task.execute_remotely(queue_name="default", exit_process=True)

# Команда для запуска скрипта
command = [
    "accelerate", "launch", "--mixed_precision=fp16", "train_text_to_image_lora.py",
    "--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4",
    "--dataset_name=lambdalabs/naruto-blip-captions",
    "--caption_column=text",
    "--resolution=512",
    "--random_flip",
    "--train_batch_size=1",
    "--num_train_epochs=100",
    "--checkpointing_steps=5000",
    "--learning_rate=1e-04",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--seed=42",
    "--output_dir=sd-naruto-model-lora",
    "--validation_prompt=cute dragon creature",
    "--report_to=wandb"
]

# Запуск команды в процессе
subprocess.run(command, check=True)
