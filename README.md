# QwenQueue

A local Python job queue for image generation and image editing with
`Qwen/Qwen-Image-2.1`. Jobs are stored in SQLite, and a worker loads the model
once and processes queued jobs in submission order.

## Requirements and setup

- Internet access is needed to download dependencies and the model on first use.
  Allow sufficient disk space and system/GPU memory for the model; the worker
  uses bfloat16 weights and sequential CPU offloading. Around 33GB of disk space is required to download the model. The model will be downloaded to `~/.cache/huggingface`.

Run commands from the repository root. Optionally create a Conda environment:

```bash
conda create -n qwenqueue python=3.11
conda activate qwenqueue
```

Install dependencies

```bash
python -m pip install 'torch>=2.4.0' torchvision 'transformers>=5.17'
python -m pip install git+https://github.com/huggingface/diffusers
python -m pip install accelerate pillow
```

## Usage

Start one worker in a terminal:

```bash
python main.py --worker
```

Start the worker before submitting jobs, and wait for
`Worker started.` to appear. Stop it with Ctrl+C.

Submit a generation job from another terminal in the same repository directory:

```bash
python main.py --submit --prompt "An orange on a table"
```

Specify image dimensions and inference steps:

```bash
python main.py --submit \
  --prompt "A mountain lake at sunrise" \
  --width 1024 --height 1024 --steps 40
```

For the intended image-editing workflow, place an image in `input/` and supply its filename:

```bash
python main.py --submit \
  --img_input table.png \
  --prompt "Add an orange on the table"
```

The `script/` directory can be used to store bash scripts.

```bash
#!/bin/bash

cd ../
python main.py --submit \
        --prompt "Create a bowl of sallad"
```
