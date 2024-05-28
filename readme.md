# A Watermark for Low-entropy and Unbiased Generation in Large Language Models

## Run C4 Dataset

### STA-1:
```bash
python sample_watermark/run_C4.py --cuda_visible_devices 0 --model_name_or_path 'meta-llama/Llama-2-7b-hf' --n_sample_per_token 1  --gamma 0.5 --max_new_tokens 200
```

### STA-M:
```bash
python sample_watermark_threshold/run_C4.py --cuda_visible_devices 0 --use_entropy_threshold True --entropy_threshold 0.9 --model_name_or_path 'meta-llama/Llama-2-7b-hf' --n_sample_per_token 1 --n_sample_per_token_above_entropy_threshold 16 --gamma 0.5 --max_new_tokens 200
```

## Run HumanEval Dataset

### Install HumanEval
Clone and install the HumanEval package by following the instructions on https://github.com/openai/human-eval under the `data` folder.

### STA-1:
```bash
python sample_watermark/run_human_eval.py --cuda_visible_devices 0 --pass_k 10 --model_name_or_path 'codellama/CodeLlama-7b-Instruct-hf' --n_sample_per_token 1 --prompt_template_name 'instruct' --gamma 0.5 --max_new_tokens 512
```

### STA-M:
```bash
python sample_watermark_threshold/run_human_eval.py --cuda_visible_devices 0 --pass_k 10 --use_entropy_threshold True --entropy_threshold 1.05 --model_name_or_path 'codellama/CodeLlama-7b-Instruct-hf' --n_sample_per_token 1 --n_sample_per_token_above_entropy_threshold 16 --prompt_template_name 'instruct' --gamma 0.5 --max_new_tokens 512
```

## Citing this work

If you use this work, please cite:
```bibtex
@article{mao2024watermark,
  title={A Watermark for Low-entropy and Unbiased Generation in Large Language Models},
  author={Mao, Minjia and Wei, Dongjun and Chen, Zeyu and Fang, Xiao and Chau, Michael},
  journal={arXiv preprint arXiv:2405.14604},
  year={2024}
}
```
