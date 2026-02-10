# Semantic Voting

This is the official implementation for ICLR 2026 paper *"Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks"*. 

Codes are based on the [**open-r1** project](https://github.com/huggingface/open-r1). We would like to express our gratitude to their contribution.

### Environment Setup
Please follow the instructions from **open-r1**.

### Dataset
We have implemented data processing scripts for `wmt24pp`, `wmt14`, `wmt19`, `cnn_dailymail`, `pubmed_summary`, and `alpaca_eval` in `src/dat_processor`. For these datasets, please download from their official huggingface repos and adapt the path in corresponding scripts.

For customizing datasets, please implement new scripts just as the format of our examples and register each function in `src/data_processor/processor_registers.py`.

### Self-generation
> python batch_scripts/batch_evaluate.py

Please remind to customize your local model directory and accelerator config in this scripts.

### Train
To run semantic voting examples.
> python batch_scripts/semantic_pipeline.py

To run entropy minimazition examples.
> python batch_scripts/entropy_pipeline.py

To run self-judging examples.
> python batch_scripts/SRLM_pipeline.py

### Evaluation
> python batch_scripts/batch_evaluate.py

Please remind to customize your testing model path and accelerator config in this scripts.
