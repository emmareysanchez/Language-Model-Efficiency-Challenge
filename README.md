# Language-Model-Efficiency-Challenge
Fine-tuning of a pre-trained language model on a specific NLP benchmark, using a single GPU and a maximum of 24 hours of training

## Setup and execution instructions
It is recommended to create a virtual environment in wsl2 with python 3.8.10 and install the requirements there. To be able to fine-tune or evaluate a model, you should have access to a GPU with at least 12GB of VRAM.
1. Clone the repository
```bash
git clone https://github.com/emmareysanchez/Language-Model-Efficiency-Challenge.git
```

2. Install the requirements
```bash
pip install -r requirements.txt
```

3. Install the PyTorch version that is compatible with the GPU you have access to. For example, if you have access to a GPU with CUDA 12.1, you can install the following version:
```bash
python3 -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/test/cu121
```

3. To train a model, change the pertinent parameters at the beginning of the `src/fine_tuning.py` script and run it as a module:
```bash
python -m src.fine_tuning
```

4. To evaluate a model, change the checkpoint path at the beginning of the `nlp_ifeval.ipynb` notebook and run it.

5. To obtain the metrics of the model's predictions, change [responses_file] with the path to the file generated in the previous step and run the following command from the `ifeval` directory:
```bash
# If you are in the root directory
cd ifeval
# Run the evaluation script
python3 -m instruction_following_eval.evaluation_main --input_data instruction_following_eval/data/input_data.jsonl --input_response_data [responses_file] --output_dir instruction_following_eval/data/evaluation_results
```
