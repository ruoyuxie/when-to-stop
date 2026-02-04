# Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals
[![Website](https://img.shields.io/badge/Website-Project%20Page-red)](https://royxie.com/when-to-stop-project) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![arXiv](https://img.shields.io/badge/arXiv-2406.15968-b31b1b.svg)](https://arxiv.org/abs/2502.01025)


## 📝 Overview
This is the official repository for [Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals](https://arxiv.org/abs/2502.01025) (NeurIPS 2025). The repo contains the original implementation of the paper, including both the datasets and source code. Check out the project [website](https://royxie.com/when-to-stop-project/) for more information.

⭐ If you find our implementation or paper helpful, please consider citing our work ⭐ :
```latex
@article{xie2025when,
    title={Language Models (Mostly) Know When to Stop Reading},
    author={Xie, Roy and Wang, Junlin and Rosu, Paul and Deng, Chunyuan and Sun, Bolun and Lin, Zihao and Dhingra, Bhuwan},
    journal={Advances in Neural Information Processing Systems},
    year={2025}
}
```


## 📦 Prerequisites
`Python 3.10` is recommended. You can create a virtual environment and install the required packages as follows:
```bash
python -m venv when-to-stop-env
source when-to-stop-env/bin/activate
pip install -r requirement.txt
```
Running the pipeline also requires an OpenAI API key for evaluation. You can export your OpenAI API key as follows: 
```bash
export OPENAI_API_KEY="<your OpenAI api key>".
```

## 🔧 Training & Evaluation Configuration
You can adjust configuration parameters, such as the `number of data points`, `number of heads`, `classification threshold`, or `evaluation methods`, by editing the following file: 
```bash
src/config.py
```

## 🚀 Evaluate the Pipeline
Run the full pipeline, which will first `probe` the model to select the most sufficient heads, then `train` the classifier on the selected heads, and finally `evaluate` the results. You can use a Llama 3.2 1B model as an example with the following command:
```bash
python src/run.py --model_name meta-llama/Llama-3.2-1B-Instruct --data_dir "data/short" --output_dir "./results" 
```

## 📬 Contact
For questions or issues, please open an [issue](https://github.com/ruoyuxie/when-to-stop/issues) on GitHub or [contact](https://royxie.com/) the authors directly.

