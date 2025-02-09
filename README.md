# Knowing When to Stop: Dynamic Context Cutoff for Large Language Models

[![Website](https://img.shields.io/badge/Website-Project%20Page-red)](https://royxie.com/when-to-stop-project)  
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
[![arXiv](https://img.shields.io/badge/arXiv-2406.15968-b31b1b.svg)](https://arxiv.org/abs/2502.01025)

This is the official repository for [Knowing When to Stop: Dynamic Context Cutoff for Large Language Models](https://arxiv.org/pdf/2502.01025). It contains both the datasets and source code.


### Environment
`Python 3.10` is recommended. Before running the code, export your OpenAI API key : 
```bash
export OPENAI_API_KEY="<your OpenAI api key>".
```

### Installation
Install the required packages:
```bash
pip install -r requirement.txt
```
### Running the Pipeline
Run the full pipeline, which performs probing, trains the classifier, and evaluates the results, using a Llama 1B model with this command:
```bash
python src/run.py --model_name meta-llama/Llama-3.2-1B-Instruct --data_dir "data/short" --output_dir "./results" 
```

### Configuration
To adjust configuration parameters, such as the `number of data points`, `number of heads`, `classification threshold`, or `evaluation methods`, please edit the following: 
```bash
src/config.py
```


### Citation 
If you find our implementation or paper helpful, please consider citing:
```latex
@article{xie2025cutoff,
    title={Knowing When to Stop: Dynamic Context Cutoff for Large Language Models},
    author={Xie, Roy and Wang, Junlin and Rosu, Paul and Deng, Chunyuan and Sun, Bolun and Lin, Zihao and Dhingra, Bhuwan},
    journal={https://arxiv.org/abs/2502.01025},
    year={2025}
}
```
For questions or issues, please open an [issue](https://github.com/ruoyuxie/when-to-stop/issues) on GitHub or [email](https://royxie.com/) directly.
