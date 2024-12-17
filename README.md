## How to run the code:

1. Clone the repository
```bash
git clone https://github.com/YUECHE77/Syllabus-Agent.git
```

2. Install Package
```Shell
conda create -n agent python=3.10 -y
conda activate agent
pip install -r requirements.txt
```

You need to download the following models from huggingface:

1. google-bert/bert-base-uncased
2. BAAI/bge-base-en-v1.5
3. BAAI/bge-reranker-base
4. NousResearch/Hermes-2-Pro-Llama-3-8B (Not required)

Also, our trained BERT model: https://drive.google.com/drive/folders/1TeOfVXg-rWr4_MvtsC4pbL9k5HXgsFF7?usp=sharing

You can also find our datasets, experiments results, as well as the processed syllabus (already contained in this repo) from the link above.

It's also necessary to apply an API key from togetherAI: https://api.together.xyz/signin
Recommend: Put the API key in a .env file for safety.

Then, change the model path in python scripts.

We provide:

1. `demo.py` to run single-turn inference. Use `python demo.py`
2. `multi_turn_demo.py` to run multi-turn conversation in the command line. Use `python multi_turn_demo.py`
3. `user_interface.py` as our final and ultimate version of work. You are able to interact with our agent in a well-designed UI (by us). Use `python user_interface.py`

Additionally, we also have localized version and multiple-functions version.

## Introduction to the repository

1. Use GLM, GPT, LLaMA, and Qwen as agent: ./agents/
2. Our binary classifier using BERT and customer model. Also the code to construct dataset: ./binary_classifier/
3. Evaluate the performance of each component: ./evaluations/
4. Useful functions: ./utilities/