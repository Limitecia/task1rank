# SemEval-2024 Task 3: Codalab Service for [THOR-ECAC](https://github.com/nicolay-r/THOR-ECAC) • [![twitter](https://img.shields.io/twitter/url/https/shields.io.svg?style=social)](https://twitter.com/nicolayr_/status/1777005686611751415)

![](https://img.shields.io/badge/Python-3.10-lightgreen.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2404.03361-b31b1b.svg)](https://arxiv.org/abs/2404.03361)
[![arXiv](https://img.shields.io/badge/github-task_description-ffffff.svg)](https://nustm.github.io/SemEval-2024_ECAC/)

> **Update 05 March 2024**: The quick [arXiv paper](https://arxiv.org/abs/2404.03361) breakdowns 🔨 are @ [Twitter/X post](https://twitter.com/nicolayr_/status/1777005686611751415)

This repository shares data and submission-related code for training and handling results of  
[THoR-ECAC framework](https://github.com/nicolay-r/THOR-ECAC), as a part of the SemEval-2024 
paper **[nicolay-r at SemEval-2024 Task 3: Using Flan-T5 for Reasoning Emotion Cause in Conversations with Chain-of-Thought on Emotion States](https://arxiv.org/abs/2404.03361)**

### [👉THoR-ECAC framework👈](https://github.com/nicolay-r/THOR-ECAC) 

# Usage

1. Install necessary project dependencies as follows:
```bash
pip install -r dependencies.txt
```

2. Use [**download.py**](download.py) scripts for fetching ⬇️ all the task-related resources:
```bash
python download.py
```

3. Use the following shared scripts:
* **Resources Preparation for `pair-based` experiments in context**:
  * [0_emotion_state.py](e3_pair_ft/0_emotion_state.py) -- script for pretraining data preparation, based on `states`;
  * [0_emotion_cause.py](e3_pair_ft/0_emotion_cause.py) -- script for fine-tuning data preparation, based on `causes`;
  * [1_ps_vocab.py](e3_pair_ft/1_ps_vocab.py) -- vocabulary preparation for manual spans corrections.
* **Spans correction algorithm**
   [implementation](https://github.com/nicolay-r/SemEval2024-Task3/blob/b68d69da9b96f5ce6ab5b16521521d44ae1c504b/e3_pair_ft/utils_e.py#L56)
   of the vocabulary-based spans-correction technique mentioned in paper.
  ![image](https://github.com/nicolay-r/SemEval2024-Task3/assets/14871187/7f07a26d-60eb-4553-bb1b-e026d6b9d9d9)
* **Codalab submissions forming**:  
  * [2_submit](e3_pair_ft/2_submit.py) -- script for forming `*.json.zip` archive, compatible for submitting on Codalab platfom.
* Conversation data analysis:
  * [task_statistics_json.py](task_statistics_json.py) -- `json` data analyzer;
  * [task_statistics_submission.py](task_statistics_submission.py) -- `*json.zip` submissions analyzer.
  
# References
You can cite this work or [THoR-ECAC framework](https://github.com/nicolay-r/THOR-ECAC) as follows:
```bibtex
@article{rusnachenko2024nicolayr,
  title={nicolay-r at SemEval-2024 Task 3: Using Flan-T5 for Reasoning Emotion Cause in Conversations with Chain-of-Thought on Emotion States},
  booktitle = "Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics",
  author={Nicolay Rusnachenko and Huizhi Liang},
  year= "2024",
  month= jun,
  address = "Mexico City, Mexico",
  publisher = "Association for Computational Linguistics"
}
```
