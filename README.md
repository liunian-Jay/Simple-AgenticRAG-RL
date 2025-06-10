
<div align="center">

# 🤖Simple-AgenticRAG with GRPO
</div>

💡 This project aims to provide a minimal implementation of Agentic RAG using reinforcement learning (especially GRPO), and to provide Agentic RAG researchers with a more flexible base project to try and improve.   

💪 The project has just built a demo and is being improved. If you are interested, feel free to participate and improve it.  

<!-- <div align="center">
<img src="images/framework.png" alt="framework" width="800">liang

**GainRAG Framework**
</div> -->


## 🛠 Installation



The main dependencies are [torch](https://pytorch.org/get-started/locally/)>=2.4.0, vllm, , DeepSpeed, [pyserini](https://github.com/castorini/pyserini/tree/master).


```bash
conda create -n AgenticRAG python=3.9
conda activate AgenticRAG
pip install torch torchvision torchaudio 
pip install vllm
```
</details>


## 💡 Preparation
***Download Corpus & Index***

Retrieval is performed on the set of Wikipeda passages used in DPR. Download passages based on [pyserini](https://github.com/castorini/pyserini/tree/master):

```bash
wget https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz
tar xvfz lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz -C indexes # Unzip to a local directory
rm lucene-index.wikipedia-dpr-100w.20210120.d1b9e6.tar.gz
```




## 🎯 Training




## 📈 Run Evaluation
<details>
<summary>
Download Evaluation Data:
</summary>
  
[HotpotQA](https://hotpotqa.github.io/), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), [WebQuestions](https://nlp.stanford.edu/software/sempre/), [NaturalQA](https://ai.google.com/research/NaturalQuestions), [TriviaQA](http://nlp.cs.washington.edu/triviaqa/), [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
</details>

## 📝TODO
- Potential bugs that still exist
- Improvements in training efficiency 
- Integration of other RL algorithms 
- Integration of other search engines

## Acknowledge
The project is inspired by [simple_GRPO](https://github.com/lsdefine/simple_GRPO) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1). We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.



Thanks for your interest in our work!



