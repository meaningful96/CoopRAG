# Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers (NeurIPS, 2025)
- Official code repository for **NeurIPS 2025** paper "Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers"
- **Author**: Youmin Ko, Sungjong Seo, Hyunjoon Kim

<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/8b4089f4-07c2-40ac-a119-3082493b9c0d">
</p>

## Requirments
We used 2 A6000(VRAM: 48GB) GPU 

- torch==2.4.1
- transformers==4.46.1
- torchvision==0.1.1
- torchaudio==2.4.1
- tqdm==4.66.6
- trl==0.8.3
- numpy==2.0.2

```bash
pip install -r requirements.txt
conda install conda-forge::faiss-gpu # install faiss

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

## Dataset Download
We have applied question unrolling to the HotpotQA, MuSiQue, 2WikiMultihopQA, and NaturalQuestions (NQ) datasets.
- You can download the **preprocessed datasets in [here](https://drive.google.com/drive/folders/1J_WEkw93nd_bgvoqRWYTCSs13D33MUBa?usp=sharing)**
- HotpotQA, MuSiQue, 2WikiMultihopQA are from HippoRAG
- NaturalQuestions (NQ) is from REAL
-  We have standardized the structure of all datasets to match the original **MuSiQue** dataset.
```python
Data = [
{"id": 'question index',
 "question": 'Q',
 "sub_questions": [subq1, subq2, ...], # Sub Questions
 "sub_triples": [[head1, relation1, tail1], [head2, relation2, tail2], ...], # Uncertain Reasoning Chain
 "answer": '',
 "paragraphs":
    [
    {
      'idx': 'document index',
     'paragraph_text': '',
      'title': '',
     'is_supporting': bool # True or False
     },
     {...}
    ]
}
]
```
<details>
  <summary>Question Unrolling</summary>
 
 - Using GPT-4o-mini
 ```bash
 CUDA_VISIBLE_DEVICES=0 python3 unrolling.py
 ```
</details> 

## Unrolling-Augmented Generation with RaLa

<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/a107c59a-2bf0-43db-ae30-d40cc4d889f7">
</p>

### Training

- Run `./scripts/{Datasets}.sh` file
- Datasets: `hotpotqa`, `2wikimultihop`, `musique`, `nq.sh`
```bash
cd RaLa

# HotpotQA
CUDA_VISIBLE_DEVICES=0,1 bash scripts/hotpotqa.sh
```

### Evaluation and Retrieve top-k Documents
- Run `./scripts/eval_{datasets}.sh` file
- Datasets: `hotpot`, `2wikimultihop`, `musique`, `nq.sh`

```bash
# HotpotQA
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_hotpot.sh
```

## Reasoning Chain Completion and Inference
```bash
cd .. # from RaLa directory
python3 inference.py --dataset hotpotqa

# --dataset: hotpotqa, musique, 2wikimultihop, nq
```

## ChatGPT Classifier-based Grading (Factuality Checking)
We conducted our experiments using the ChatGPT-classifier–based grading protocol from [SimpleQA](https://openai.com/index/introducing-simpleqa/). In this setup, a grader model is shown both the model’s predicted answer and the gold answer, and it assigns one of three labels; Correct, Incorrect, or Not attempted. Correct means the prediction fully contains the gold answer without contradiction; Incorrect means it contradicts the gold answer (even hedged contradictions); and Not attempted means the response does not fully provide the gold answer and also does not contradict it (e.g., “I don’t know,” refusal/deferral, blank, or partial mention that falls short of the exact target).

This protocol enables consistent, automatic evaluation for short, fact-seeking questions, letting us compare models on factuality while also observing conservative behavior via the Not attempted rate. In our study, we follow this procedure without browsing, relying solely on the classifier to judge each prediction against the reference answer.

```bash
python3 factchecking.py --dataset hotpotqa
```
