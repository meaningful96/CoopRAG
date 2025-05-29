# Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers (CoopRAG) 
<p align="center">
<img width="1000" alt="1" src="https://github.com/user-attachments/assets/8b4089f4-07c2-40ac-a119-3082493b9c0d">
</p>


We used 2 A6000 GPU

# Requirements
- torch==2.4.1
- transformers==4.46.1
- torchvision==0.1.1
- torchaudio==2.4.1
- tqdm==4.66.6
- trl==0.8.3
- numpy==2.0.2

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

# Datasets Download
We have applied question unrolling to the HotpotQA, MuSiQue, 2WikiMultihopQA, and NaturalQuestions (NQ) datasets.   
- You can download the **preprocessed datasets in here**
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
 export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

 CUDA_VISIBLE_DEVICES=0 python3 unrolling_GPT.py
 ```
 - Using sLLM (Gemma-2-9B)
 
 ```bash
 CUDA_VISIBLE_DEVICES=0 python3 unrolling_sLLM.py
 ```

</details> 

# Unrolling-Augmented Generation with RaLa

<p align="center">
<img width="800" alt="1" src="https://github.com/user-attachments/assets/a107c59a-2bf0-43db-ae30-d40cc4d889f7">
</p>


## Training

- Run `./scripts/{Datasets}.sh` file
- Datasets: `hotpotqa`, `2wikimultihop`, `musique`, `nq.sh`
```bash
cd RaLa

# HotpotQA
CUDA_VISIBLE_DEVICES=0,1 bash scripts/hotpotqa.sh
```

## Evaluation and Retrieve top-k Documents
- Run `./scripts/eval_{datasets}.sh` file
- Datasets: `hotpot`, `2wikimultihop`, `musique`, `nq.sh`

```bash
# HotpotQA
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_hotpot.sh
```

# Reasoning Chain Completion and Inference
```bash
cd .. # from RaLa directory
CUDA_VISIBLE_DEVICES=0 python3 inference_sLLM.py
```
