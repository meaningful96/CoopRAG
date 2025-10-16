def initial_input_tripx(instruction, example, context, question):
    return f"""
{instruction}

---

{example}

---
Now your question is

DOCUMENTS:
{context}

QUESTION: 
{question}

RESPONSE:

"""

def initial_input(instruction, example, context, question, sub_questions, triples):
    return f"""
{instruction}

---

{example}

---
Now your question is

DOCUMENTS:
{context}

MAIN_QUESTION: 
{question}

SUB_QUESTIONS:
{sub_questions}

REASONING_CHAIN:
{triples}

RESPONSE:
"""

def verification_input(instruction, example, context, question, sub_questions, triples):
    return f"""
{instruction}

---

{example}

---
Now your question is

DOCUMENTS:
{context}

MAIN_QUESTION: 
{question}

SUB_QUESTIONS:
{sub_questions}

REASONING_CHAIN:
{triples}

Complete Reasoning Chain:"""

def inference_input(instruction, example, context, question, sub_questions, triples):
    return f"""
{instruction}

---

{example}

---
Now your question is

DOCUMENTS:
{context}

MAIN_QUESTION: 
{question}

SUB_QUESTIONS:
{sub_questions}

REASONING_CHAIN:
{triples}

Generated Answer:"""



