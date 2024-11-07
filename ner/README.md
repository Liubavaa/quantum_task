# Mountain Name NER Model

## Project Overview
This project trains a Named Entity Recognition (NER) model to detect mountain names within text. Using a custom-generated dataset containing sentences with mountains, the model distinguishes when mountain names appear.

### Dataset choice
The dataset includes sentences from multiple prompts to ensure diversity:
1. **Mountain-Specific Sentences**: Sentences explicitly mentioning mountains to help the model recognize mountain entities.
2. **Mixed-Context Sentences**: Sentences containing other locations name to improve the model’s understanding of difference between mountains and other locations.
3. **Non-Location-Proper-Names Sentences**: Sentences containing other proper names, so model could distinguish between locations and other proper names in sentences.
4. **Out-of-Context Sentences**: Sentences where mountain names are not related to actual mountains, teaching the model to distinguish mountain names in different contexts.

### Model choice
I selected the `distilbert-base-multilingual-cased` model because:

1. **Multilingual Support**:  
   The model is designed to work with multiple languages, which is essential given that mountain names are often used in a variety of languages (e.g., "Everest" in English, "Monte Bianco" in Italian).

2. **Efficiency**:  
   DistilBERT is a smaller, faster version of BERT, with 60% fewer parameters while retaining 97% of BERT's language understanding capabilities. 


## Solution Outline
1. **Data Generation**: Created with a mix of prompts for balanced training samples.
2. **Data Processing**: 
   - Each mountain name is annotated as a `B-mountain` or `I-mountain` tag, while non-mountain tokens are labeled `O`.
   - Data is saved as a Hugging Face `Dataset` for easy integration into `transformers`.
3. **Model Training**: Trained with `distilbert-base-multilingual-cased` for token classification, fine-tuning it to identify mountain names across contexts.
4. **Evaluation**: Utilizes the `seqeval` metric for performance on NER tasks.

## Setup Instructions

### Installation
1. Create a virtual environment (not necessary):
   ```
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
