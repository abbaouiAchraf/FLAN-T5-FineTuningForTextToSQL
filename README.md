# Fine-tuning flan-t5-base on Spider Dataset

This repository contains the code and instructions for fine-tuning the `flan-t5-base` language model on the Spider dataset for text-to-SQL tasks.

## Dataset

The [Spider](https://yale-lily.github.io/spider) dataset is a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset, which was specifically designed for cross-domain context-dependent text-to-SQL problems.

## Model

The `flan-t5-base` model is a powerful language model from the FLAN (Fine Language ANswer) family of models released by Google AI. It is a variant of the T5 (Text-to-Text Transfer Transformer) architecture, pre-trained on a vast amount of web data using an unsupervised denoising objective.

## Setup

1. Install the required Python packages:

```
pip install -r requirements.txt
```

2. To fine tune it follow the notebook:


## Inference

To use the fine-tuned model for inference, you can load the saved checkpoint and generate SQL queries from natural language queries:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./saved_checkpoint")
model = AutoModelForSeq2SeqLM.from_pretrained("./saved_checkpoint")

query = "What are the names of employees who work in the Marketing department?"

input_ids = tokenizer.encode(query, return_tensors="pt")
output_ids = model.generate(input_ids)

sql_query = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(sql_query)
```

This example code demonstrates how to load the fine-tuned model, encode a natural language query, generate the corresponding SQL query using the model, and decode the output to a string.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or bug fixes.
