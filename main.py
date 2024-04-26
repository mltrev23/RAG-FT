import logging
from transformers import logging as transformers_logging
import json
# Set the logging level to ERROR to minimize output (only critical issues will be reported)
logging.basicConfig(level=logging.ERROR)
transformers_logging.set_verbosity_error()

from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever
from datasets import load_dataset

"""
dataset = load_dataset('text', data_files='bittensor.txt', split = 'train')
data_dict = dataset.to_dict()

#json_str = json.dumps(data_dict, ensure_ascii=False, indent=4)

# Convert to JSON and write to a file
with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data_dict, f, ensure_ascii=False, indent=4)
"""
dataset = load_dataset('json', data_files='dataset.json', split='train')
dataset = [dataset]
print(dataset)
# Load pre-trained RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", dataset=dataset, index_name="compressed")
print('------------------------------------------------')
print(retriever)
print('------------------------------------------------')
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever).to('cuda')

# Example query
input_query = "What is the bittensor?"
# stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])
# stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])
# Encode the input query
input_ids = tokenizer(input_query, return_tensors="pt").input_ids.to('cuda')

# Generate an answer
generated_ids = model.generate(input_ids)

# Decode and print the answer
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
