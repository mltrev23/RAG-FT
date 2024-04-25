from transformers import RagTokenizer, RagTokenForGeneration, RagRetriever

# Load pre-trained RAG model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", dataset="wiki_dpr", index_name="compressed")
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever).to('cuda')

# Example query
input_query = "What is the capital of France?"

# Encode the input query
input_ids = tokenizer(input_query, return_tensors="pt").input_ids.to('cuda')

# Generate an answer
generated_ids = model.generate(input_ids)

# Decode and print the answer
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
