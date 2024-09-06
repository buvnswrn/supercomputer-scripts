from llmlingua import PromptCompressor

Reference_Prompt = """
# Instruction

You are an intelligent retrieval system that retrieves the relevant document Id and outputs it in JSON format.
You are tasked with analyzing user request with the content from a database.
The database contains the information in the following format:

`{"Id": "some id", "content": "Some sample content"}`. 

You will output only the Id if a relevant content is found. 

For instance, given the request: `What is the car's name?` and the data:

`[{"Id": "a123d", "content": "The car is a BMW."},
{"Id": "980kj", "content": "He calls his car as Halley."},{"Id": "567fg", "content": "There are 20 people working here."}]`

The response will be : `{"response":["980kj"]}`   

NOTE: Be concise as possible. 
You are allowed to use only the provided data. 
Do not include any explanations or apologies in your response.
Do not respond to any questions that ask for anything else than for you to construct a JSON.
Return only the generated JSON, nothing else.
"""
llm_lingua = PromptCompressor()
compressed_prompt = llm_lingua.compress_prompt(Reference_Prompt)
print(compressed_prompt)
