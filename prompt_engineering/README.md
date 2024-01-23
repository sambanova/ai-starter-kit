

Prompt Engineering Starter Kit
======================

# Overview
## About this template
This AI Starter Kit 

Preview at https://promptask.streamlit.app/

### Prompt engineering




Finally, prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. 

For example, in the given template, the following prompt was used to generate a response from the LLM, where ```question``` is the user query and ```context``` are the documents retrieved by the retriever.
```python
custom_prompt_template = """Use the following pieces of context to answer the question at the end. If the answer to the question cannot be extracted from given CONTEXT than say I do not have information regarding this.
{context}








Question: {question}
Helpful Answer:"""
CUSTOMPROMPT = PromptTemplate(
template=custom_prompt_template, input_variables=["context", "question"]
)
```


This modification can be done in the following location:
```
file: app.py
function: get_conversation_chain
```




