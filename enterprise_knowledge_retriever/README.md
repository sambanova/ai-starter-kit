

Enterprise Knowledge Retrieval
======================

# Overview
## About this template
This AI Starter Kit is an example of a semantic search workflow that can be built using the SambaNova platform to get answers to your questions using your PDFs as the source. It includes:
 -   A configurable SambaStudio connector to run inference off a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation of the semantic search workflow and prompt construction strategies.

This sample is ready-to-use. We provide two options to help you run this demo by following a few simple steps described in the [Getting Started](#getting-started) section. It also serves as a starting point for customization to your organization's needs, which you can learn more about in the [Customizing the Template](#customizing-the-template) section.


## Workflow
This AI Starter Kit implements two distinct workflows that pipelines a series of operations.


### Ingestion
This workflow is an example of parsing and indexing data for subsequent Q&A. The steps are:


1. **Document parsing:** Python package [pypdf2](https://pypi.org/project/PyPDF2/) is used to extract text from the PDF documents. There are multiple [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) available for text extraction from PDF on LangChain website. Depending on the quality and the format of the PDF files, this step might require customization for different use cases.


2.  **Split data:** Once the data has been parsed and its content extracted, we need to split the data into chunks of text to be embedded and stored in a vector database. This size of the chunk of text depends on the context (sequence) length offered by the model, and generally, larger context lengths result in better performance. The method used to split text also has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is split using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).


3. **Embed data:** For each chunk of text from the previous step, we use an embeddings model to create a vector representation of it. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).


4. **Store embeddings:** Embeddings for each chunk, along with content and relevant metadata (such as source documents) are stored in a vector database. The embedding acts as the index in the database. In this template, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This AI template is setup to use [Chroma](https://github.com/chroma-core/chroma) as the vector database because it is a free, open-source option with straightforward setup, but can easily be updated to use another if desired. In terms of metadata, ```filename``` and ```page``` are also attached to the embeddings which are extracted during document parsing of the pdf documents.


### Retrieval
This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A off your data. The steps are:


 1.  **Embed query:** Given a user submitted query, the first step is to convert it into a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Because of this, it is recommended to use the *same* embedding model to generate embeddings. In this sample, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same model  in the ingestion workflow.
 
 2.  **Retrieve relevant content:** Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. The vector database therefore also acts as a retriever for fetching relevant information from the database.

### Response
**SambaNova Large language model (LLM):** Once the relevant information is retrieved, the content is sent to a SambaNova LLM to generate the final response to the user query. 


   - **Prompt engineering:** The user's query is combined with the retrieved content along with instructions to form the prompt before being sent to the LLM. This process involves prompt engineering, and is an important part in ensuring quality output. In this AI template, customized prompts are provided to the LLM to improve the quality of response for this use case.



## Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:


- streamlit (version 1.25.0)
- langchain (version 0.0.252)
- sentence_transformers (version 2.2.2)
- instructorembedding (version 1.0.1)
- chromadb (version 0.4.8)
- PyPDF2 (version 3.0.1)
- python-dotenv (version 1.0.0)



# Getting started

## 1. Deploy your model in SambaStudio
Begin by deploying your LLM of choice (e.g. Llama 2 13B chat, etc) to an endpoint for inference in SambaStudio either through the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

## 2. Integrate your model
Integrate your LLM deployed on SambaStudio with this AI starter kit in two simple steps:
1. Clone repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```
2. Update API information for the SambaNova LLM and, optionally, the vector database.  These are represented as configurable variables in the environment variables file in sn-ai-starter-kit/export.env. For example, an endpoint with the URL
"https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
would be entered in the config file (with no spaces) as:
```
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
VECTOR_DB_URL=http://host.docker.internal:6333
```
3.  Install system dependencies
      

   - Ubuntu instalation:
      ```
      sudo apt install tesseract-ocr
      ```
   - Mac Homebrew instalation:
      ```
      brew install tesseract
      ```
  - Windows instalation:
      > [Windows tessearc instalation](https://github.com/UB-Mannheim/tesseract/wiki)

   - For other linux dstributions follow the [**Tesseract-OCR installation guide**](https://tesseract-ocr.github.io/tessdoc/Installation.html) 

4. Install requirements: It is recommended to use virtualenv or conda environment for installation, and to update pip.
```
cd ai_starter_kit/enterprise_knowledge_retriever
python3 -m venv enterprise_knowledge_env
source enterprise_knowledge_env/bin/activate
pip  install  -r  requirements.txt
```
## 3. Deploy the starter kit
To run the demo, run the following commands:
```
streamlit run streamlit/app.py --browser.gatherUsageStats false 
```


# Customizing the template


The provided example template can be further customized based on the use case.


## Import Data




**PDF Format:** Different packages are available to extract text out of PDF files. They can be broadly categorized in two classes, as below
- OCR-based: [pytesseract](https://pypi.org/project/pytesseract/)
- Non-OCR based: [pymupdf](https://pypi.org/project/PyMuPDF/), [pypdf](https://pypi.org/project/pypdf/), [unstructured](https://unstructured.io/)
Most of these packages have easy [integrations](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) with the Langchain library.


This modification can be done in the following location:
```
file: app.py
function: get_data_for_splitting
```


## Split Data


You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).


The **RecursiveCharacterTextSplitter** inside the vectordb class, which is used for this template, can be further customized using the `chunk_size` and `chunk_overlap` parameters. For LLMs with a long sequence length, a larger value of `chunk_size` could be used to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter is used to maintain continuity between different chunks.


```python
text_chunks = vectordb.get_text_chunks(docs=raw_text, chunk_size=1000, chunk_overlap=200, meta_data=meta_data)

```


This modification can be done in the following location:
```
file: app.py
function: get_text_chunks
```


## Embed data


There are several open-source embedding models available on HuggingFace. [This leaderboard](https://huggingface.co/spaces/mteb/leaderboard) ranks these models based on the Massive Text Embedding Benchmark (MTEB). A number of these models are available on SambaStudio and can be further fine-tuned on specific datasets to improve performance.


This modification can be done in the following location:
```
file: ai-starter-kit/vectordb/vector_db.py
function: load_embedding_model
```


## Store embeddings


The template can be customized to use different vector databases to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/) provides a broad collection of vector stores that can be easily integrated.


This modification can be done in the following location:
```
file: app.py
function: create_vector_store
```




## Retrieval


Similar to the vector stores, a wide collection of retriever options is also available depending on the use case. In this template, the vector store was used as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://js.langchain.com/docs/modules/data_connection/retrievers/).


This modification can be done in the following location:
```
file: app.py
function: get_qa_retrieval_chain 
```


## Large language model (LLM)




The template uses the SN LLM model, which can be further fine-tuned to improve response quality. To train a model in SambaStudio, learn how to [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)


### Prompt engineering




Finally, prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in the given template, the following prompt was used to generate a response from the LLM, where ```question``` is the user query and ```context``` are the documents retrieved by the retriever.
```python
custom_prompt_template = """[INST]<<SYS>> You are a helpful assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the answer is not in the context, say that you don't know. Cross check if the answer is contained in provided context. If not than say \"I do not have information regarding this\". Do not use images or emojis in your answer. Keep the answer conversational and professional.<</SYS>>

{context}    

Question: {question}

Helpful answer: [/INST]"""

CUSTOMPROMPT = PromptTemplate(
template=custom_prompt_template, input_variables=["context", "question"]
)
```


This modification can be done in the following location:
```
file: prompts/llama7b-knowledge_retriever-custom_qa_prompt.yaml
```




