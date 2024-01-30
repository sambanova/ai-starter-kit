
<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

Web Crawled Data Retrieval
======================

# Overview
## About this template
This AI Starter Kit is an example of a semantic search workflow that can be built using the SambaNova platform to get answers to your questions using website-crawled information as the source. 

 -   A configurable SambaStudio connector to run inference off a deployed model.
 -   A configurable integration with a third-party vector database.
 -   An implementation of the semantic search workflow and prompt construction strategies.

This sample is ready to use. We provide instructions to help you run this demo by following a few simple steps described in the [Getting Started](#getting-started) section. it also includes a simple explanation with useful resources for understanding what is happening in each step of the [workflow](#workflow), Then it also serves as a starting point for customization to your organization's needs, which you can learn more about in the [Customizing the Template](#customizing-the-template) section.

# Getting started

## 1. Deploy your model in SambaStudio
Begin by deploying your LLM of choice (e.g. Llama 2 13B chat, etc) to an endpoint for inference in SambaStudio either through the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

## 2. Integrate your model
Integrate your LLM deployed on SambaStudio with this AI starter kit in two simple steps:
1. Clone repo.
```
git clone https://github.com/sambanova/ai-starter-kit.git
```
2. Update API information for the SambaNova LLM and, optionally, the vector database.  These are represented as configurable variables in the environment variables file in sn-ai-starter-kit/web_crawled_data_retriever/export.env. For example, an endpoint with the URL
"https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
would be entered in the config file (with no spaces) as:
```
BASE_URL="https://api-stage.sambanova.net"
PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
VECTOR_DB_URL=http://host.docker.internal:6333
```
3. Install requirements: It is recommended to use virtualenv or conda environment for installation, and to update pip.
```
cd ai_starter_kit/web_crawled_data_retriever
python3 -m venv web_crawling_env
source web_crawling_env/bin/activate
pip install -r requirements.txt
```
## 3. Deploy the starter kit
To run the demo, run the following commands:
```
sh run.sh
```

After deploying the starter kit you should see the following application user interface

![capture of web_crawled_retriever_demo](./web_crawled_app.png)
## 4. Starterkit usage 

1- Pick the data source, that could be previous stored [FAISS](https://github.com/facebookresearch/faiss) vectorstore or a list of website URLs

2- Icude URLs, put each of the sites you want to crawl in the text area and press include URL, you also can clear the list if you want.

3-Chosse the crawling depth determining how many layers of internal links to explore. (limited to 2)

> Be cautious as the crawling depth increases, leading to exponential growth in the number of processed sites. Consider resource implications for efficient workflow performance.

4- Process Crawled websites, there will be created a vectorstore in memory that you can also store on disk if you want.

5- Ask questions about website data!

# Workflow
This AI Starter Kit implements two distinct workflows that pipelines a series of operations.


## 1. Ingestion
This workflow is an example of crawling, parsing and indexing data for subsequent Q&A. The steps are:

1. **Website crawling**: 
    Langchain [AsyncHtmlLoader](https://python.langchain.com/docs/integrations/document_loaders/async_html) Document which is built on top of [requests](https://requests.readthedocs.io/en/latest/) and [aiohttp](https://docs.aiohttp.org/en/stable/) python packages, is used to scrape the html from the websites.

    An iterative approach is employed to delve deeper into the website's references. The process starts by loading the initial HTML content and extracting links present within it using the [beautifulSoup](https://www.crummy.com/software/BeautifulSoup/) package. For each extracted link, the workflow repeats the crawling process, loading the linked page's HTML content, and again identifying additional links within that content. This iterative cycle continues for 'n' iterations, where 'n' represents the specified depth. With each iteration, the workflow explores deeper levels of the website hierarchy, progressively expanding the scope of data retrieval.

2. **Document parsing:**

    Document transformers are tools used to transform and manipulate documents. They take in structured documents as input and apply various transformations to extract specific information or modify the document's content. Document transformers can perform tasks such as extracting properties, generating summaries, translating text, filtering redundant documents, and more. These transformers are designed to process a large number of documents efficiently and can be used to preprocess data before further analysis or to generate new versions of the documents with desired modifications.

    Langchain Document Transformer [html2text](https://python.langchain.com/docs/integrations/document_transformers/html2text) is used to extract plain and clear text from the HTML documents. There are other document transformers like[BeautfulSoup transformer](https://python.langchain.com/docs/integrations/document_transformers/beautiful_soup) available for plain text extraction from HTML included in the LangChain package. Depending on the required information you need to extract form websites, this step might require some customization.

3.  **Data splitting:** 
    Due to token limits in actual Large language models, once the website's data has been parsed and its content extracted, we need to split the data into chunks of text to be embedded and stored in a vector database. This size of the chunk of text depends on the context (sequence) length offered by the model, and generally, larger context lengths result in better performance. The method used to split text also has an impact on performance (for instance, making sure there are no word breaks, sentence breaks, etc.). The downloaded data is splited using [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter).


3. **Embed data:** 
    For each chunk of text from the previous step, we use an embeddings model to create a vector representation of it. These embeddings are used in the storage and retrieval of the most relevant content given a user's query. The split text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html).

    *For more information about what an embeddings is click [here](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)*


4. **Store embeddings:** 
    Embeddings for each chunk, along with content and relevant metadata (such as source website) are stored in a vector database. The embedding acts as the index in the database. In this template, we store information with each entry, which can be modified to suit your needs. There are several vector database options available, each with their own pros and cons. This AI template is setup to use [FAISS](https://github.com/facebookresearch/faiss) as the vector database because it is a free, open-source option with straightforward setup, but can easily be updated to use another if desired. In terms of metadata, ```website source```  is also attached to the embeddings which are stored during  webcrawling process.


## 2. Retrieval
This workflow is an example of leveraging data stored in a vector database along with a large language model to enable retrieval-based Q&A of your data. This method is called [Retrieval Augmented Generation RAG](https://netraneupane.medium.com/retrieval-augmented-generation-rag-26c924ad8181), The steps are:

 1.  **Embed query:** Given a user submitted query, the first step is to convert it into a common representation (an embedding) for subsequent use in identifying the most relevant stored content. Because of this, it is recommended to use the *same* embedding model to generate embeddings. In this sample, the query text is embedded using [HuggingFaceInstructEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceInstructEmbeddings.html), which is the same model  in the ingestion workflow.
 
 2.  **Retrieve relevant content:**
    Next, we use the embeddings representation of the query to make a retrieval request from the vector database, which in turn returns *relevant* entries (content) in it. The vector database therefore also acts as a retriever for fetching relevant information from the database.
    
     *More information about embeddings and their retrieval [here](https://pub.aimind.so/llm-embeddings-explained-simply-f7536d3d0e4b)*
 
 *Find more information about Retrieval augmented generation with LangChain [here](https://python.langchain.com/docs/modules/data_connection/)*

## 3. Response  
**SambaNova Large language model (LLM):** Once the relevant information is retrieved, the content is sent to a SambaNova LLM to generate the final response to the user query. 

   - **Prompt engineering:** The user's query is combined with the retrieved content along with instructions to form the prompt before being sent to the LLM. This process involves prompt engineering, and is an important part in ensuring quality output. In this AI template, customized prompts are provided to the LLM to improve the quality of response for this use case.

     *Learn more about [Prompt engineering](https://www.promptingguide.ai/)*


# Customizing the template


The provided example template can be further customized based on the use case.


## Crawl websites

**website scraping** Different packages are available to crawl and extract out of websites. In the demo app it is implemented the [AsyncHtmlLoader](https://python.langchain.com/docs/integrations/document_loaders/async_html), langchain also includes a cople of [HTML loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/html) that can be used.
This modification can be done in the following location:
```
file: [app.py](app.py)
function: load_htmls
```

**Iterative web crawling:** for each provided site after the scraping, all the referenced links are saved and filtered using [beautifulSoup](https://www.crummy.com/software/BeautifulSoup/) package, then the web crawling method iterates 'n' times scraping this sites and finding referenced links, this depth is limited to 2 maximum depth, but you can modify this limit, and the behavior of the web crawling in the following location:
```
file: [app.py](app.py)
function: web_crawl
```
> Be cautious as the crawling depth increases, leading to exponential growth in the number of processed sites. Consider resource implications for efficient workflow performance.

**Document transformations** Depending on the loader used for scraping the sites, you may want or not to use some transformation method to clean up the downloaded documents. , this could be done in the following location:
```
file: [app.py](app.py)
function: clean_docs
```
*[LangChain](https://python.langchain.com/docs/integrations/document_transformers) provides several document transformers that can be used with you data*

## Split Data

You can experiment with different ways of splitting the data, such as splitting by tokens or using context-aware splitting for code or markdown files. LangChain provides several examples of different kinds of splitting [here](https://python.langchain.com/docs/modules/data_connection/document_transformers/).


The **RecursiveCharacterTextSplitter**, which is used for this template, can be further customized using the `chunk_size` and `chunk_overlap` parameters. For LLMs with a long sequence length, a larger value of `chunk_size` could be used to provide the LLM with broader context and improve performance. The `chunk_overlap` parameter is used to maintain continuity between different chunks.


```python
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=100,
chunk_overlap=20
)
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
file: app.py
function: get_vectorstore
```


## Store embeddings

The template can be customized to use different vector databases to store the embeddings generated by the embedding model. The [LangChain vector stores documentation](https://python.langchain.com/docs/integrations/vectorstores) provides a broad collection of vector stores that can be easily integrated.

This modification can be done in the following location:
```
file: app.py
function: get_vectorstore
```


## Retrieval

Similar to the vector stores, a wide collection of retriever options is also available depending on the use case. In this template, the vector store was used as a retriever, but it can be enhanced and customized, as shown in some of the examples [here](https://python.langchain.com/docs/integrations/retrievers).


This modification can be done in the following location:
```
file: app.py
function: get_conversation_chain
```


## Large language model (LLM)

The template uses the SN LLM model, which can be further fine-tuned to improve response quality. To train a model in SambaStudio, learn how to [prepare your training data](https://docs.sambanova.ai/sambastudio/latest/generative-data-prep.html), [import your dataset into SambaStudio](https://docs.sambanova.ai/sambastudio/latest/add-datasets.html) and [run a training job](https://docs.sambanova.ai/sambastudio/latest/training.html)


### Prompt engineering

Finally, prompting has a significant effect on the quality of LLM responses. Prompts can be further customized to improve the overall quality of the responses from the LLMs. For example, in the given template, the following prompt was used to generate a response from the LLM, where ```question``` is the user query and ```context``` are the documents retrieved by the retriever.
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
> *Learn more about [Prompt engineering](https://www.promptingguide.ai/)*

## Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- streamlit (version 1.25.0)
- langchain (version 1.1.2)
- sentence_transformers (version 2.2.2)
- instructorembedding (version 1.0.1)
- faiss-cpu (version 1.7.4)
- python-dotenv (version 1.0.0)
- beautifulsoup4 (version 4.12.3)
- html2text (version 2020.1.16)
