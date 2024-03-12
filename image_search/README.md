<a href="https://sambanova.ai/">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../images/SambaNova-light-logo-1.png" height="60">
  <img alt="SambaNova logo" src="../images/SambaNova-dark-logo-1.png" height="60">
</picture>
</a>

SambaNova AI Starter Kits
====================
# Image Search
<!-- TOC -->
- [Overview](#overview)
    - [About this template](#about-this-template)
- [Getting started](#getting-started)
    - [Deploy your models in SambaStudio](#deploy-your-models-in-sambastudio)
        - [Deploy your open-clip endpoint](#deploy-your-open-clip-endpoint)
        - [Use the Open Clip batch inference job](#use-the-open-clip-batch-inference-job)
    - [Set the starter kit and integrate your models](#set-the-starter-kit-and-integrate-your-models)
    - [Deploy the starter kit](#deploy-the-starter-kit)
- [Starterkit usage](#starterkit-usage)
- [Workflow](#workflow)
        - [Image ingestion and embedding](#image-ingestion-and-embedding)
        - [Vector DataBase storage](#vector-database-storage)
        - [Image Retrieval](#image-retrieval)
            - [Search with text](#search-with-text)
            - [Search With image](#search-with-image)
- [Third-party tools and data sources](#third-party-tools-and-data-sources)

<!-- /TOC -->


# Overview 
## About this template
This AI Starter Kit exemplifies a simple approach to image search by image description or image similarity leveraging [Open Clip](https://github.com/mlfoundations/open_clip) embedding models that are deployed using the SambaNova platform, this template provides:

- Batch ingestion / inference for image collestions
- Serach image method with text as imput
- Search image method with image as imput
- Notebook and scripts for custom multimodal [Chroma](https://docs.trychroma.com/multi-modal) data base
- Notebook for downloading test images from [pixbay](https://pixabay.com/)  

# Getting started
## Deploy your models in SambaStudio
### Deploy your open-clip endpoint

Begin by deploying your OpenClip model to an endpoint for inference in SambaStudio either through the GUI or CLI, as described in the [SambaStudio endpoint documentation](https://docs.sambanova.ai/sambastudio/latest/endpoints.html).

### Use the Open Clip batch inference job

This Starter kit automatically will use the Sambanova CLI Snapi to create an Open Clip project and run batch inference jobs for doing image embedding step, you will only need to set your environment API Authorization Key (The Authorization Key will be used to access to the API Resources on SambaStudio), the steps for getting this key is decribed [here](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html#_acquire_the_api_key)

## Set the starter kit and integrate your models

Set your local environment and Integrate your endpoind deployed on SambaStudio with this AI starter kit following this steps:

1. Clone repo.
    ```bash
    git clone https://github.com/sambanova/ai-starter-kit.git
    ```
2. Update API information for the SambaNova LLM and your environment [sambastudio key](#use-the-open-clip-batch-inference-job). 
    
    These are represented as configurable variables in the environment variables file in the root repo directory **```ai-starter-kit/.env```**. For example, an endpoint with the URL
    "https://api-stage.sambanova.net/api/predict/nlp/12345678-9abc-def0-1234-56789abcdef0/456789ab-cdef-0123-4567-89abcdef0123"
    and and a samba studio key ```"1234567890abcdef987654321fedcba0123456789abcdef"```
    would be entered in the environment file (with no spaces) as:
    ```yaml
    BASE_URL="https://api-stage.sambanova.net"
    PROJECT_ID="12345678-9abc-def0-1234-56789abcdef0"
    ENDPOINT_ID="456789ab-cdef-0123-4567-89abcdef0123"
    API_KEY="89abcdef-0123-4567-89ab-cdef01234567"
    VECTOR_DB_URL=http://host.docker.internal:6333
    SAMBASTUDIO_KEY="1234567890abcdef987654321fedcba0123456789abcdef"
    ```
3. Install requirements.

    It is recommended to use virtualenv or conda environment for installation, and to update pip.
    ```bash
    cd ai-starter-kit/image_search
    python3 -m venv image_search_env
    source image_search_env/bin/activate
    pip install -r requirements.txt
    ```
4. Download and install Sambanova CLI.

    Follow the instructions in this [guide](https://docs.sambanova.ai/sambastudio/latest/cli-setup.html) for installing Sambanova SNSDK and SNAPI, (you can omit the *Create a virtual environment* step since you are using the just created ```image_search_env``` environment)

5. Set up config file.

    - Uptate de value of the base_url key in the ```urls``` section of [```config.yaml```](config.yaml) file. Set it with the url of your sambastudio environment
    -  Uptate de value of the open_clip_app_id key in the apps section of [```config.yaml```](config.yaml) file. to find this app id you should execute the following comand in your terminal:
        ```bash
        snapi app list 
        ```
    - Search for the ```OpenCLIP CLIP-ViT-B-32 Backbone``` section in the oputput and copy in the config file the ID value.

## Deploy the starter kit

To run the demo, run the following command

```bash
streamlit run streamlit/app.py --browser.gatherUsageStats false  
```

After deploying the starter kit you should see the following streamlit user interface

![capture of image_search_demo](./docs/image_search.png)


# Starterkit usage 

1- Pick your source (mage Collection or VectorDB). You can process your images in JPG or PNG format stored in a folder selecting ```set images folder path```. Alternatively, you can select a previously created VectorDB selecting ```Use existing vector db```.

> if images folder path is selected this will pass all the images trough the openclip model and will store the embeding of each image in a multimodal chroma vectorDB this will take several minutes

2- Select the search method, Choose between searching by image or text.

3- Describe the image you are searching for in text, or alternatively, upload a new image to search by image similarity.

# Workflow

### Image ingestion and embedding

This step involves the batch ingestion and inference process for image collections. Upon selecting the source as "Image Collection" and providing the path to the folder containing JPG or PNG images, the images are passed through the OpenClip model for embedding generation.

### Vector DataBase storage

After the image embeddings are generated using the OpenClip model, they are stored in a Vector DataBase (VectorDB). This VectorDB serves as a repository for efficiently storing and retrieving the embeddings of images processed during the ingestion phase. By leveraging vector simylarity strategies, the embeddings can be quickly accessed for subsequent image retrieval tasks.

### Image Retrieval

#### Search with text

In this method of image retrieval, users input a textual description of the image they are searching for. Leveraging the embeddings stored in the VectorDB, the system matches the textual description against the corresponding image embeddings to identify relevant images. This search method enables users to find images based on semantic similarities derived from the text input.

#### Search With image

In order to perform image retrieval by uploading an image. The system compares the embedding of the uploaded image with the embeddings stored in the VectorDB to identify images with similar visual features. This method allows users to search for images based on visual similarity, facilitating tasks such as finding visually related images or identifying visually similar objects within a dataset.

# Third-party tools and data sources

All the packages/tools are listed in the requirements.txt file in the project directory. Some of the main packages are listed below:

- ipyplot (version 1.1.0 )
- ftfy (version  4.0.11)
- python-dotenv (version 1.0.1)
- scikit-learn (version 1.4.1.post1)
- chromadb (version 0.4.24)
- matplotlib (version 3.8.3)
- streamlit (version 1.32.0)
- watchdog (version 4.0.0)
- pillow (version 10.2.0)