import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import io
from typing import Any, List, Optional, Tuple, TypeVar, Union

import chromadb
import numpy as np
import requests
import yaml
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings, Images, is_document, is_image
from dotenv import load_dotenv
from PIL import Image

from image_search.src.clip_batch_inference import BatchClipProcessor

load_dotenv(os.path.join(repo_dir, '.env'))

Embeddable = Union[Documents, Images]
D = TypeVar('D', bound=Embeddable, contravariant=True)


class ClipEmbbeding(EmbeddingFunction[D]):
    def __init__(self) -> None:
        pass

    def embed_image(self, img_path: Optional[str] = None, img: Optional[Any] = None) -> Any:
        base_url = os.environ.get('CLIP_BASE_URL', '')
        api_key = os.environ.get('CLIP_API_KEY', '')
        url = base_url.replace('nlp', 'file')
        if img_path:
            files = {'predict_file': open(img_path, 'rb')}
        elif img:
            files = {'predict_file': img}
        else:
            raise Exception('please provide a image path or a bytes image file')
        headers = {'key': api_key}
        response = requests.post(url, files=files, headers=headers)
        return response.json()['data'][0]

    def embed_text(self, text: Any) -> Any:
        base_url = os.environ.get('CLIP_BASE_URL', '')
        api_key = os.environ.get('CLIP_API_KEY', '')
        input_data = {'inputs': [text]}
        headers = {'key': api_key, 'Content-Type': 'application/json'}
        response = requests.post(base_url, json=input_data, headers=headers)
        return response.json()['data'][0]

    def __call__(self, input: D) -> Any:
        embeddings: Embeddings = []
        for item in input:
            if is_document(item):
                output = self.embed_text(item)
            elif is_image(item):
                image = Image.fromarray(item)
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                output = self.embed_image(img=buffer.getvalue())
            embeddings.append(output)
        return embeddings


class ImageSearch:
    def __init__(self, path: Optional[str] = None, embbeding: ClipEmbbeding[Any] = ClipEmbbeding()) -> None:
        if path is None:
            self.client = chromadb.PersistentClient(path=os.path.join(kit_dir, 'data', 'vector_db'))
        else:
            self.client = chromadb.PersistentClient(path=path)
        self.embedding_function = embbeding

    def init_collection(self, name: str = 'image_collection', distance: str = 'l2') -> None:
        # try:
        # self.client.delete_collection(name=name)
        # except:
        # pass
        self.collection = self.client.get_or_create_collection(
            name=name, embedding_function=self.embedding_function, metadata={'hnsw:space': distance}
        )
        # collection.get()

    def get_images(self, folder_path: Optional[str] = None) -> Tuple[List[str], Any]:
        if folder_path is None:
            folder_path = os.path.join(kit_dir, 'data', 'images')
        images = []
        paths = []
        for root, _dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    path = os.path.join(root, file)
                    paths.append(path)
                    image = np.array(Image.open(os.path.join(root, file)))
                    images.append(image)
        print(f'got {len (images)} images')
        return paths, images

    def add_images(self, path: Optional[str] = None) -> None:
        if path is None:
            path = os.path.join(kit_dir, 'data', 'images')
        config_path = os.path.join(kit_dir, 'config.yaml')
        with open(config_path, 'r') as file:
            ingestion_mode = yaml.safe_load(file)['clip']['ingestion_mode']
        if ingestion_mode == 'batch_inference_job':
            clip = BatchClipProcessor(config_path=config_path)
            df = clip.process_images(path)
            embeddings = [element['image_vec'] for element in list(df['predictions'])]
            paths = list(df['image_path'].apply(lambda x: os.path.join(path, x)))
            self.collection.add(
                embeddings=embeddings, metadatas=[{'source': path} for path in paths], ids=paths, uris=paths
            )
        elif ingestion_mode == 'online_inference':
            paths, images = self.get_images(path)
            self.collection.add(images=images, metadatas=[{'source': path} for path in paths], ids=paths)
        else:
            raise Exception(f'ingestion mode {ingestion_mode} not supported')

    def search_image_by_text(self, query: str, n: int = 5) -> Any:
        result = self.collection.query(query_texts=[query], include=['uris', 'distances'], n_results=n)  # type: ignore
        assert result['ids'] is not None and result['distances'] is not None
        return result['ids'][0], result['distances'][0]

    def search_image_by_image(self, path: str, n: int = 5) -> Any:
        image = np.array(Image.open(path))
        result = self.collection.query(query_images=[image], include=['uris', 'distances'], n_results=n)  # type: ignore
        assert result['ids'] is not None and result['distances'] is not None
        return result['ids'][0], result['distances'][0]
