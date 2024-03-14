import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import chromadb
import numpy as np
import io
from PIL import Image
from chromadb.api.types import is_image, is_document, Images,  Documents, EmbeddingFunction, Embeddings
from typing import cast, Union, TypeVar
from src.clip_batch_inference import BatchClipProcessor

from dotenv import load_dotenv
load_dotenv(os.path.join(repo_dir,".env"))

Embeddable = Union[Documents, Images]
D = TypeVar("D", bound=Embeddable, contravariant=True)

class ClipEmbbeding(EmbeddingFunction[D]):
    def __init__(self) -> None:
        pass
    def __call__(self, input: D) -> Embeddings:
        embeddings: Embeddings = []
        for item in input:     
            if is_document(item):
                #TODO implement SN endpoint inference
                output = None
            elif is_image(item):
                image = Image.fromarray(item)
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                buffer
                #TODO implement SN endpoint inference
                output = None
            embeddings.append(output["embedding"])
        return cast(Embeddings, embeddings)

class ImageSearch():
    def __init__(self, path = None, embbeding = ClipEmbbeding):
        if path is None:
            self.client = chromadb.PersistentClient(path=os.path.join(kit_dir,"data/vector_db"))
        else:
            self.client = chromadb.PersistentClient(path=path)
        clip_embedding=ClipEmbbeding()
        self.embedding_function=clip_embedding
        
    def init_collection(self, name="image_collection", distance="cosine"):
        # try:
        #     client.delete_collection(name="image_collection")
        # except:
        #     pass
        self.collection=self.client.get_or_create_collection(
            name="image_collection", 
            embedding_function=self.embedding_function, 
            metadata={"hnsw:space": distance}
            )
        #collection.get()
        
    def get_images(self, folder_path = None ):
        if folder_path is None:
            folder_path=os.path.join(kit_dir,"data/images")
        images=[]
        paths=[]
        for root, _dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    path=os.path.join(root, file)
                    paths.append(path)
                    image= np.array(Image.open(os.path.join(root, file)))
                    images.append(image)
        print(f"got {len (images)} images")
        return paths,images

    def add_images(self, path):
        clip = BatchClipProcessor(config_path=os.path.join(kit_dir,"config.yaml"))
        df = clip.process_images(path)
        embeddings = list(df["predictions"]) 
        paths = list(df["input"].apply(lambda x: os.path.join(kit_dir,'data/images',x)))
        self.collection.add(
            embeddings=embeddings,
            metadatas=[{"source": path} for path in paths],
            ids=paths,
            uris=paths
        )
    
    def search_image_by_text(self, query, n=5):
        result=self.collection.query(query_texts=[query],include=["uris", "distances"],n_results=n)
        return result['uris'][0], result["distances"][0]

    def search_image_by_image(self, path, n=5):
        image= np.array(Image.open(path))
        result=self.collection.query(query_images=[image],include=["uris", "distances"],n_results=n)
        return result['uris'][0], result["distances"][0]