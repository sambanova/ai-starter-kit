"""
DEPRECATED: This file is no longer maintained and may be removed in future releases.

Copyright 2023 SambaNova Systems, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import glob
from haystack.document_stores import InMemoryDocumentStore, FAISSDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever

class RetrieverBase:
    def __init__(self, doc_dir, tokenizer, top_k=10):
        self.retriever = None
        self.top_k = top_k
        self.tokenizer = tokenizer

    def __call__(self, query):
        candidate_documents = self.retriever.retrieve(
            query=query,
            top_k=self.top_k,
        )
        return [(d.content, d.meta["name"]) for d in candidate_documents]

    def get_all_docs(self, doc_dir):
        docs = []
        all_files = glob.glob(os.path.join(doc_dir, "*", "*.txt"))
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if filename.startswith("document_store") or filename.startswith(
                "faiss_document_store.db"
            ):
                continue
            with open(file_path) as f:
                lines = f.readlines()
                content = "".join(lines)
            content_length = len(self.tokenizer.encode(content))
            if content_length > 4000:
                print(f"Skip file {filename} [{content_length} tokens]")
                continue

            docs.append({"content": content, "meta": {"name": filename}})
        return docs

    def load_valid_api(self, valid_api_file):
        valid_api_func = []
        with open(valid_api_file) as f:
            for line in f:
                valid_api_func.append(line.strip())
        print(valid_api_func)
        return valid_api_func


class RetrieverWithBM25(RetrieverBase):
    def __init__(self, doc_dir, tokenizer, top_k=10):
        super().__init__(doc_dir, tokenizer, top_k)
        document_store = InMemoryDocumentStore(use_bm25=True)
        docs = self.get_all_docs(doc_dir)
        document_store.write_documents(docs)
        self.retriever = BM25Retriever(document_store=document_store)


class RetrieverWithEmbedding(RetrieverBase):
    def __init__(self, doc_dir, top_k=10):
        super().__init__(doc_dir, top_k)

        db_path = f"sqlite:///{doc_dir}/faiss_document_store.db"
        faiss_path = f"{doc_dir}/document_store"
        if os.path.exists(faiss_path):
            document_store = FAISSDocumentStore.load(index_path=faiss_path)
        else:
            document_store = FAISSDocumentStore(
                sql_url=db_path,
                faiss_index_factory_str="Flat",
                embedding_dim=768,
                return_embedding=True,
            )
            docs = self.get_all_docs(doc_dir)
            document_store.write_documents(docs)

        self.retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            model_format="sentence_transformers",
        )
        if not os.path.exists(faiss_path):
            document_store.update_embeddings(
                self.retriever, update_existing_embeddings=False
            )
            document_store.save(index_path=faiss_path)