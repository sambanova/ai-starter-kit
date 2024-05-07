from typing import Any, List, Dict, Union
import inspect
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
import torch
from langchain.chains import RetrievalQA

from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)


# Obtained from Fei Wu @SambaNova Systems
class VectorStoreRetrieverReranker(VectorStoreRetriever):
    """
    Custom vectorstore retriever with filter and reranking functionality.
    """

    def _get_relevant_documents(
        self,
        query: str,
        filter: Dict[str, Any] = None,
        reranker: Any = None,
        tokenizer: Any = None,
        reranker_thresh: float = None,
        final_k: int = 4,
    ) -> List[Document]:
        """
        Overrided method in Langchain with filter and reranking functionality.
        The reranking only support BAAI/bge-reranker-large model.

        Args:
            query (str): The user query
            filter (dict): The filtering information
            reranker (XLMRobertaForSequenceClassification): The loaded reranker model.
            tokenizer (XLMRobertaTokenizerFast): The tokenizer of the reranekr model.
            reranker_thresh (float): The threshold for reranker.
            final_k (int): The number of documents to select by the reranker.

        Returns:
            docs_sorted (list): The list of relevant documents retrieved.
        """
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(query, filter=filter, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query, filter=filter, **self.search_kwargs
                )
            )
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == "mmr":
            docs = self.vectorstore.max_marginal_relevance_search(
                query, filter=filter, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")


        if (reranker is None) or (tokenizer is None):
            return docs

        pairs = []
        for d in docs:
            pairs.append([query, d.page_content])

        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            scores = (
                reranker(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )

        scores_list = scores.tolist()
        scores_sorted_idx = sorted(
            range(len(scores_list)), key=lambda k: scores_list[k], reverse=True
        )
        if reranker_thresh is not None:
            docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>reranker_thresh]
        else:
            docs_sorted = [docs[k] for k in scores_sorted_idx if scores_list[k]>0]
            docs_sorted = docs_sorted[:final_k]

        return docs_sorted
    

class RetrievalQAReranker(RetrievalQA):
    """
    Custom RetrievalQA Chain with filter and reranking functionality.
    """

    def _get_docs(
        self,
        question: str,
        filter: Dict[str, Any] = None,
        reranker: Any = None,
        tokenizer: Any = None,
        reranker_thresh: float = None,
        final_k: int = 4,
        *,
        run_manager: CallbackManagerForChainRun
    ) -> List[Document]:
        """
        Overrided get docs.

        Args:
            question (str): The user query
            filter (dict): The filtering information
            reranker (XLMRobertaForSequenceClassification): The loaded reranker model.
            tokenizer (XLMRobertaTokenizerFast): The tokenizer of the reranekr model.
            reranker_thresh (float): The threshold for reranker.
            final_k (int): The number of documents to select by the reranker.
            run_manager (CallbackManagerForChainRun): Predefined function in Langchain.
        """
        return self.retriever._get_relevant_documents(question, filter, reranker, tokenizer, reranker_thresh, final_k)

    def _call(
        self, inputs: Dict[str, Any], run_manager: CallbackManagerForChainRun = None
    ) -> Dict[str, Any]:
        """
        Overrided call method so we can provide filter and reranker.

        Args:
            inputs (Dict[str, Any]): The input arguments, including question, filter (optional), reranker (optional), tokenizer (optional), reranker_thresh (optional), final_k (optional)
            run_manager (CallbackManagerForChainRun, optional): Predefined function in Langchain.. Defaults to None.

        Returns:
            Dict[str, Any]: The answer from llm.
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs[self.input_key]

        # Get filter conditions
        filter = inputs.get("filter", None)
        # Get reranker arguments
        reranker = inputs.get("reranker", None)
        tokenizer = inputs.get("tokenizer", None)
        reranker_thresh = inputs.get("reranker_thresh", None)
        final_k = inputs.get("final_k", None)

        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(
                question, filter, reranker, tokenizer, reranker_thresh, final_k, run_manager=_run_manager
            )
        else:
            docs = self._get_docs(question, filter, reranker, tokenizer, reranker_thresh, final_k)
        answer = self.combine_documents_chain.run(
            input_documents=docs, question=question, callbacks=_run_manager.get_child()
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}