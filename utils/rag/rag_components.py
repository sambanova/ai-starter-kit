from typing import List
from utils.rag.base_components import BaseComponents
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt

class RAGComponents(BaseComponents):

    def __init__(self, configs, embeddings, vectorstore, examples=None):

        self.qa_chain = None
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.examples = examples
        
        self.configs = configs
        self.prompts_paths = configs["prompts"]
        
    ### RAG Chains

    def init_router(self) -> None:

        router_prompt =  load_prompt(self.prompts_paths["router_prompt.yaml"]) 
        self.router = router_prompt | self.llm | JsonOutputParser()
    
    def init_example_judge(self) -> None:
        
        example_judge_prompt = load_prompt(self.prompts_paths["example_judge_prompt.yaml"])
        self.example_judge = example_judge_prompt | self.llm | JsonOutputParser()

    def init_reform_chain(self) -> None:

        reformulation_prompt = load_prompt(self.prompts_paths["reformulation_prompt.yaml"])
        self.reformulation_chain = reformulation_prompt | self.llm | StrOutputParser()

    def init_entity_chain(self) -> None:
        
        entity_prompt = load_prompt(self.prompts_paths["entity_prompt.yaml"])
        self.entity_chain = entity_prompt | self.llm | JsonOutputParser()

    def init_subquery_chain(self) -> None:

        subquery_prompt = load_prompt(self.prompts_paths["subquery_prompt"])
        self.subquery_chain = subquery_prompt | self.llm | StrOutputParser()

    def init_retrieval_grader(self) -> None:

        retrieval_grader_prompt = load_prompt(self.prompts_paths["retrieval_grader_prompt"])
        self.retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

    def init_qa_chain(self) -> None:

        qa_prompt = load_prompt(self.prompts_paths["qa_prompt"])
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def init_final_generation(self) -> None:

        final_chain_prompt = load_prompt(self.prompts_paths["final_chain_prompt"])
        self.final_chain = final_chain_prompt | self.llm | StrOutputParser()

    ### RAG functionalities

    def initialize_rag(self, state) -> None:

        print("---Initializing---")
        question = state["question"]
        print(question) 

        return {"answers": [] , "original_question": question}

    def initialize_complex_rag(self, state):

        print("---Initializing---")
        # Lazy hardcoding for now.
        # from enterprise_knowledge_retriever.prompts.example_prompts import examples
        question = state["question"]
        print(question)

        return {"counter": 0, "subquestions": [], "answers": [], "examples": self.examples, "original_question": question}
    
    def route_question(self, state) -> str:

        print("---ROUTING QUESTION---")
        question = state["question"]
        source = self.router.invoke({"question": question})  

        print(source['answer_type'])

        if source['answer_type'] == 'answer_generation':
            print("---ROUTE QUESTION TO LLM---")
            return "answer_generation"
        
        elif source['answer_type'] == 'subquery_generation':
            print("---ROUTE QUESTION TO SUBQUERY_GENERATION---")
            return "subquery_generation"
        
    def use_examples(self, state):

        question = state["question"]
        response =self.example_judge.invoke({"question": question})

        if response["generate_or_example"] == "answer_generation":
            print("---ROUTE ORIGINAL QUESTION---")
            return "answer_generation"
        elif response["generate_or_example"] == "example_selection":
            print("---ROUTE TO EXAMPLE SELECTION---")
            return "example_selection"
        
    def get_example_selector(self, 
                            embeddings: HuggingFaceInstructEmbeddings, 
                            vectorstore_cls: Chroma,
                            examples: list,
                            selector_type=SemanticSimilarityExampleSelector,
                            k: int=1,
                            ) -> SemanticSimilarityExampleSelector:
        """Create a dynamic example selector for few shot examples during QA Chain calls
        Args:
            embeddings (HuggingFaceInstructEmbeddings): embedding model
            vector_database (Chroma): Chroma vector database
            examples (list): list of examples to use
            selector_type (_type_, optional): Selection type - hardcoded to SemanticSimilarityExampleSelector for now. Defaults to SemanticSimilarityExampleSelector.
            k (int, optional): number of example to obtain. Defaults to 1.
        Returns:
            SemanticSimilarityExampleSelector: the similarity based example selector
        """
            
        example_selector = selector_type.from_examples(
        input_keys=['query'],
        examples=examples,
        embeddings=embeddings,
        vectorstore_cls=vectorstore_cls,
        k=k,
        )

        return example_selector
    
    def get_examples(self, 
                     example_selector: HuggingFaceInstructEmbeddings, 
                     query: str) -> List:
        """simple function to get examples from the example selector
        Args:
            example_selector (HuggingFaceInstructEmbeddings): embedding model
            query (str): user input query
            examples (List): list of examples for the example selector
        Returns:
            eg_template (str): the prompt template including few shot examples
        """

        # example_prompt = PromptTemplate(input_variables=["query", "example"], 
        #                                        template="Input: {query}\nOuptut: {example}"
        # )

        # Might need to follow the proper prompt template for the model here.
        # similar_prompt = FewShotPromptTemplate(
        # example_selector=example_selector,
        # example_prompt=example_prompt,
        # prefix="Provide a new query based on the example query-response pairs.",
        # suffix="Input: {query}\nOutput:",
        # input_variables=["query"],
        # )

        # sel_examples = similar_prompt.format(query=query)

        sel_examples = []
        selected_examples = example_selector.select_examples({"query": query})
        print("selected examples", selected_examples)
        for e in selected_examples:
            sel_examples.append(e["query"]+"\n"+e["example"])

        return sel_examples
        
    def reformulate_query(self, state):

        print("---REFORMULATING QUESTION---")

        question = state["question"]
        examples = state["examples"]
        print("examples from state: ", examples)

        example_selector = self.get_example_selector(embeddings=self.embeddings, 
                        vectorstore_cls=Chroma,
                        examples=examples, 
                        k=self.configs["retrieval"]["example_selector_k"])

        examples = self.get_examples(example_selector=example_selector,
                                    query=question)

        print("---FOUND EXAMPLES---")
        print(examples)

        examples = "\n\n".join(e for e in examples)
        question = self.reformulation_chain.invoke({"question": question, "examples": examples})

        print("---REFORMULATED QUESTION---")
        print(question)

        return {"question": question, "examples": examples}

    def retrieve(self, state): #TODO: Generalize to not having to use filtering unless specified.

        question = state["question"]

        if self.configs["retrieval"]["rerank"]:
            search_kwargs = {
                "score_threshold": self.configs["retrieval"]["score_threshold"], 
                "k": self.configs["retrieval"]["k_retrieved_documents"]
                }
        else:
            search_kwargs = {
                "score_threshold": self.configs["retrieval"]["score_threshold"], 
                "k": self.configs["retrieval"]["final_k_retrieved_documents"]
                }

        retriever = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs=search_kwargs)

        print("---RETRIEVING FOR QUESTION---")
        print(question)

        documents = retriever.invoke(question)

        if self.configs["retrieval"]["rerank"]:
            print("---RERANKING DOCUMENTS---")
            documents = self.rerank_docs(question, documents, self.configs["retrieval"]["final_k_retrieved_documents"])

        return {"documents": documents, "question": question}
    
    def retrieve_w_filtering(self, state):

        question = state["question"]
        entities = state["entities"]
        original_question = state["original_question"]
        subquestions = state["subquestions"]
        counter = state["counter"]

        if len(subquestions)==0:
            "---ANSWERING QUESTION---"
            question = question
        else:
            "---ANSWERING SUBQUESTIONS---"
            question = subquestions.pop(0) # Move pop later in the chain
            counter += 1

        q_entities = []
        question_list = question.replace("?","").replace("*","").lower().split()

        # Double check logic here
        for e in entities:
            if e.split(".")[0] in question_list: # A dumb hack to avoid messing with the vectorestore creation
                q_entities.append(e)
        entity = q_entities.pop(0).lower()

        search_kwargs = {"k": self.configs["retrieval"]["top_k"], "filter": {self.configs["retrieval"]["entity_key"]: {"$eq": entity}}}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        documents = retriever.invoke(question)

        if self.configs["retrieval_complex_rag"]["rerank"]:
            print("---RERANKING DOCUMENTS---")
            documents = self.rerank_docs(question, documents, self.configs["retrieval"]["final_k"])

        print("---NUM DOCUMENTS RETRIEVED---")
        print(len(documents))

        return {"documents": documents, "question": question, "counter": counter, "subquestions": subquestions, "original_question": original_question}

    def rag_generate(self, state):

        print("---GENERATING---")
        question = state["question"]
        documents = state["documents"]
        answers = state["answers"]

        print("---ANSWERING---")
        print(question)

        docs = self._format_docs(documents)

        print("---DOCS---")
        print("length: ", len(docs))
        print(docs)

        generation = self.qa_chain.invoke({"question": question, "context": docs})

        "---ANSWER---"
        print(generation)
        answers.append(generation)

        return {"generation": generation, "answers": answers}
    
    def grade_documents(self, state):

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score['score']

            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {"documents": filtered_docs, "question": question}
    
    def get_new_query(self, state):

        question = state["question"]

        return {"question": question}
    
    def generate_subquestions(self, state):

        question = state["question"]
        subquestions = self.subquery_chain.invoke({"question": question})

        print("---SUBQUESTIONS---")
        print(subquestions)
        subquestions = subquestions.split("\n")

        return {"subquestions": subquestions}
    
    def detect_entities(self, state):

        print("---DETERMINING ENTITIES---")
        question = state["question"]
        subquestions = state["subquestions"]

        if len(subquestions) == 0:
            response = self.entity_chain.invoke({"question": question})

        else:
            response = self.entity_chain.invoke({"question": subquestions[0]})

        entities = response["entity_name"].split()
        entities = [e.lower() for e in entities]
        entities = [e.replace(",", "") for e in entities]
        entities = [e + ".pdf" for e in entities] # Dumb hack to avoid dealing with the vectorstore logic, for now.
        print(entities)

        return {"entities": entities}
    
    def determine_cont(self, state):

        subquestions = state["subquestions"]
        print(len(subquestions))
        
        if len(subquestions) == 0:
            print("---FINISHED---")
            return "continue"
        else:
            print("---ITERATING ON RAG CHAIN---")
            return "iterate"
        
    def return_final_answer(self, state):

        answers = state["answers"]
        answers = "\n\n".join(a for a in answers)
        original_question = state["original_question"]

        print("---ANSWERING---")
        print(original_question)

        print("---INTERMEDIATE ANSWERS---")
        print(answers)

        print("---FINAL ANSWER---")
        final_answer = self.final_chain.invoke({"question": original_question, "answers": answers}) 
        print(final_answer)   

        return {"generation": final_answer}