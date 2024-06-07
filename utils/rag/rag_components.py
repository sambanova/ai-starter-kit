import os
from typing import List, Dict, Optional, Any
from utils.rag.base_components import BaseComponents
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt

current_dir = os.getcwd()
kit_dir = os.path.abspath(os.path.join(current_dir, "..")) + "/"

class RAGComponents(BaseComponents):
    """
    A class to hold the components of a RAG pipeline.

    Attributes:
        qa_chain (Optional[Any]): The QA chain component of the RAG pipeline.
        vectorstore: The vector store component of the RAG pipeline.
        embeddings: The embeddings component of the RAG pipeline.
        examples (Optional[list]): The examples component of the RAG pipeline.
        configs (Dict): The configuration of the RAG pipeline.
        prompts_paths (Optional[list]): The paths to the prompts used in the RAG pipeline.
    """

    def __init__(self, 
                 configs: Dict, 
                 embeddings: HuggingFaceInstructEmbeddings, 
                 vectorstore: Chroma, 
                 examples: Optional[list] = None):
        
        self.qa_chain: Optional[Any] = None
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.examples: Optional[list] = examples
        
        self.configs: Dict = self.load_config(configs)
        self.prompts_paths: Optional[list] = self.configs["prompts"]
        
    ### RAG Chains
    
    def init_router(self) -> None:
        """
        Initializes the router with the loaded prompt and LLM.

        This method loads the router prompt from the prompts_paths dictionary and combines it with the LLM (Large Language Model) and a JsonOutputParser to create the router.

        Args:
            self (object): The object that this method is called on.

        Returns:
            None
        """

        router_prompt: Any = load_prompt(kit_dir + self.prompts_paths["router_prompt"]) 
        self.router: Any = router_prompt | self.llm | JsonOutputParser()
    
    def init_example_judge(self) -> None:
        """
        Initializes the example judge with the loaded prompt and LLM.

        Args:
            self (object): The object that this method is called on.

        Returns:
            None
        """
        
        example_judge_prompt: Any = load_prompt(kit_dir + self.prompts_paths["example_judge_prompt"])
        self.example_judge: Any = example_judge_prompt | self.llm | JsonOutputParser()

    def init_reform_chain(self) -> None:
        """
        Initializes the reformulation chain.

        This method loads the reformulation prompt from the prompts_paths dictionary
        and combines it with the language model (LLM) and a StrOutputParser to form
        the reformulation chain.

        Args:
            self (object): The object that this method is called on.

        Returns:
            None
        """

        reformulation_prompt: Dict[str, Any] = load_prompt(kit_dir + self.prompts_paths["reformulation_prompt"])
        self.reformulation_chain: List[Any] = reformulation_prompt | self.llm | StrOutputParser()

    def init_entity_chain(self) -> None:
        """
        Initializes the entity chain by combining the entity prompt, the language model, and the JSON output parser.

        Args:
            self (object): The instance of the rag_components class.

        Returns: 
            None
        """
        
        entity_prompt: Any = load_prompt(kit_dir + self.prompts_paths["entity_prompt"])
        self.entity_chain: Any = entity_prompt | self.llm | JsonOutputParser()

    def init_subquery_chain(self) -> None:
        """
        Initializes the subquery chain.

        This method loads the subquery prompt from the prompts paths and combines it with the LLM (Large Language Model) output
        and the StrOutputParser to form the subquery chain.

        Args:
            self (object): The instance of the rag_components class.

        Returns:
            None
        """

        subquery_prompt: Any = load_prompt(kit_dir + self.prompts_paths["subquery_prompt"])
        self.subquery_chain: Any = subquery_prompt | self.llm | StrOutputParser()

    def init_retrieval_grader(self) -> None:
        """
        Initializes the retrieval grader component.

        This method loads the retrieval grader prompt and combines it with the language model (LLM) and a JSON output parser to create the retrieval grader.

        Args:
            self (object): The object instance.

        Returns:
            None
        """

        retrieval_grader_prompt: Any = load_prompt(kit_dir + self.prompts_paths["retrieval_grader_prompt"])
        self.retrieval_grader: Any = retrieval_grader_prompt | self.llm | JsonOutputParser()

    def init_qa_chain(self) -> None:
        """
        Initializes the QA chain by loading the QA prompt and combining it with the language model.

        Args:
            self (object): The instance of the class.

        Returns:
            None
        """

        qa_prompt: Any = load_prompt(kit_dir + self.prompts_paths["qa_prompt"])
        self.qa_chain: Any = qa_prompt | self.llm | StrOutputParser()

    def init_hallucination_chain(self) -> None:

        hallucination_prompt = load_prompt(kit_dir + self.prompts_paths["hallucination_prompt"])
        self.hallucination_chain = hallucination_prompt | self.llm | JsonOutputParser()

    def init_grading_chain(self) -> None:

        grading_prompt = load_prompt(kit_dir + self.prompts_paths["grading_prompt"])
        self.grading_chain = grading_prompt | self.llm | JsonOutputParser()
    
    def init_failure_chain(self) -> None:

        failure_prompt = load_prompt(kit_dir + self.prompts_paths["failure_prompt"])
        self.failure_chain = failure_prompt | self.llm | StrOutputParser()

    def init_aggregation_chain(self) -> None:

        aggregation_prompt = load_prompt(kit_dir + self.prompts_paths["aggregation_prompt"])
        self.aggregation_chain = aggregation_prompt | self.llm | StrOutputParser()

    def init_final_generation(self) -> None:
        """
        Initializes the final generation process.

        This method loads the final chain prompt and combines it with the language model (LLM) and a string output parser.

        Args:
            self (object): The object instance.

        Returns:
            None
        """

        final_chain_prompt: Any = load_prompt(kit_dir + self.prompts_paths["final_chain_prompt"])
        self.final_chain: Any = final_chain_prompt | self.llm | StrOutputParser()

    ### RAG functionalities

    def initialize_rag(self, state: Dict[str, Any]) -> None:
        """
        Initializes the RAG components with the given state.

        Args:
            state (Dict[str, Any]): A dictionary containing the state of the RAG components.

        Returns:
            None
        """

        print("---Initializing---")
        question: str = state["question"]
        print(question) 

        return {"answers": [], "original_question": question}

    def initialize_complex_rag(self, state: Dict) -> None:
        """
        Initializes the complex RAG with the given state.

        Args:
            state (Dict[str, str]): A dictionary containing the state of the RAG, including the question.

        Returns:
            None
        """

        print("---Initializing---")
        question: str = state["question"]
        print(question)

        return {"rag_counter": 0, "subquestions": [], "answers": [], "examples": self.examples, "original_question": question}

    
    def route_question(self, state: Dict[str, str]) -> str:
        """
        Routes a question to the appropriate component based on the answer type.

        Args:
            state (Dict[str, str]): A dictionary containing the question and other relevant information.

        Returns:
            str: The type of component to route the question to.
        """

        print("---ROUTING QUESTION---")
        question: str = state["question"]
        source: Dict[str, str] = self.router.invoke({"question": question})  

        print(source['answer_type'])

        if source['answer_type'] == 'answer_generation':
            print("---ROUTE QUESTION TO LLM---")
            return "answer_generation"
        
        elif source['answer_type'] =='subquery_generation':
            print("---ROUTE QUESTION TO SUBQUERY_GENERATION---")
            return "subquery_generation"
        
    def use_examples(self, state: Dict[str, str]) -> str:
        """
        Determine whether to use answer generation or example selection based on the input state.

        Args:
            state (Dict[str, str]): A dictionary containing the current state of the system, including the question.

        Returns:
            str: A string indicating whether to use answer generation ("answer_generation") or example selection ("example_selection").
        """

        question = state["question"]
        response = self.example_judge.invoke({"question": question})
        print("---Decision---")
        print(response)

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
        """
        Creates an example selector for semantic similarity-based retrieval.

        Args:
            embeddings (HuggingFaceInstructEmbeddings): Embeddings for the examples.
            vectorstore_cls (Chroma): Vector store class.
            examples (List[...]): A list of examples.
            selector_type (type, optional): Type of example selector. Defaults to SemanticSimilarityExampleSelector.
            k (int, optional): Number of examples to retrieve. Defaults to 1.

        Returns:
            SemanticSimilarityExampleSelector: An example selector for semantic similarity-based retrieval.
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
                     example_selector: 'HuggingFaceInstructEmbeddings', 
                     query: str) -> List[str]:
        """
        Retrieves a list of examples based on the given query and example selector.

        Args:
            example_selector (HuggingFaceInstructEmbeddings): An instance of the HuggingFaceInstructEmbeddings class.
            query (str): The query string used to select examples.

        Returns:
            List[str]: A list of example strings, where each string is in the format "query\nexample".
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

        sel_examples: List[str] = []
        selected_examples = example_selector.select_examples({"query": query})
        print("selected examples", selected_examples)
        for e in selected_examples:
            sel_examples.append(e["query"]+"\n"+e["example"])

        return sel_examples
        
    def reformulate_query(self, state: Dict[str, str]) -> Dict[str, str]:
        """
        Reformulates a question based on the given state.

        Args:
            state (Dict[str, str]): A dictionary containing the question and examples.

        Returns:
            Dict[str, str]: A dictionary containing the reformulated question and examples.
        """

        print("---REFORMULATING QUESTION---")

        question: str = state["question"]
        examples: List[str] = state["examples"]
        print("examples from state: ", examples)

        example_selector = self.get_example_selector(embeddings=self.embeddings, 
                        vectorstore_cls=Chroma,
                        examples=examples, 
                        k=self.configs["retrieval"]["example_selector_k"])

        examples = self.get_examples(example_selector=example_selector,
                                    query=question)

        print("---FOUND EXAMPLES---")
        print(examples)

        examples: str = "\n\n".join(e for e in examples)
        question: str = self.reformulation_chain.invoke({"question": question, "examples": examples})

        print("---REFORMULATED QUESTION---")
        print(question)

        return {"question": question, "examples": examples}

    def retrieve(self, state: Dict) -> Dict: 
        """
        Retrieves relevant documents based on a given question.

        Args:
            state (Dict): A dictionary containing the question to be retrieved.

        Returns:
            Dict: A dictionary containing the retrieved documents and the original question.
        """

        question: str = state["question"]

        if self.configs["retrieval"]["rerank"]:
            search_kwargs: Dict = {
                "score_threshold": self.configs["retrieval"]["score_threshold"], 
                "k": self.configs["retrieval"]["k_retrieved_documents"]
            }
        else:
            search_kwargs: Dict = {
                "score_threshold": self.configs["retrieval"]["score_threshold"], 
                "k": self.configs["retrieval"]["final_k_retrieved_documents"]
            }

        retriever: object = self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs=search_kwargs)

        print("---RETRIEVING FOR QUESTION---")
        print(question)

        documents: List = retriever.invoke(question)

        if self.configs["retrieval"]["rerank"]:
            print("---RERANKING DOCUMENTS---")
            documents = self.rerank_docs(question, documents, self.configs["retrieval"]["final_k_retrieved_documents"])

        return {"documents": documents, "question": question}
    
    def retrieve_w_filtering(self, state: Dict) -> Dict:
        """
        Retrieves documents from the vector store with filtering based on the entity.

        Args:
            state (Dict): A dictionary containing the current state of the RAG model.

        Returns:
            Dict: A dictionary containing the retrieved documents, the current question, the rag counter, the subquestions, and the original question.
        """

        question = state["question"]
        entities = state["entities"]
        original_question = state["original_question"]
        subquestions = state["subquestions"]
        counter = state["rag_counter"]

        if len(subquestions) == 0:
            print("---ANSWERING QUESTION---")
            question = question
            print(question)
        else:
            print("---ANSWERING SUBQUESTIONS---")
            question = subquestions.pop(0)  # Move pop later in the chain
            print(question)
            counter += 1

        q_entities: List[str] = []
        question_list: List[str] = question.replace("?", "").replace("*", "").lower().split()

        # Double check logic here
        for e in entities:
            if e.split(".")[0] in question_list:  # A dumb hack to avoid messing with the vectorstore creation
                q_entities.append(e)
        entity: str = q_entities.pop(0).lower()

        search_kwargs: Dict = {"k": self.configs["retrieval"]["k_retrieved_documents"], "filter": {self.configs["retrieval"]["entity_key"]: {"$eq": entity}}}

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        documents: List = retriever.invoke(question)

        if self.configs["retrieval"]["rerank"]:
            print("---RERANKING DOCUMENTS---")
            documents = self.rerank_docs(question, documents, self.configs["retrieval"]["final_k_retrieved_documents"])

        print("---NUM DOCUMENTS RETRIEVED---")
        print(len(documents))

        return {"documents": documents, "question": question, "rag_counter": counter, "subquestions": subquestions, "original_question": original_question}

    def rag_generate(self, state: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Generates an answer to a question using a question answering chain.

        Args:
            state (Dict[str, List[str]]): A dictionary containing the question, documents, and answers.
                - question (str): The question to be answered.
                - documents (List[str]): A list of documents to be used for answering the question.
                - answers (List[str]): A list of previous answers.

        Returns:
            Dict[str, List[str]]: A dictionary containing the generated answer and the updated list of answers.
                - generation (str): The generated answer.
                - answers (List[str]): The updated list of answers.
        """

        print("---GENERATING---")
        question: str = state["question"]
        documents: List[str] = state["documents"] # TODO: Check typing
        answers: List[str] = state["answers"]

        print("---ANSWERING---")
        print(question)

        docs: List[str] = self._format_docs(documents)

        print("---DOCS---")
        print("length: ", len(docs))
        print(docs)

        generation: str = self.qa_chain.invoke({"question": question, "context": docs})

        "---ANSWER---"
        print(generation)
        if not isinstance(answers, List):
            answers = [answers]

        answers.append(generation)

        return {"generation": generation, "answers": answers}
    
    def grade_documents(self, state: Dict[str, str]) -> Dict[str, List]:
        """
        Grades a list of documents based on their relevance to a given question.

        Args:
            state (Dict[str, str]): A dictionary containing the question and documents to be graded.
                The dictionary should have the following keys:
                    - question (str): The question to which the documents should be graded.
                    - documents (List[str]): A list of documents to be graded.

        Returns:
            Dict[str, List]: A dictionary containing the graded documents and the original question.
                The dictionary has the following keys:
                    - documents (List): A list of documents that are relevant to the question.
                    - question (str): The original question.
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question: str = state["question"]
        documents: List[str] = state["documents"]
        
        # Score each doc
        filtered_docs: List = []
        for d in documents:
            score: Dict[str, str] = self.retrieval_grader.invoke({"question": question, "document": d})
            grade: str = score['score']

            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)

            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue

        return {"documents": filtered_docs, "question": question}
    
    def pass_state(self, state: Dict) -> Dict:
        """
        Get a new query based on the given state.

        Args:
            state (Dict): A dictionary containing the current state.

        Returns:
            Dict: A dictionary containing the new query.
        """

        # question: str = state["question"]

        # return {"question": question}
        return state
    
    def generate_subquestions(self, state: Dict[str, str]) -> Dict[str, List[str]]:
        """
        Generates subquestions based on the given question.

        Args:
            state (Dict[str, str]): A dictionary containing the question to generate subquestions for.

        Returns:
            Dict[str, List[str]]: A dictionary containing the subquestions as a list of strings.
        """

        question = state["question"]
        subquestions = self.subquery_chain.invoke({"question": question})

        print("---SUBQUESTIONS---")
        print(subquestions)
        subquestions = subquestions.split("\n")

        return {"subquestions": subquestions}
    
    def detect_entities(self, state: Dict) -> Dict:
        """
        Detects entities in a given question or subquestion.

        Args:
            state (Dict): A dictionary containing the question and subquestions.

        Returns:
            Dict: A dictionary containing the detected entities.
        """

        print("---DETERMINING ENTITIES---")
        question: str = state["question"]
        print("question: ", question)
        subquestions: List[str] = state["subquestions"]

        if len(subquestions) == 0:
            response: Dict = self.entity_chain.invoke({"question": question})

        else:
            response: Dict = self.entity_chain.invoke({"question": subquestions[0]})
        
        print("entities: ", response["entity_name"])
        if not isinstance(response["entity_name"], List):
            entities: List[str] =[response["entity_name"]]
        else:
            entities: List[str] = response["entity_name"]
        # entities: List[str] = response["entity_name"].split()
        # entities: List[str] = response["entity_name"]
        entities: List[str] = [e.lower() for e in entities]
        entities: List[str] = [e.replace(",", "") for e in entities]
        entities: List[str] = [e + ".pdf" for e in entities]  # Dumb hack to avoid dealing with the vectorstore logic, for now.
        print(entities)

        return {"entities": entities}
    
    def determine_cont(self, state: Dict) -> str:
        """
        Determine whether to continue or iterate based on the state of the RAG chain.

        Args:
            state (Dict): A dictionary containing the current state of the RAG chain.

        Returns:
            str: Either "continue" if the RAG chain is finished, or "iterate" if it's not.
        """

        subquestions = state["subquestions"]
        print(len(subquestions))
        
        if len(subquestions) == 0:
            print("---FINISHED---")
            return "continue"
        else:
            print("---ITERATING ON RAG CHAIN---")
            return "iterate"
        
    def check_hallucinations(self, state):

        print("---CHECK FOR HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        
        score = self.hallucination_chain.invoke(
        {"documents": documents, "generation": generation}
    )
        grade = score["score"]

        # Check hallucination
        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            # Check question-answering
            print("---GRADE GENERATION vs QUESTION---")
            score = self.grading_chain.invoke({"question": question, "generation": generation})
            grade = score["score"]
            if grade == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
            return "not supported"
        
    def failure_msg(self, state):

        question = state["question"]

        failure_msg = self.failure_chain.invoke({"question": question})
        
        return {"answers": failure_msg}
        
    def aggregate_answers(self, state: Dict) -> Dict:
        """
        Returns the final answer based on the intermediate answers and original question.

        Args:
            state (Dict): A dictionary containing the intermediate answers and original question.
                The dictionary should have two keys: "answers" and "original_question".
                "answers" should be a list of strings, and "original_question" should be a string.

        Returns:
            Dict: A dictionary with a single key-value pair. The key is "generation", and the value is the final answer as a string.
        """

        answers: List[str] = state["answers"]
        if isinstance(answers, list):
            answers: str = "\n\n".join(a for a in answers)
        original_question: str = state["original_question"]

        print("---ANSWERING---")
        print(original_question)

        print("---INTERMEDIATE ANSWERS---")
        print(answers)

        print("---FINAL ANSWER---")
        final_answer: str = self.aggregation_chain.invoke({"question": original_question, "answers": answers}) 
        print(final_answer)   

        return {"generation": final_answer}
    
    def final_answer(self, state):

        original_question: str = state["original_question"]
        generation = state["generation"]

        final_answer = self.final_chain.invoke({"question": original_question, "generation": generation})
        
        return {"generation": final_answer}