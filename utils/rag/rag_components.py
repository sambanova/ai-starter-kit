import os
import sys
from typing import Any, Dict, List, Optional, Type
from utils.rag.base_components import BaseComponents  # type: ignore
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.documents.base import Document
from langchain.vectorstores import Chroma
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import load_prompt
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))


sys.path.append(kit_dir)
sys.path.append(repo_dir)

from utils.logging_utils import log_method # type: ignore


class RAGComponents(BaseComponents):
    """
    This class represents the components of the RAG system.

    Attributes:
    - qa_chain: The QA chain component of the RAG model.
    - vectorstore: The vector store component of the RAG model.
    - embeddings: The embeddings component of the RAG model.
    - examples: Optional examples dictionary.
    - configs: The configuration dictionary.
    - prompts_paths (Dict): The paths to the prompts in the configuration dictionary.
    """

    def __init__(
        self, configs: str, embeddings: Embeddings, vectorstore: Chroma, 
        examples: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initializes the RAG components.

        Args:
            configs: The configuration file path.
            embeddings: The embeddings model.
            vectorstore: The vector store object.
            examples: The examples dictionary. Defaults to None.

        Returns:
            None
        """

        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.examples: Optional[Dict[str, str]] = examples

        self.configs: dict = self.load_config(configs)
        self.prompts_paths: dict = self.configs['prompts']

    ### RAG Chains

    def init_router(self) -> None:
        """
        Initializes the router component.

        This method loads the router prompt and combines it with the language
        model and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        router_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['router_prompt'])
        self.router = router_prompt | self.llm | JsonOutputParser()

    def init_example_judge(self) -> None:
        """
        Initializes the example judge component.

        This method loads the example judge prompt from the repository and combines
        it with the language model and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        example_judge_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['example_judge_prompt'])
        self.example_judge = example_judge_prompt | self.llm | JsonOutputParser()

    def init_reform_chain(self) -> None:
        """
        Initializes the reformulation chain by loading the reformulation prompt and
        combining it with the language model and a string output parser.

        Args:
            None

        Returns:
            None
        """

        reformulation_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['reformulation_prompt'])
        self.reformulation_chain = reformulation_prompt | self.llm | StrOutputParser()

    def init_entity_chain(self) -> None:
        """
        Initializes the entity chain by loading the entity prompt and combining
        it with the language model and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        entity_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['entity_prompt'])
        self.entity_chain = entity_prompt | self.llm | JsonOutputParser()

    def init_subquery_chain(self) -> None:
        """
        Initializes the subquery chain by loading the subquery prompt and
        combining it with the language model and string output parser.

        Args:
            None

        Returns:
            None
        """

        subquery_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['subquery_prompt'])
        self.subquery_chain = subquery_prompt | self.llm | StrOutputParser()

    def init_retrieval_grader(self) -> None:
        """
        Initializes the retrieval grader component.

        This method loads the retrieval grader prompt from the repository and
        combines it with the language model and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        retrieval_grader_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['retrieval_grader_prompt'])
        self.retrieval_grader = retrieval_grader_prompt | self.llm | JsonOutputParser()

    def init_qa_chain(self) -> None:
        """
        Initializes the QA chain by loading the QA prompt and
        combining it with the large language model and a string
        output parser.

        Args:
            None

        Returns:
            None
        """

        qa_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['qa_prompt'])
        self.qa_chain = qa_prompt | self.llm | StrOutputParser()

    def init_hallucination_chain(self) -> None:
        """
        Initializes the hallucination chain for the model.

        This method loads the hallucination prompt from the repository and
        combines it with the language model (LLM) and a JSON output parser to
        create the hallucination chain.

        Args:
            None

        Returns:
            None
        """

        hallucination_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['hallucination_prompt'])
        self.hallucination_chain = hallucination_prompt | self.llm | JsonOutputParser()

    def init_grading_chain(self) -> None:
        """
        Initializes the grading chain by loading the grading prompt and
        combining it with the LLM and a JSON output parser.

        Args:
            None

        Returns:
            None
        """

        grading_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['grading_prompt'])
        self.grading_chain = grading_prompt | self.llm | JsonOutputParser()

    def init_aggregation_chain(self) -> None:
        """
        Initializes the aggregation chain for the model.

        This method loads the aggregation prompt from the repository and
        combines it with the language model and a string output parser to form
        the aggregation chain.

        Args:
            None

        Returns:
            None
        """

        aggregation_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['aggregation_prompt'])
        self.aggregation_chain = aggregation_prompt | self.llm | StrOutputParser()

    def init_final_generation(self) -> None:
        """
        Initializes the final generation process by loading the final chain
        prompt and combining it with the language model (LLM) and a string
        output parser.

        Args:
            None

        Returns:
            None
        """

        final_chain_prompt: Any = load_prompt(repo_dir + '/' + self.prompts_paths['final_chain_prompt'])
        self.final_chain = final_chain_prompt | self.llm | StrOutputParser()

    ### RAG functionalities

    def load_embedding_model(self) -> object:
        """
        Loads the SambaStudio embedding model or E5 large from HuggingFaceInstructEmbeddings
        if using cpu  Option of sambastudio or cpu can be found in the configs file.

        Args:
            None

        Returns:
            The loaded embedding model.
        """

        embeddings = self.vectordb.load_embedding_model(type=self.embedding_model_info)

        return embeddings

    @log_method
    def initialize_rag(self, state: dict) -> dict:
        """
        Initializes the state of the RAG components for LangGraph.

        Args:
            state: A dictionary containing the state of the RAG components.

        Returns:
            The state dict with the question stored in original question.
        """

        print('---Initializing---')

        print('---INITIAL STATE---')
        print(state)

        question: str = state['question']
        print(question)

        return {'answers': [], 'original_question': question}

    @log_method
    def initialize_complex_rag(self, state: dict) -> dict:
        """
        Initializes the complex RAG with the given state.

        Args:
            state: A dictionary containing the state of the RAG.

        Returns:
            The initialized state dict.
        """

        print('---Initializing---')
        question: str = state['question']
        print(question)

        return {
            'rag_counter': 0,
            'subquestions': [],
            'answers': [],
            'examples': self.examples,
            'original_question': question,
        }

    @log_method
    def route_question(self, state: dict) -> str:
        """
        Routes a question to the appropriate component based on the answer type.

        Args:
            state: A dictionary containing the state, including the question
            and other relevant information.

        Returns:
            The type of component to route the question to.
        """

        print('---ROUTING QUESTION---')
        question: str = state['question']
        source: Dict[str, str] = self.router.invoke({'question': question})

        print(source['answer_type'])

        if source['answer_type'] == 'answer_generation':
            print('---ROUTE QUESTION TO LLM---')
            routing = 'answer_generation'

        elif source['answer_type'] == 'subquery_generation':
            print('---ROUTE QUESTION TO SUBQUERY_GENERATION---')
            routing = 'subquery_generation'

        return routing

    @log_method
    def use_examples(self, state: dict) -> str:
        """
        Determine whether to use answer generation or example
        selection based on the input state.

        Args:
            state: A dictionary containing the current state of the
            system, including the question.

        Returns:
            A string indicating whether to use answer generation
            ("answer_generation") or example selection ("example_selection").
        """

        question: str = state['question']
        try:
            response = self.example_judge.invoke({'question': question})
        except Exception as e:
            print(e)
        print('---Decision---')
        print(response)

        if response['generate_or_example'] == 'answer_generation':
            print('---ROUTE ORIGINAL QUESTION---')
            routing = 'answer_generation'

        elif response['generate_or_example'] == 'example_selection':
            print('---ROUTE TO EXAMPLE SELECTION---')
            routing = 'example_selection'

        return routing

    @log_method
    def get_example_selector(
        self,
        embeddings: Embeddings,
        vectorstore_cls: Type[VectorStore],
        examples: List[Dict[str, str]],
        selector_type: SemanticSimilarityExampleSelector,
        k: int = 1,
    ) -> SemanticSimilarityExampleSelector:
        """
        Creates an example selector for semantic similarity-based retrieval.

        Args:
            embeddings: Embeddings for the examples.
            vectorstore_cls: Vector store class.  Tested with chromadb.
            examples: A list of examples, which are dictionaries of paired strings.
            selector_type: Type of example selector. Defaults to SemanticSimilarityExampleSelector.
            k: Number of examples to retrieve. Defaults to 1.

        Returns:
            An example selector for semantic similarity-based retrieval.
        """

        example_selector = selector_type.from_examples(
            input_keys=['query'],
            examples=examples,
            embeddings=embeddings,
            vectorstore_cls=vectorstore_cls,
            k=k,
        )

        return example_selector

    @log_method
    def get_examples(self, example_selector: SemanticSimilarityExampleSelector, query: str) -> List[str]:
        """
        Retrieves a list of examples based on the given query and example selector.

        Args:
            example_selector:: An instance of the HuggingFaceInstructEmbeddings class.
            query: The query string used to select examples.

        Returns:
            A list of example strings, where each string is in the format "query\nexample".
        """

        sel_examples: List[str] = []
        selected_examples = example_selector.select_examples({'query': query})
        print('selected examples', selected_examples)
        for e in selected_examples:
            sel_examples.append(e['query'] + '\n' + e['example'])

        return sel_examples

    @log_method
    def reformulate_query(self, state: dict) -> dict:
        """
        Reformulates a question based on the given state.

        Args:
            state: A dictionary containing the question and examples.

        Returns:
            A dictionary containing the reformulated question and examples.
        """

        print('---REFORMULATING QUESTION---')

        question: str = state['question']
        examples: list[dict[str, str]] = state['examples']
        print('examples from state: ', examples)

        selector = SemanticSimilarityExampleSelector(vectorstore=self.vectorstore)
        example_selector = self.get_example_selector(
            selector_type=selector,
            embeddings=self.embeddings,
            vectorstore_cls=Chroma,
            examples=examples,
            k=self.configs['retrieval']['example_selector_k'],
        )

        list_examples = self.get_examples(example_selector=example_selector, query=question)

        print('---FOUND EXAMPLES---')
        print(examples)

        str_examples: str = '\n\n'.join(e for e in list_examples)
        new_question: str = self.reformulation_chain.invoke({'question': question, 'examples': str_examples})

        print('---REFORMULATED QUESTION---')
        print(question)

        return {'question': new_question, 'examples': examples}

    @log_method
    def retrieve(self, state: dict) -> dict:
        """
        Retrieves relevant documents based on a given question.

        Args:
            state: A dictionary containing the question to be retrieved.

        Returns:
            A dictionary containing the retrieved documents and the original question.
        """

        question: str = state['question']
        search_kwargs: dict = {}

        if self.configs['retrieval']['rerank']:
            search_kwargs = {
                'score_threshold': self.configs['retrieval']['score_threshold'],
                'k': self.configs['retrieval']['k_retrieved_documents'],
            }
        else:
            search_kwargs = {
                'score_threshold': self.configs['retrieval']['score_threshold'],
                'k': self.configs['retrieval']['final_k_retrieved_documents'],
            }

        retriever = self.vectorstore.as_retriever(search_type='similarity_score_threshold', search_kwargs=search_kwargs)

        print('---RETRIEVING FOR QUESTION---')
        print(question)

        documents: List[Document] = retriever.invoke(question)

        if self.configs['retrieval']['rerank']:
            print('---RERANKING DOCUMENTS---')
            documents = self.rerank_docs(question, documents, self.configs['retrieval']['final_k_retrieved_documents'])

        return {'documents': documents, 'question': question}

    @log_method
    def retrieve_w_filtering(self, state: dict) -> dict:
        """
        Retrieves documents from the vector store with filtering based on the entity.

        Args:
            state: A dictionary containing the current state of the RAG model.

        Returns:
            The state dict, with updated retrieved documents, the current question,
            the rag counter, the subquestions, and the original question.
        """

        question: str = state['question']
        entities: List[str] = state['entities']
        original_question: str = state['original_question']
        subquestions: List[str] = state['subquestions']
        counter: int = state['rag_counter']

        if len(subquestions) == 0:
            print('---ANSWERING QUESTION---')
            question = question
            print(question)
        else:
            print('---ANSWERING SUBQUESTIONS---')
            question = subquestions.pop(0)
            print(question)
            counter += 1

        q_entities: List[str] = []
        # Remove "?" and "*" prior to entity determination
        question_list: List[str] = question.replace('?', '').replace('*', '').lower().split()

        entity: str = ''
        try:
            for e in entities:
                if e.split('.')[0] in question_list:
                    q_entities.append(e)
            entity = q_entities.pop(0).lower()
        except:
            entity = ''

        search_kwargs: dict = {
            'k': self.configs['retrieval']['k_retrieved_documents'],
            'filter': {self.configs['retrieval']['entity_key']: {'$eq': entity}},
        }

        retriever = self.vectorstore.as_retriever(search_kwargs=search_kwargs)

        documents: List[Document] = retriever.invoke(question)

        if self.configs['retrieval']['rerank']:
            print('---RERANKING DOCUMENTS---')
            try:
                documents = self.rerank_docs(
                    question, documents, self.configs['retrieval']['final_k_retrieved_documents']
                )
            except:
                pass

        print('---NUM DOCUMENTS RETRIEVED---')
        print(len(documents))
        for d in documents:
            print(d.page_content)

        return {
            'documents': documents,
            'question': question,
            'rag_counter': counter,
            'subquestions': subquestions,
            'original_question': original_question,
        }

    @log_method
    def rag_generate(self, state: dict) -> dict:
        """
        Generates an answer to a question using a question answering chain.

        Args:
            state : A dictionary containing the question, documents, answers,
            and other variables.

        Returns:
            The updated state dict containing the generated answer and the
            updated list of answers.
        """

        print('---GENERATING---')
        question: str = state['question']
        documents: List[str] = state['documents']
        answers: List[str] = state['answers']

        print('---ANSWERING---')
        print(question)

        docs: str = self._format_docs(documents)

        print('---DOCS---')
        n_tokens = len(docs.split()) * 1.3
        print('number of approximate tokens (n words *1.3): ', n_tokens)
        print(docs)

        generation: str = self.qa_chain.invoke({'question': question, 'context': docs})

        print('---ANSWER---')
        print(generation)
        if not isinstance(answers, List):
            answers = [answers]

        answers.append(generation)

        return {'generation': generation, 'answers': answers}

    @log_method
    def grade_documents(self, state: dict) -> dict:
        """
        Grades a list of documents based on their relevance to a given question.

        Args:
            state: A dictionary containing the question and documents to be graded.

        Returns:
            A dictionary containing the graded documents and the
            original question.
        """

        print('---CHECK DOCUMENT RELEVANCE TO QUESTION---')
        question: str = state['question']
        documents: List[Document] = state['documents']

        # Score each doc
        filtered_docs: List = []
        for d in documents:
            try:
                score: Dict[str, str] = self.retrieval_grader.invoke({'question': question, 'document': d.page_content})
            except Exception as e:
                print(e)
            grade: str = score['score']

            # Document relevant
            if grade.lower() == 'yes':
                print('---GRADE: DOCUMENT RELEVANT---')
                filtered_docs.append(d)

            # Document not relevant
            else:
                print('---GRADE: DOCUMENT NOT RELEVANT---')
                continue

        return {'documents': filtered_docs, 'question': question}

    @log_method
    def pass_state(self, state: dict) -> dict:
        """
        Get a new query based on the given state.

        Args:
            state: A dictionary containing the current state.

        Returns:
            A dictionary containing the current state.
        """

        return state

    @log_method
    def generate_subquestions(self, state: dict) -> dict:
        """
        Generates subquestions based on the given question.

        Args:
            state: The input state dict.

        Returns:
            The updated state dict containing the subquestions as a list of strings.
        """

        question: str = state['question']
        subquestions: str = self.subquery_chain.invoke({'question': question})

        print('---SUBQUESTIONS---')
        print(subquestions)
        subquestions_list: List[str] = subquestions.split('\n')

        return {'subquestions': subquestions_list}

    @log_method
    def detect_entities(self, state: dict) -> dict:
        """
        Detects entities in a given question or subquestion.

        Args:
            state: The input state dict.

        Returns:
            The updated state dict containing the detected entities.
        """

        print('---DETERMINING ENTITIES---')
        question: str = state['question']
        subquestions: List[str] = state['subquestions']
        response: dict = {}
        entities: List[str] = []

        try:
            if len(subquestions) == 0:
                response = self.entity_chain.invoke({'question': question})

            else:
                response = self.entity_chain.invoke({'question': subquestions[0]})

            print('entities: ', response['entity_name'])
            if not isinstance(response['entity_name'], list):
                entities = [response['entity_name']]
            else:
                entities = response['entity_name']
            entities = [e.lower() for e in entities]
            entities = [e.replace(',', '') for e in entities]
            # Dumb hack to avoid dealing with the vectorstore logic, for now.
            entities = [e + '.pdf' for e in entities]
            print(entities)
        except:
            entities = []

        return {'entities': entities}

    @log_method
    def determine_cont(self, state: dict) -> str:
        """
        Determine whether to continue or iterate based on the state of the RAG chain.

        Args:
            state: A dictionary containing the current state of the RAG chain.

        Returns:
            Either "continue" if the RAG chain is finished, or "iterate" if it's not.
        """

        subquestions: List[str] = state['subquestions']

        if len(subquestions) == 0:
            print('---FINISHED---')
            return 'continue'
        else:
            print('---ITERATING ON RAG CHAIN---')
            return 'iterate'

    @log_method
    def check_hallucinations(self, state: dict) -> str:
        """
        Checks if the generated text is grounded in the provided documents and addresses the question.

        Args:
            state: The state dictionary containing the question, documents, and generation, and other variables.

        Returns:
            A string indicating the usefulness of the generated text.
        """

        print('---CHECK FOR HALLUCINATIONS---')
        question: str = state['question']
        documents: List[Document] = state['documents']
        generation: str = state['generation']

        docs: str = self._format_docs(documents)
        score: Dict[str, str] = {}

        try:
            score = self.hallucination_chain.invoke({'documents': docs, 'generation': generation})
        except Exception as e:
            print(e)
        print(score)
        grade: str = score['score']

        # Check hallucination
        if grade == 'yes':
            print('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---')
            # Check question-answering
            print('---GRADE GENERATION vs QUESTION---')
            score = self.grading_chain.invoke({'question': question, 'generation': generation})
            grade = score['score']
            if grade == 'yes':
                print('---DECISION: GENERATION ADDRESSES QUESTION---')
                routing = 'useful'
            else:
                print('---DECISION: GENERATION DOES NOT ADDRESS QUESTION---')
                routing = 'not useful'
        else:
            print('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---')
            routing = 'not supported'

        return routing

    @log_method
    def failure_msg(self, state: dict) -> dict:
        """
        This method generates a failure message based on the given state.

        Args:
            state: A dictionary containing the current state of the system.

        Returns:
            The updated state dictionary containing the failure message.
        """

        question: str = state['question']

        failure_msg: str = self.failure_chain.invoke({'question': question})

        return {'answers': failure_msg}

    @log_method
    def aggregate_answers(self, state: dict) -> dict:
        """
        Returns the final answer based on the intermediate answers and original question.

        Args:
            A dictionary containing the intermediate answers and original question.
                The dictionary should have these two keys: "answers" and "original_question".
                "answers" should be a list of strings, and "original_question" should be a string.

        Returns:
            A dictionary with a single key-value pair. The key is "generation",
            and the value is the final answer as a string.
        """

        answers: List[str] = state['answers']
        if isinstance(answers, list):
            answers_str: str = '\n\n'.join(a for a in answers)
        else:
            answers_str = answers
            
        original_question: str = state['original_question']

        print('---ANSWERING---')
        print(original_question)

        print('---INTERMEDIATE ANSWERS---')
        print(answers_str)

        print('---FINAL ANSWER---')
        final_answer: str = self.aggregation_chain.invoke({'question': original_question, 'answers': answers_str})
        print(final_answer)

        return {'generation': final_answer}

    @log_method
    def final_answer(self, state: dict) -> dict:
        """
        This method is used to generate the final answer based on the original question and the generated text.

        Args:
            state: The state dictionary containing the original
            question, the generated text, and other variables.

        Returns:
            The updated state dictionary containing the final answer.
        """

        original_question: str = state['original_question']
        generation: str = state['generation']

        print('---Final Generation---')
        print(generation)

        final_answer: str = self.final_chain.invoke({'question': original_question, 'answers': generation})

        return {'generation': final_answer}
