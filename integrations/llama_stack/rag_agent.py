# Create http client
from llama_stack_client import LlamaStackClient

from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent


def main() -> None:
    client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

    # Register a vector db
    vector_db_id = 'my_documents'
    response = client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model='all-MiniLM-L6-v2',
        embedding_dimension=384,
        provider_id='faiss',
    )

    # You can insert a pre-chunked document directly into the vector db
    chunks = [
        {
            'document_id': 'doc1',
            'content': 'Your document text here',
            'mime_type': 'text/plain',
        },
    ]
    client.vector_io.insert(vector_db_id=vector_db_id, chunks=chunks)

    # You can then query for these chunks
    chunks_response = client.vector_io.query(vector_db_id=vector_db_id, query='What do you know about...')

    urls = ['memory_optimizations.rst', 'chat.rst', 'llama3.rst']
    documents = [
        Document(
            document_id=f'num-{i}',
            content=f'https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}',
            mime_type='text/plain',
            metadata={},
        )
        for i, url in enumerate(urls)
    ]

    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=512,
    )

    # Query documents
    results = client.tool_runtime.rag_tool.query(
        vector_db_ids=[vector_db_id],
        content='What do you know about...',
    )

    # Create agent with memory
    agent = Agent(
        client,
        model='meta-llama/Llama-3.3-70B-Instruct',
        instructions='You are a helpful assistant',
        tools=[
            {
                'name': 'builtin::rag/knowledge_search',
                'args': {
                    'vector_db_ids': [vector_db_id],
                },
            }
        ],
    )
    session_id = agent.create_session('rag_session')

    # Ask questions about documents in the vector db, and the agent will query the db to answer the question.
    response = agent.create_turn(
        messages=[{'role': 'user', 'content': 'How to optimize memory in PyTorch?'}],
        session_id=session_id,
    )

    # Initial document ingestion
    response = agent.create_turn(
        messages=[{'role': 'user', 'content': 'I am providing some documents for reference.'}],
        documents=[
            {
                'content': 'https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/memory_optimizations.rst',
                'mime_type': 'text/plain',
            }
        ],
        session_id=session_id,
    )

    # Query with RAG
    response = agent.create_turn(
        messages=[{'role': 'user', 'content': 'What are the key topics in the documents?'}],
        session_id=session_id,
    )


if __name__ == '__main__':
    main()
