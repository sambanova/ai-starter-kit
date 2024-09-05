import os
import sys
import logging
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, ".."))
repo_dir = os.path.abspath(os.path.join(kit_dir, ".."))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st
from utils.vectordb.vector_db import VectorDb
from utils.agents.corrective_rag import CorrectiveRAG, RAGGraphState
from utils.agents.teams.tavily_search_team import TavilySearchTeam
from utils.agents.teams.return_message_team import ReturnTeam
from utils.agents.teams.corrective_rag_team import CRAGSupervisor
from utils.agents.teams.corrective_rag_team import TeamCRAG
from utils.model_wrappers.api_gateway import APIGateway 
from utils.parsing.sambaparse import parse_doc_streamlit


CONFIG_PATH = os.path.join(kit_dir,'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir,f"data/my-vector-db")

logging.basicConfig(level=logging.INFO)
logging.info("URL: http://localhost:8501")


def get_config_info(CONFIG_PATH: str):
        """
        Loads json config file
        """
        # Read config file
        with open(CONFIG_PATH, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config["api"]
        llm_info =  config["llm"]
        embedding_model_info = config["embedding_model"]
        retrieval_info = config["retrieval"]
        prompts = config["prompts"]
        
        return api_info, llm_info, embedding_model_info, retrieval_info, prompts

def load_embedding_model(embedding_model_info: dict) -> None:
        embeddings = APIGateway.load_embedding_model(
            type=embedding_model_info["type"],
            batch_size=embedding_model_info["batch_size"],
            coe=embedding_model_info["coe"],
            select_expert=embedding_model_info["select_expert"]
            ) 
        return embeddings  


initial_state = RAGGraphState

session_uuid = "1234"

config = {
        "configurable": {
            "thread_id": session_uuid
        }
    }

def handle_userinput(app, supervisor_app, user_question):
    if user_question:
        with st.spinner("Processing..."):
            response, current_state = st.session_state.conversation.call_rag(st.session_state.app, 
            question=user_question, 
            config=config)

        update_dict = {
            "query_history": [user_question],
            "answer_history": [response["answer"]]
            }
        print(update_dict)
        supervisor_app.update_state(config=config, values=update_dict)
        print(app.get_state(config=config))
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response["answer"])
        st.session_state.state = current_state

        # List of sources
        # try:
        sources =set([
            f'{sd.metadata["filename"]}'
            for sd in response["source_documents"]
            ])
        # except:
            # sources = ["Tavily Internet Search"]

        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ""
        for index, source in enumerate(sources, start=1):
            # source_link = f'<a href="about:blank">{source}</a>'
            source_link = source
            sources_text += (
                f'<font size="2" color="grey">{index}. {source_link}</font>  \n'
            )
        st.session_state.sources_history.append(sources_text)
        # update_dict = {
        #     "query_history": st.session_state.chat_history[::2],
        #     "message_history": st.session_state.chat_history[1::2]
        # }


        # app.update_state(config=config, values=update_dict)

    for ques, ans, source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
    ):
        with st.chat_message("user"):
            st.write(f"{ques}")

        with st.chat_message(
            "ai",
            avatar="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
        ):
            st.write(f"{ans}")
            if st.session_state.show_sources:
                with st.expander("Sources"):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )

def main(): 

    vdb = VectorDb()
    _, _, embedding_model_info, _, _ = get_config_info(CONFIG_PATH=CONFIG_PATH)
    embeddings = load_embedding_model(embedding_model_info=embedding_model_info)
    default_collection = 'agent_workflows_default_collection'

    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = True
    if "sources_history" not in st.session_state:
        st.session_state.sources_history = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "app" not in st.session_state:
        st.session_state.app = None
    if "supervisor" not in st.session_state:
        st.session_state.supervisor = None
    if "state" not in st.session_state:
        st.session_state.state = initial_state

    with st.sidebar:
        st.title("Setup")
        st.markdown("**1. Pick a datasource**")
        datasource = st.selectbox(
            "", ("Upload files (create new vector db)", "Use existing vector db")
        )
        if "Upload" in datasource:
            docs = st.file_uploader(
                "Add PDF or TXT files", accept_multiple_files=True, type=["pdf","txt"]
            )
            st.markdown("**2. Process your documents and create vector store**")
            st.markdown(
                "**Note:** Depending on the size and number of your documents, this could take several minutes"
            )
            st.markdown("Create database")
            if st.button("Process"):
                with st.spinner("Processing"):
                    # get pdf text
                    # get the text chunks
                    text_chunks = parse_doc_streamlit(docs, kit_dir=kit_dir)
                    # create vector store
                    st.session_state.embeddings = embeddings
                    vectorstore = vdb.create_vector_store(text_chunks, embeddings, db_type="chroma", output_db=None)
                    st.session_state.vectorstore = vectorstore

                    # instantiate rag
                    rag = CorrectiveRAG(
                    configs=CONFIG_PATH,
                    embeddings = st.session_state.embeddings,
                    vectorstore=st.session_state.vectorstore,
                    )
                    
                    # Initialize chains
                    rag.initialize()

                    # Build nodes
                    workflow = rag.create_rag_nodes()

                    # Build graph
                    rag_app = rag.build_rag_graph(workflow)

                    search = TavilySearchTeam(
                    configs=CONFIG_PATH,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                    )
                    search.initialize()
                    workflow = search.create_search_nodes()
                    search_app = search.build_search_graph(workflow)

                    return_msg = ReturnTeam(
                    configs=CONFIG_PATH,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                    )
                    return_msg.initialize()
                    return_app = return_msg.create_return_team()

                    sup = CRAGSupervisor(
                    configs=CONFIG_PATH,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                    )
                    sup.initialize()
                    supervisor_app = sup.create_supervisor()
                    st.session_state.supervisor = supervisor_app

                    team = TeamCRAG(
                    supervisor_app=supervisor_app,
                    rag_app=rag_app,
                    search_app=search_app,
                    return_app=return_app,
                    )

                    team.create_team_graph()
                    app = team.build_team_graph()
                    st.session_state.app = app


                    st.session_state.conversation = team 
                    st.toast(f"File uploaded! Go ahead and ask some questions",icon='🎉')
            st.markdown("[Optional] Save database for reuse")
            save_location = st.text_input("Save location", "./data/my-vector-db").strip()
            if st.button("Process and Save database"):
                with st.spinner("Processing"):
                    # get pdf text
                    # get the text chunks
                    text_chunks = parse_doc_streamlit(docs, kit_dir=kit_dir)
                    # create vector store
                    st.session_state.embeddings = embeddings
                    vectorstore = vdb.create_vector_store(text_chunks, embeddings,output_db=save_location,
                                                           db_type="chroma", collection_name=default_collection)
                    st.session_state.vectorstore = vectorstore

                    # instantiate rag
                    rag = CorrectiveRAG(
                    configs=CONFIG_PATH,
                    embeddings = st.session_state.embeddings,
                    vectorstore=st.session_state.vectorstore,
                    )
                    
                    # Initialize chains
                    rag.initialize()

                    # Build nodes
                    workflow = rag.create_rag_nodes()

                    # Build graph
                    rag_app = rag.build_rag_graph(workflow)

                    search = TavilySearchTeam(
                    configs=CONFIG_PATH,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                    )
                    search.initialize()
                    workflow = search.create_search_nodes()
                    search_app = search.build_search_graph(workflow)

                    return_msg = ReturnTeam(
                    configs=CONFIG_PATH,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                    )
                    return_msg.initialize()
                    return_app = return_msg.create_return_team()

                    sup = CRAGSupervisor(
                    configs=CONFIG_PATH,
                    embeddings=embeddings,
                    vectorstore=vectorstore,
                    )
                    sup.initialize()
                    supervisor_app = sup.create_supervisor()
                    st.session_state.supervisor = supervisor_app

                    team = TeamCRAG(
                    supervisor_app=supervisor_app,
                    rag_app=rag_app,
                    search_app=search_app,
                    return_app=return_app,
                    )

                    team.create_team_graph()
                    app = team.build_team_graph()
                    st.session_state.app = app


                    st.session_state.conversation = team 
                    st.toast(f"File uploaded and saved to {PERSIST_DIRECTORY}! Go ahead and ask some questions",icon='🎉')

        else:
            db_path = st.text_input(
                f"Absolute path to your DB folder",
                placeholder="E.g., /Users/<username>/path/to/your/vectordb",
            ).strip()
            st.markdown("**2. Load your datasource and create vectorstore**")
            st.markdown(
                "**Note:** Depending on the size of your vector database, this could take a few seconds"
            )
            if st.button("Load"):
                print(db_path)
                with st.spinner("Loading vector DB..."):
                    if db_path == "":
                        st.error("You must provide a provide a path", icon="🚨")
                    else:
                        if os.path.exists(db_path):
                            # load the vectorstore
                            st.session_state.embeddings = embeddings
                            vectorstore = vdb.load_vdb(db_path, embeddings, collection_name=default_collection)
                            st.toast("Database loaded")

                            # assign vectorstore to session
                            st.session_state.vectorstore = vectorstore

                            # instantiate rag
                            rag = CorrectiveRAG(
                            configs=CONFIG_PATH,
                            embeddings = st.session_state.embeddings,
                            vectorstore=st.session_state.vectorstore,
                            )
                            
                            # Initialize chains
                            rag.initialize()

                            # Build nodes
                            workflow = rag.create_rag_nodes()

                            # Build graph
                            rag_app = rag.build_rag_graph(workflow)

                            search = TavilySearchTeam(
                            configs=CONFIG_PATH,
                            embeddings=embeddings,
                            vectorstore=vectorstore,
                            )
                            search.initialize()
                            workflow = search.create_search_nodes()
                            search_app = search.build_search_graph(workflow)

                            return_msg = ReturnTeam(
                            configs=CONFIG_PATH,
                            embeddings=embeddings,
                            vectorstore=vectorstore,
                            )
                            return_msg.initialize()
                            return_app = return_msg.create_return_team()

                            sup = CRAGSupervisor(
                            configs=CONFIG_PATH,
                            embeddings=embeddings,
                            vectorstore=vectorstore,
                            )
                            sup.initialize()
                            supervisor_app = sup.create_supervisor()
                            st.session_state.supervisor = supervisor_app

                            team = TeamCRAG(
                            supervisor_app=supervisor_app,
                            rag_app=rag_app,
                            search_app=search_app,
                            return_app=return_app,
                            )

                            team.create_team_graph()
                            app = team.build_team_graph()
                            st.session_state.app = app


                            st.session_state.conversation = team # assign conversation to session
                        else:
                            st.error("database not present at " + db_path, icon="🚨")

        st.markdown("**3. Ask questions about your data!**")

        with st.expander("Additional settings", expanded=True):
            st.markdown("**Interaction options**")
            st.markdown(
                "**Note:** Toggle these at any time to change your interaction experience"
            )
            show_sources = st.checkbox("Show sources", value=True, key="show_sources")

            st.markdown("**Reset chat**")
            st.markdown(
                "**Note:** Resetting the chat will clear all conversation history"
            )
            if st.button("Reset conversation"):
                # reset create conversation chain
                st.session_state.conversation = documentRetrieval.get_qa_retrieval_chain(
                    st.session_state.vectorstore
                )
                st.session_state.chat_history = []
                st.toast(
                    "Conversation reset. The next response will clear the history on the screen"
                )

    st.title(":orange[SambaNova] Analyst Assistant")
    user_question = st.chat_input("Ask questions about your data")
    handle_userinput(st.session_state.app, st.session_state.supervisor, user_question)


if __name__ == "__main__":
    main()
