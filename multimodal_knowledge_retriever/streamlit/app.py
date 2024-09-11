import logging
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

import streamlit as st

from multimodal_knowledge_retriever.src.multimodal import MultimodalRetrieval

logging.basicConfig(level=logging.INFO)
logging.info('URL: http://localhost:8501')


def handle_user_input(user_question: str) -> None:
    if user_question:
        with st.spinner('Processing...'):
            response = st.session_state.qa_chain(user_question)
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response['answer'])

        # List of sources
        sources = set(
            [
                f'{sd.metadata["filename"]} {" - Page "+str(sd.metadata.get("page_number")) \
                            if sd.metadata.get("page_number")else " - "+sd.metadata["file_directory"].split("/")[-1]}'
                for sd in response['source_documents']
            ]
        )
        image_sources = [
            os.path.join(sd.metadata['file_directory'], sd.metadata['filename'])
            for sd in response['source_documents']
            if sd.metadata['filename'].endswith(('.png', '.jpeg', '.jpg'))
        ]
        # Create a Markdown string with each source on a new line as a numbered list with links
        sources_text = ''
        for index, source in enumerate(sources, start=1):
            # source_link = f'<a href="about:blank">{source}</a>'
            source_link = source
            sources_text += f'<font size="2" color="grey">{index}. {source_link}</font>  \n'

        st.session_state.sources_history.append(sources_text)
        st.session_state.image_sources_history.append(image_sources)

    for ques, ans, source, image_source in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
        st.session_state.sources_history,
        st.session_state.image_sources_history,
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.write(f'{ans}')
            if st.session_state.show_sources:
                c1, c2 = st.columns(2)
                with c1.expander('Sources'):
                    st.markdown(
                        f'<font size="2" color="grey">{source}</font>',
                        unsafe_allow_html=True,
                    )
                if image_source:
                    with c2.expander('Images'):
                        for image in image_source:
                            st.image(image)


def main() -> None:
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    if 'multimodal_retriever' not in st.session_state:
        st.session_state.multimodal_retriever = MultimodalRetrieval()
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'sources_history' not in st.session_state:
        st.session_state.sources_history = []
    if 'image_sources_history' not in st.session_state:
        st.session_state.image_sources_history = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = True
    if 'input_disabled' not in st.session_state:
        st.session_state.input_disabled = True

    st.title(':orange[SambaNova] Multimodal Assistant')
    user_question = st.chat_input('Ask questions about your data', disabled=st.session_state.input_disabled)
    handle_user_input(user_question)

    with st.sidebar:
        st.title('Setup')
        st.markdown('**1. Upload your files**')
        docs = st.file_uploader('Add your files', accept_multiple_files=True, type=['pdf', 'jpg', 'jpeg', 'png'])
        st.markdown('**2. Set ingestion steps**')
        table_summaries = st.toggle('summarize Tables', value=True)
        text_summaries = st.toggle('summarize Text', value=False)
        st.markdown('**3. Set retrieval steps**')
        raw_image_retrieval = st.toggle('Answer over raw images', value=True)
        st.caption(
            '**Note** If selected the kit will use raw images to generate the answers, if not, image summaries will be \
                used instead'
        )
        st.markdown('**4. Process your documents and create an in memory vector store**')
        st.caption('**Note:** Depending on the size and number of your documents, this could take several minutes')
        if st.button('Process'):
            if docs:
                with st.spinner('Processing this could take a while...'):
                    st.session_state.qa_chain = st.session_state.multimodal_retriever.st_ingest(
                        docs, table_summaries, text_summaries, raw_image_retrieval
                    )
                    st.toast('Vector DB successfully created!')
                    st.session_state.input_disabled = False
                    st.rerun()
            else:
                st.error('You must provide at least one document', icon='🚨')
        st.markdown('**3. Ask questions about your data!**')


if __name__ == '__main__':
    main()
