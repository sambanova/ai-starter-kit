import logging
import os
import sys
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)

from function_calling.src.function_calling import FunctionCallingLlm
from function_calling.src.tools import calculator, get_time, python_repl, query_db

logging.basicConfig(level=logging.INFO)

TOOLS = {
    'get_time': get_time,
    'calculator': calculator,
    'python_repl': python_repl,
    'query_db': query_db,
}


@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


def set_fc_llm(tools):
    set_tools = [TOOLS[name] for name in tools]
    st.session_state.fc = FunctionCallingLlm('sambaverse', set_tools)


def handle_userinput(user_question):
    global output
    if user_question:
        with st.spinner('Processing...'):
            with st_capture(output.code):
                response = st.session_state.fc.function_call_llm(
                    query=user_question, max_it=st.session_state.max_iterations, debug=True
                )

        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)

    for ques, ans in zip(
        st.session_state.chat_history[::2],
        st.session_state.chat_history[1::2],
    ):
        with st.chat_message('user'):
            st.write(f'{ques}')

        with st.chat_message(
            'ai',
            avatar='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
        ):
            st.write(f'{ans}')


def main():
    global output
    st.set_page_config(
        page_title='AI Starter Kit',
        page_icon='https://sambanova.ai/hubfs/logotype_sambanova_orange.png',
    )

    if 'fc' not in st.session_state:
        st.session_state.fc = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'tools' not in st.session_state:
        st.session_state.tools = ['get_time', 'calculator']
    if 'max_iterations' not in st.session_state:
        st.session_state.max_iterations = 5

    st.title(':orange[SambaNova] Function Calling Assistant')
    user_question = st.chat_input('Ask something')

    with st.sidebar:
        st.title('Setup')
        st.markdown('**1. Select the tools you want the model to have access to**')
        st.session_state.tools = st.multiselect(
            'Available tools', ['get_time', 'calculator', 'python_repl', 'query_db'], ['get_time', 'calculator']
        )
        st.markdown('**2. Set the maximum number of iterations your want the model to run**')
        st.number_input('Max iterations', value=5, max_value=20)
        st.markdown('**Note:** The response could be not completed if the max number of iterations is to low')
        if st.button('Set'):
            with st.spinner('Processing'):
                set_fc_llm(st.session_state.tools)
                st.toast(f'Tool calling assistant set! Go ahead and ask some questions', icon='🎉')

        st.markdown('**3. Ask questions about your data!**')

        with st.expander('**Execution scratchpad**', expanded=True):
            output = st.empty()

        with st.expander('Additional settings', expanded=False):
            st.markdown('**Interaction options**')

            st.markdown('**Reset chat**')
            st.markdown('**Note:** Resetting the chat will clear all interactions history')
            if st.button('Reset conversation'):
                st.session_state.chat_history = []
                st.session_state.sources_history = []
                st.toast('Interactions reset. The next response will clear the history on the screen')

    handle_userinput(user_question)


if __name__ == '__main__':
    main()
