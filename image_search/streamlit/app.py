import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))

sys.path.append(kit_dir)
sys.path.append(repo_dir)


from typing import Any, List

import streamlit as st

from image_search.src.image_search import ImageSearch

st.set_page_config(
    page_title='AI Starter Kit',
    page_icon=os.path.join(repo_dir, 'images', 'SambaNova-icon.svg'),
    layout='wide',
)

if 'image_search' not in st.session_state:
    st.session_state.image_search = None
if 'images_path' not in st.session_state:
    st.session_state.images_path = None
if 'db_path' not in st.session_state:
    st.session_state.db_path = None
if 'top_number' not in st.session_state:
    st.session_state.top_number = 3
if 'search_disabled' not in st.session_state:
    st.session_state.search_disabled = True

st.header('Image Search AI starter kit', divider='red')
with st.sidebar:
    st.header('App Settings', divider='red')

    datasource = st.selectbox(
        '**1. Pick a datasource**', ('Set images folder path (create new vector db)', 'Use existing vector db')
    )
    if datasource is not None and 'Set' in datasource:
        st.session_state.images_path = st.text_input(
            '**2. Set the absolute path to your images folder**',
            placeholder='E.g., /Users/<username>/Downloads/<images_folder>',
        ).strip()
        if st.button('Process'):
            with st.spinner('Processing, this could take several minutes'):
                if os.path.exists(st.session_state.images_path):
                    st.session_state.image_search = ImageSearch()
                    st.session_state.image_search.init_collection()
                    st.session_state.image_search.add_images(st.session_state.images_path)
                    st.toast(f'vector_db stored in {os.path.join(kit_dir,"/data/vector_db")}')
                    st.session_state.search_disabled = False
                else:
                    st.toast('the provided image path does not exist')

    elif datasource is not None and 'Use' in datasource:
        st.session_state.db_path = st.text_input(
            '**2. Set th absolute path to your Chroma Vector DB folder**',
            placeholder='E.g., /Users/<username>/Downloads/<vectordb_folder>',
        ).strip()
        if st.button('Load vectordb'):
            with st.spinner('Loading'):
                if os.path.exists(st.session_state.db_path):
                    st.session_state.image_search = ImageSearch(path=st.session_state.db_path)
                    st.session_state.image_search.init_collection()
                    st.session_state.search_disabled = False
                else:
                    st.toast('the provided vector_db path does not exist')
    st.session_state.top_number = st.slider('Number of Search Results', value=3, min_value=1, max_value=10)


col1, col2 = st.columns([2, 3])

search_results: Any | List[Any] = []

with col1:
    choise = st.radio(
        'Search by:',
        ['**Description**', '**Image**'],
        captions=['Write the description of the image you want to search', 'Upload an image and search similar images'],
        disabled=st.session_state.search_disabled,
    )
if choise == '**Description**':
    with col2:
        st.header('Search by description')
        search_term = st.text_input('Search')
    if st.button('Search', disabled=st.session_state.search_disabled, type='primary'):
        if search_term:
            with st.spinner('Searching'):
                paths, distances = st.session_state.image_search.search_image_by_text(
                    f'A picture of {search_term}', st.session_state.top_number
                )
                search_results = zip(paths, distances)
        else:
            st.toast('You need to write a description first')
elif choise == '**Image**':
    with col2:
        st.header('Search by Image')
        uploaded_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if st.button('Search', disabled=st.session_state.search_disabled, type='primary'):
        if uploaded_image is not None:
            with st.spinner('Searching'):
                paths, distances = st.session_state.image_search.search_image_by_image(
                    uploaded_image, st.session_state.top_number
                )
                search_results = zip(paths, distances)
        else:
            st.toast('You need to upload an image first')

st.markdown('***')
if search_results:
    st.subheader('Search Results:')
    for path, distance in search_results:
        st.image(path, width=1000)
        st.write(f'Distance: {distance}')
        st.markdown('***')
