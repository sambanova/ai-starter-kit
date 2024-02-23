import os
import sys
sys.path.append("../")
import streamlit as st
import glob
import time
import pandas as pd
from dotenv import load_dotenv
load_dotenv("../export.env")


audio_save_location=("./data/conversations/audio")
transcript_save_location=("./data/conversations/transcription")

def convert_to_dialogue_structure(transcription):
    dialogue = ''  
    for index, row in transcription.iterrows():
        speaker = 'speaker' + str(row['speaker'])
        text = row['text']
        dialogue += speaker + ': ' + text + '\n'   
    return dialogue

def process_audio(audio_path):
    time.sleep(5)
    #TODO make transcription
    #TODO save transcription csv
    return pd.read_csv("./data/conversations/transcription/911_transcript.csv")

def analyse_transcription(trancription):
    dialogue = convert_to_dialogue_structure(trancription) 
    #TODO execute src for dialoge analysis
    return dialogue

def handle_userinput():
    # display block
    st.title("**Post Call Anaysis AI Starter Kit**")
    if st.session_state.transcription is None:
        st.info("Start selecting and processing the input in the side bar ")
    if st.session_state.audio_path:
        st.audio(st.session_state.audio_path)
    if st.session_state.transcription is not None:
        st.dataframe(st.session_state.transcription)
        if st.button("Analyse transcription"):
            if st.session_state.entities_list:
                with st.spinner("Processing"):
                    st.session_state.analysis_result=analyse_transcription(st.session_state.transcription)
            else:
                st.error("You must set classes and entities in Analysis setting sidebar", icon="🚨")
    if st.session_state.analysis_result:
        st.write(st.session_state.analysis_result)
                


def main():
    
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
    )

    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcript_path" not in st.session_state:
        st.session_state.transcript_path = None
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "classes_list" not in st.session_state:
        st.session_state.classes_list = ["undefined"]
    if "entities_list" not in st.session_state:
        st.session_state.entities_list = []
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
        
    # Sidebar
    with st.sidebar:
        st.title("**SetUp**")
        # Audio section
        st.title("Audio input")
        with st.expander("Audio input settings"):
            st.markdown("**1. Select audio source**")
            datasource = st.selectbox(
                    "Select source", ("Upload your own call recording", "Select a preset audio")
            )
            if "Upload" in datasource:
                st.markdown("**2. Upload file:**")
                audio_file = st.file_uploader(
                    "Add WAV file", accept_multiple_files=False, type="wav"
                )
                if st.button("Save audio"):
                    if audio_file:
                        with st.spinner("uploading"):
                            st.session_state.audio_path = os.path.join(audio_save_location,audio_file.name)
                            with open(st.session_state.audio_path, "wb") as f:
                                f.write(audio_file.getvalue())
                            st.toast("Audio saved in " + audio_save_location)
                    else:
                        st.error("You must provide an audio file", icon="🚨")
            elif "Select" in datasource:
                st.markdown("**2. Select preset**")
                audios = glob.glob(os.path.join(audio_save_location, '*.wav'))
                selected_audio = st.selectbox("presets", audios)
                if st.button("Select audio preset"):
                    st.session_state.audio_path = selected_audio
            st.markdown("**3. Get transcription**")
            if st.button("process audio"):
                if st.session_state.audio_path:
                    with st.spinner("Processing"):
                        st.session_state.transcription=process_audio(st.session_state.audio_path)
                else:
                    st.error("You must provide an audio file", icon="🚨")
        
        # Transcript Section
        st.title("Text transcript input") 
        with st.expander("Transcript input settings"):
            st.markdown("**1. Select Trascription source**")
            datasource = st.selectbox(
                    "Select source", ("Upload your own transcription file", "Select a preset transcription file")
            )  
            if "Upload" in datasource:
                st.markdown("**2. Upload file:**")
                transcription_file = st.file_uploader(
                    "Add csv file", accept_multiple_files=False, type="csv"
                )
                if st.button("Save transcription"):
                    if transcription_file:
                        with st.spinner("uploading"):
                            st.session_state.transcript_path = os.path.join(transcript_save_location,transcription_file.name)
                            with open(st.session_state.transcript_path, "wb") as f:
                                f.write(transcription_file.getvalue())
                            st.toast("Transcription file saved in " + transcript_save_location)
                            st.session_state.transcription=pd.read_csv(st.session_state.transcript_path)
                    else:
                        st.error("You must provide a trasncription file", icon="🚨")
            elif "Select" in datasource:
                st.markdown("**2. Select preset**")
                transcriptions = glob.glob(os.path.join(transcript_save_location, '*.csv'))
                selected_transcript = st.selectbox("presets", transcriptions)
                if st.button("Select transcription preset"):
                    st.session_state.transcript_path = selected_transcript
                    st.session_state.transcription=pd.read_csv(st.session_state.transcript_path)
                
        # Analysis setup
        st.title("Analysis Setings")
        st.markdown("**1. Include the main topic classes to classify**")
        col_a1, col_a2 = st.columns((3,2))
        new_class = col_a1.text_input("Add class:", "general information request")
        col_a2.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(2)
                {
                    padding-top: 3%;
                } 
            </style>
            """,unsafe_allow_html=True
        )
        if col_a2.button("Include class") and new_class:
            st.session_state.classes_list.append(new_class) 
        with st.expander(f"{len(st.session_state.classes_list)} Classes",expanded=True):
            st.write(st.session_state.classes_list)
        if st.button("Clear Classes List"):
            st.session_state.classes_list = ["undefined"]
            st.experimental_rerun()
            
        st.markdown("**2. Include entities to extract**")
        col_b1, col_b2 = st.columns((3,2))
        new_entity = col_b1.text_input("Add entity:", "names")
        col_b2.markdown(
            """
            <style>
                div[data-testid="column"]:nth-of-type(2)
                {
                    padding-top: 3%;
                } 
            </style>
            """,unsafe_allow_html=True
        )
        if col_b2.button("Include entity") and new_entity:
            st.session_state.entities_list.append(new_entity) 
        with st.expander(f"{len(st.session_state.entities_list)} Entities",expanded=True):
            st.write(st.session_state.entities_list)
        if st.button("Clear Entities List"):
            st.session_state.entities_list = []
            st.experimental_rerun()
                                
    handle_userinput()
if __name__ == "__main__":
    main()
