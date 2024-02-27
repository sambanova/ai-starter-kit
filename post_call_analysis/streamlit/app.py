import os
import sys
sys.path.append("../")
import streamlit as st
from streamlit_javascript import st_javascript
import glob
import time
import pandas as pd
from post_call_analysis.src import analysis, plot


audio_save_location=("./data/conversations/audio")
transcript_save_location=("./data/conversations/transcription")

def convert_to_dialogue_structure(transcription):
    dialogue = ''  
    for _, row in transcription.iterrows():
        speaker = str(row['speaker'])
        text = str(row['text'])
        dialogue += speaker + ': ' + text + '\n'   
    return dialogue

def process_audio(audio_path):
    time.sleep(0.5)
    #TODO make transcription
    #TODO save transcription csv
    return pd.read_csv("./data/conversations/transcription/911_transcript.csv")

def analyse_transcription(transcription, transcription_path, facts_path ,classes, entities):
    dialogue = convert_to_dialogue_structure(transcription) 
    conversation = analysis.load_conversation(dialogue, transcription_path)
    result=analysis.call_analysis_parallel(conversation.page_content, documents_path=facts_path, classes_list=classes, entities_list=entities)
    return result

def handle_userinput():
    if st_javascript("""function darkMode(i){return (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)}(1)"""):
        dark_mode = True
    else:
        dark_mode = False
    #display block
    st.title("Post Call Anaysis AI Starter Kit")
    if st.session_state.transcription is None:
        st.info("Start selecting and processing the input in the side bar ")
    if st.session_state.audio_path:
        st.title("**Call audio to process**")
        st.audio(st.session_state.audio_path)
    if st.session_state.transcription is not None:
        st.title("**Call transcript to process**")
        if st.session_state.audio_path:
            #diarization_plot = plot.plot_diarization(st.session_state.audio_path , st.session_state.transcription, dark_mode)
            #st.pyplot(diarization_plot)
            plot_diarization_plotly = plot.plot_diarization_plotly(st.session_state.audio_path , st.session_state.transcription)
            st.plotly_chart(plot_diarization_plotly, use_container_width=True)
        st.markdown("**Transcription**")
        st.dataframe(st.session_state.transcription, use_container_width=True)
        if st.button("Analyse transcription"):
            if st.session_state.entities_list and st.session_state.facts_path and st.session_state.classes_list:
                with st.spinner("**Processing** this could take some minutes"):
                    st.session_state.analysis_result=analyse_transcription(st.session_state.transcription,
                                                                           st.session_state.transcript_path,
                                                                           st.session_state.facts_path ,
                                                                           st.session_state.classes_list,
                                                                           st.session_state.entities_list)
            else:
                st.error("You must set classes, entities and factual check documents path in Analysis setting sidebar", icon="🚨")
    if st.session_state.analysis_result:
        st.title("Analysis Results")
        with st.container(border=True):
            st.markdown("**Conversation Summary**")
            st.markdown(st.session_state.analysis_result["summary"])
        c1, c2 = st.columns((5,4),gap="small")
        with c1:
            with st.container(border=True):
                st.markdown("**classification**")
                st.write(st.session_state.analysis_result["classification"])  
            with st.container(border=True):
                st.markdown("**Sentiment analysis**")
                st.write(st.session_state.analysis_result["sentiment"])
            with st.container(border=True):        
                st.markdown("**Factual Accuracy Analysis**")
                st.write(st.session_state.analysis_result["factual_analysis"])
        with c2: 
            with st.container(border=True):
                st.markdown("**Extracted entities**")
                entities_items = st.session_state.analysis_result["entities"].items()
                entities_tabs= st.tabs(st.session_state.analysis_result["entities"].keys())
                for tab, item in zip(entities_tabs, entities_items):
                    with tab:
                        st.write(item[1])
            with st.container(border=True):
                st.markdown("**Call quality Assessment**")
                quallity_gague = plot.plot_quallity_gauge(st.session_state.analysis_result["quality_score"])
                st.plotly_chart(quallity_gague)         
        
        #st.write(st.session_state.analysis_result)
                


def main():
    
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/wp-content/uploads/2021/05/logo_icon-footer.svg",
        layout="wide"
    )

    if "audio_path" not in st.session_state:
        st.session_state.audio_path = None
    if "transcript_path" not in st.session_state:
        st.session_state.transcript_path = None
    if "transcription" not in st.session_state:
        st.session_state.transcription = None
    if "classes_list" not in st.session_state:
        st.session_state.classes_list = ["undefined", "emergency", "general information", "sales", "complains"]
    if "entities_list" not in st.session_state:
        st.session_state.entities_list = ["name", "address", "city", "phone number"]
    if "facts_path" not in st.session_state:
        st.session_state.facts_path = None
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
        with st.expander("Audio input settings"):
            st.markdown("**1. Include the main topic classes to classify**")
            col_a1, col_a2 = st.columns((3,2))
            new_class = col_a1.text_input("Add class:", "general information request")
            col_a2.markdown('#')
            if col_a2.button("Include class") and new_class:
                st.session_state.classes_list.append(new_class) 
            st.write(st.session_state.classes_list)
            if st.button("Clear Classes List"):
                st.session_state.classes_list = []
                st.experimental_rerun()
                
            st.markdown("**2. Include entities to extract**")
            col_b1, col_b2 = st.columns((3,2))
            new_entity = col_b1.text_input("Add entity:", "names")
            col_b2.markdown('#')
            if col_b2.button("Include entity") and new_entity:
                st.session_state.entities_list.append(new_entity) 
            st.write(st.session_state.entities_list)
            if st.button("Clear Entities List"):
                st.session_state.entities_list = []
                st.experimental_rerun()
                
            st.markdown("**3. Select path with documents for factual check**")
            col_c1, col_c2 = st.columns((3,2))
            facts_path = col_c1.text_input("set factual check documents path", "./data/documents")
            col_c2.markdown('#')
            if col_c2.button("set path") and facts_path:
                if os.path.exists(facts_path):
                    st.session_state.facts_path=(facts_path) 
                else:
                    st.error(f"{facts_path} does not exist", icon="🚨")                            
    handle_userinput()
    
if __name__ == "__main__":
    main()
