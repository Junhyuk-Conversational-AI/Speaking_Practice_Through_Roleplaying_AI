import io
import base64
from openai import OpenAI, audio
import requests
from openai.types.audio import transcription
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
os.environ["OPENAI_API_KEY"] = "Your API Key"
# To run this code
# streamlit run app.py

# Set page configuration for wide layout
st.set_page_config(page_title="Roleplay", layout="wide")

host_url = "http://localhost:8000"
chat_url = f"{host_url}/chat"
transcribe_url = f"{host_url}/transcribe"


if "curr_page" not in st.session_state:
    st.session_state["curr_page"] = "home"


if "messages" not in st.session_state:
    st.session_state.messages = []


if "prev_audio_bytes" not in st.session_state:
    st.session_state.prev_audio_bytes = None


# Assuming you have a dictionary that holds your data like below:
roleplays = {
    'hamburger': {"display_name":'Ordering a hamburger', 'emoji': '🍔', 'difficulty': '⭐️'},
    'immigration': {'display_name':'Going through immigration', 'emoji': '🏦', 'difficulty': '⭐️'},

    'bank': {'display_name':'Getting a loan at the bank', 'emoji': '🏦', 'difficulty': '⭐️⭐️⭐️'},
    'school': {'display_name': 'New semester classroom', 'emoji': '🏫', 'difficulty': '⭐️⭐️'},
    'caffe': {'display_name': 'Ordering coffee', 'emoji': '☕️', 'difficulty': '⭐️'},
    'massage': {'display_name': 'Restaurant Reservation', 'emoji': '📞', 'difficulty': '⭐️⭐️'},
}


def go_to_chat():
    st.session_state["curr_page"] = "chat"

def go_to_home():
    st.session_state["curr_page"] = "home"
    st.session_state["roleplay"] = None
    st.session_state["roleplay_info"] = None
    st.session_state.messages = []


def roleplay_start(roleplay):
    st.session_state["roleplay"] = roleplay
    st.session_state["roleplay_info"] = roleplays[roleplay]

    go_to_chat()


# Create a function to display each roleplay in the grid
def display_roleplay(roleplay, roleplay_info, key):
    with st.container(border=True):
        st.write(f"**{roleplay_info['display_name']}**")
        st.write(roleplay_info['emoji'])
        # st.progress(int(roleplay_info['progress'].replace('%', '')), "진도")
        st.write(f"Difficulty {roleplay_info['difficulty']}")
        st.button("Start", key=f"btn_start_roleplay_{key}", on_click=roleplay_start, kwargs=dict(roleplay=roleplay))


if  st.session_state["curr_page"] == "home":

    st.title("Roleplay")
    cols = st.columns(2)
    for i, (roleplay, roleplay_info) in enumerate(roleplays.items()):
        with cols[i % 2]: 
            display_roleplay(roleplay, roleplay_info, i)


elif  st.session_state["curr_page"] == "chat":
    client = OpenAI()
    roleplay = st.session_state["roleplay"]
    roleplay_info = roleplays[roleplay]
    st.title(roleplay_info['display_name'])


    ###############################################
    # Helpers
    def stt(audio_bytes):
        audio_file = io.BytesIO(audio_bytes)
        files = {"audio_file": ("audio.wav", audio_file, "audio/wav")}
        response = requests.post(transcribe_url, files=files)
        return response.json()

    def chat(text, roleplay = None):
        user_turn = {"role": "user", "content": text}
        messages = st.session_state.messages + [user_turn]

        # send it to host url (backend) - http://localhost:8000/chat
        resp = requests.post(chat_url + f"/{roleplay}", json={"messages":messages})
        assistant_turn = resp.json()

        return assistant_turn['content']


    @st.cache_data
    def get_goals(roleplay):
        resp = requests.get(f"{host_url}/{roleplay}/goals")
        goals = resp.json()
        return goals 
        


    @st.cache_data
    def check_goals(messages, roleplay):
        resp = requests.post(f"{host_url}/{roleplay}/check_goals",
                             json={"messages": messages})
        goals = resp.json()
        return goals
        



    def get_policy_viloated(text):
        response = client.moderations.create(input=text)
        output = response.results[0]
        output_dict = output.model_dump()
        flagged_list = []
        for k, v in output_dict['categories'].items():
            if v:
                score = output_dict['category_scores'][k]
                flagged_list.append((k, score))
        return flagged_list
    

    def autoplay_audio(file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(
                md,
                unsafe_allow_html=True,
            )
   
    ###############################################
    # Conversation
    speech_file_path = "tmp_speak.mp3"
    if "goal_list" not in st.session_state:
        st.session_state.goal_list = get_goals(roleplay)

    goal_text = "\n".join([f"- {goal}" for goal in st.session_state.goal_list])
    goal_result = ""

    with st.container(border=True, height=480):
        con1 = st.container()
        con2 = st.container()
    
    user_input = ""
    
    
    with con2:
        audio_bytes = audio_recorder("talk", pause_threshold=3.0)
        if audio_bytes == st.session_state.prev_audio_bytes:
            audio_bytes = None
        st.session_state.prev_audio_bytes = audio_bytes
    
        try:
            if audio_bytes:
                with st.spinner("Voice recognition in progress..."):
                    resp_stt = stt(audio_bytes)
                    status = resp_stt['status']
                    if status == 'ok':
                        user_input = resp_stt['text']
        except Exception as e:
            print(e)
            pass
    
    
    with con1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
    
            flag_list = get_policy_viloated(user_input)
            with st.chat_message("user"):
    
                st.markdown(user_input)
                if flag_list:
                    st.warning(flag_list)
    
            with st.chat_message("assistant"):

                with st.spinner("Thinking..."):
                    bot_output = chat(user_input, roleplay)
    
                with st.spinner("Generating voice...."):
                    response = client.audio.speech.create(
                      model="tts-1",
                      voice="alloy", # alloy, echo, fable, onyx, nova, and shimmer
                      input=bot_output
                    )
                    response.stream_to_file(speech_file_path)

                # 목표 체크
                with st.spinner("Checking goals..."):
                    goal_result = check_goals(st.session_state.messages, roleplay)

                st.markdown(bot_output)
                autoplay_audio(speech_file_path)

                if "[END]" in bot_output:
                    st.balloons()
                    go_to_home()
    
            st.session_state.messages.append({"role": "assistant", "content": bot_output})


    with st.container(border=True):
        st.markdown("### Goal")
        if goal_result:
            st.markdown("\n".join([f"- {st.session_state.goal_list[g['goal_number']]}: {'✅' if g['accomplished'] else '❌'} " for g in goal_result['goal_list']]))
        else:
            st.markdown(goal_text)



