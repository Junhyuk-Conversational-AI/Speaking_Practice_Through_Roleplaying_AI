from typing import List
from fastapi import FastAPI, UploadFile, File
from openai.resources.beta.threads import messages
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import langchain_core.pydantic_v1 as pyd1
import pydantic as pyd2
from openai import OpenAI
import os
# To run this code
# uvicorn backend:app --reload

os.environ["OPENAI_API_KEY"] = "Your API Key"
client = OpenAI()
model = ChatOpenAI(model="gpt-4")
app = FastAPI()


class Turn(pyd2.BaseModel):
    role: str = pyd2.Field(description="role")
    content: str = pyd2.Field(description="content")

class Messages(pyd2.BaseModel):
    messages: List[Turn] = pyd2.Field(description="message", default=[])
# {"messages": [{"role": role, "content": conent}, ...]}

class Goal(pyd1.BaseModel):
    goal: str = pyd1.Field(description="goal.")
    goal_number: int = pyd1.Field(description="goal number.")
    accomplished: bool = pyd1.Field(description="true if goal is accomplished else false.")


class Goals(pyd1.BaseModel):
    goal_list: List[Goal] = pyd1.Field(description="대화에서 유저가 달성해야하는 목표 리스트")

    # {"goal_list": [{'goal': '~~', "goal_nubmer":0, "accomplished":True}, ...]}


parser = JsonOutputParser(pydantic_object=Goals)
format_instruction = parser.get_format_instructions()


def detect_goal_completion(messages, roleplay):
    global parser, format_instruction

    # Conversation
    conversation ="\n".join([ f"{msg['role']}: {msg['content']}" for msg in messages]) 
    # A:1234
    # B: 12312
    # ...

    # Goals
    goal_list = roleplay_to_goal_map[roleplay] # [cheese burger, coca cola]
    goals = "\n".join([f"- Goal Number {i}: {goal} " for i, goal in enumerate(goal_list)])

    
    prompt_template = """
# 대화
{conversation}
---
# 유저의 목표
{goals}
---
위 대화를 보고 유저가 goal들을 달성했는지 확인해서 아래 포맷으로 응답해.
{format_instruction}
"""

    chat_prompt_template = ChatPromptTemplate.from_messages([("user", prompt_template)])
    goal_check_chain = chat_prompt_template | model | parser

    outputs = goal_check_chain.invoke({"conversation": conversation,
                                       "goals": goals,
                                       "format_instruction": format_instruction})
    return outputs



type_to_msg_class_map = {
        "system":  SystemMessage,
        "user":  HumanMessage,
        "assistant":  AIMessage,
        }

def chat(messages):
    messages_lc = []
    for msg in messages:
        msg_class = type_to_msg_class_map[msg["role"]]
        msg_lc = msg_class(content=msg["content"])

        messages_lc.append(msg_lc)
        
    resp = model.invoke(messages_lc)
    return {"role": "assistant", "content": resp.content}


roleplay_to_system_prompt_map = {
        "hamburger": """\
- 너는 햄버거 가게의 직원이다.
- 아래의 단계로 질문을 한다.
1. 주문 할 메뉴 묻기
2. 더 주문 할 것이 없는지 묻기
3. 여기서 먹을지 가져가서 먹을지 질문한다.
4. 카드로 계산할지 현금으로 계산할지 질문한다.
4. 주문이 완료되면 인사를 하고 [END] 라고 이야기한다.
- 너는 영어로 응답한다.\
""",
        "immigration": """\
- 너는 출입국 사무소의 직원이다.
- 아래의 단계로 질문을 한다.
1. 이름 묻기
2. 여행의 목적 묻기
3. 몇일간 체류하는지 묻기
4. 어떤 호텔에서 체류하는지 묻기
5. 모든 질문에 답이 끝났으면 [END] 라고 이야기한다.
- 너는 영어로 응답한다.\
"""
        }

roleplay_to_goal_map = {
        "hamburger": ["Ordering a cheeseburger",
                      "Ordering a Coca-Cola"],
        "immigration": ["Saying I've come to watch an NBA game",
                        "Saying I'll be staying for five days"]
        }


@app.post("/chat", response_model=Turn)
def post_chat(messages: Messages):
    messages_dict = messages.model_dump() # change format to dict()
    print(messages_dict)
    resp = chat(messages=messages_dict['messages'])

    return resp



@app.post("/chat/{roleplay}", response_model=Turn)
def post_chat_role_play(messages: Messages, roleplay: str):
    messages_dict = messages.model_dump()

    # 해당 롤플레이를 위한 system prompt 가져오기
    system_prompt = roleplay_to_system_prompt_map[roleplay]
    msgs = messages_dict['messages']

    msgs = [{"role": "system", "content": system_prompt}] + msgs 
    resp = chat(messages=msgs)

    return resp


@app.get("/{roleplay}/goals")
def get_roleplay_goals(roleplay: str):
    return roleplay_to_goal_map[roleplay]


@app.post("/{roleplay}/check_goals")
def post_roleplay_check_goal(messages: Messages, roleplay: str):
    messages_dict = messages.model_dump()
    messages = messages_dict['messages']
    goal_comp = detect_goal_completion(messages, roleplay)
    return goal_comp


@app.post("/transcribe")
def transcribe_audio(audio_file: UploadFile = File(...)):
    try:

        file_name = "tmp_audio_file.wav"
        with open(file_name, "wb") as f:
            f.write(audio_file.file.read())
        
        with open(file_name, "rb") as f:
            transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                )
        
        text = transcript.text
    except Exception as e:
        print(e)
        text = f"Failed in voice recognition. {e}"
        return {"status": "fail", "text": text}
    print(f"input: {text}")

    return {"status": "ok", "text": text}




