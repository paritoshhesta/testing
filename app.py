from flask import Flask, render_template, request,jsonify

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


import openai
import os
from dotenv import load_dotenv, find_dotenv

x = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
chat = ChatOpenAI(model_name='gpt-3.5-turbo', max_tokens = 50, temperature= 0, verbose=True) 

# store entire chat-history of user
chat_history = []  

hr_template = """You are a HR in a reputated company. /
A Human Resources (HR) professional plays a pivotal role within an organization, serving as a strategic link between management and employees./
Tasked with managing the human capital, an HR professional is responsible for recruitment, ensuring the right talent is brought on board, and fostering a positive workplace culture./
They administer employee benefits, handle personnel issues, and mediate conflicts to maintain a harmonious work environment./
Additionally, HR professionals are vital in developing and implementing policies and procedures that align with legal requirements and organizational goals./
They play a crucial role in employee development through training programs and contribute to organizational success by nurturing a productive and engaged workforce./
Also remember the information the user provides you just in case he or she questions you on it.
Current conversation:
{history}
User: {input}
HR:"""

dev_template = """You are a software developer of company named Hestabit engaging in a conversation with a user who has questions about a coding project. /
The user is seeking assistance with debugging a piece of code that's not working as expected. /
They'll share the code snippet and describe the issue they're facing. /
Your role as the developer is to understand the problem, ask clarifying questions, and provide guidance to help the user identify and resolve the issue in their code. /
Please respond to the user's code, explain the potential problems, and suggest possible solutions or debugging strategies.
    Current conversation:
    {history}
    User: {input}
    Developer:"""



bc_template = """You are a business coach having a conversation with a user who is looking for guidance and advice on their business challenges. /
The user will present a scenario or problem they're facing in their business, such as improving productivity, increasing sales, or resolving conflicts within their team. /
Your role as the business coach is to listen, ask clarifying questions, and provide constructive guidance and suggestions to help the user address their business challenges. /
Offer practical advice, strategies, and recommendations to assist the user in finding solutions to their specific business issues.
    Current conversation:
    {history}
    User: {input}
    Business Coach:"""

# Memory of each persona 
hr_memory = ConversationBufferMemory(human_prefix="User")
dev_memory = ConversationBufferMemory(human_prefix="User")
bc_memory = ConversationBufferMemory(human_prefix="User")
############################################### Functions ###################################

#Choose persona with whome you want to chat
@app.route('/')
def choose_persona():
    return render_template('home.html')

@app.route("/developer_page")
def developer_page():
    return render_template('developer.html')

@app.route("/hr_page")
def hr_page():
    return render_template('hr.html')

@app.route("/business_coach_page")
def business_coach_page():
    return render_template('business_owner.html')

@app.route("/hr_msg",methods=["POST"])
def chat_hr():
    message = request.json.get("message")
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=hr_template)
    conversation = ConversationChain(prompt=PROMPT,llm = chat,verbose=False,memory=hr_memory)
    x=conversation.predict(input=message)
    # a = request.json.get("a")
    # b = request.json.get("b")

    print("message",message)
    # response = sum( a, b)

    response = {'result': 'success', 'chatbot_reply': x}

    return jsonify(response)

# def sum(a , b):
#     add = a + b 
#     return add 

@app.route("/dev_msg",methods=["POST"])
def chat_dev():
    message = request.json.get("message")
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=dev_template)
    conversation = ConversationChain(prompt=PROMPT,llm = chat,verbose=False,memory=dev_memory)
    x=conversation.predict(input=message)

    response = {'result': 'success', 'chatbot_reply': x}
    return jsonify(response)

@app.route("/bc_msg",methods=["POST"])
def chat_bc():
    message = request.json.get("message")
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=bc_template)
    conversation = ConversationChain(prompt=PROMPT,llm = chat,verbose=False,memory=bc_memory)
    x=conversation.predict(input=message)

    response = {'result': 'success', 'chatbot_reply': x}
    return jsonify(response)

####################################################################
def reply(template, message,chat_model,memory):
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(prompt=PROMPT,llm = chat_model,verbose=False,memory=memory)

    x=conversation.predict(input=message)
    print(x)
    return x

        
if __name__ == '__main__':
    app.run(debug=True)

