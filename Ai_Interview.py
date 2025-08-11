import os
from autogen_agentchat.agents import AssistantAgent,UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model = OpenAIChatCompletionClient(model='gemini-1.5-flash-8b', api_key=api_key)
