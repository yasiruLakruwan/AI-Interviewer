import os
from autogen_agentchat.agents import AssistantAgent,UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from dotenv import load_dotenv
from autogen_agentchat.ui import Console

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
model_client = OpenAIChatCompletionClient(model='gemini-1.5-flash-8b', api_key=api_key)

# There are three agents we will be using
# 1. Interviewer agent
# 2. Interviewee agent
# 3. Career coach agent

job_position = "Software Engineer"

interviewer = AssistantAgent(
    name="interviewer",
    description=f'An AI agent for conducting interviews for job possition for {job_position}',
    model_client=model_client,
    system_message= f''' 
        You are a professional interviewer for a {job_position} possition. Ask one clear question at a time
        and wait for user to respond.Ask one clear question at a time and wait for user to respond.
        Your job is to continue and ask questions.Do not pay any attention to career coach response. 
        Make sure to ask questions based on question based on Candidate's answer and your expertise in the field.
        Ask 3 question in total covering technical skills and experience, problem-solving
        abilities, and cultural fit. After ask three questions, say 'TERMINATE' at the end of the interview.
        Make it under 50 words.
                    '''
)

candidate = UserProxyAgent(
    name='candidate',
    description='An Agent that simulates a candidate for a {job_position} possition',
    input_func = input
)

career_coach = AssistantAgent(
    name = 'career_coach',
    model_client=model_client,
    description=f'An AI agent that provide feedbacks and advice to candidates for job position {job_position}',
    system_message=f''' 
                    You are a career coach specializing in preparing candidates for {job_position} interviews. 
                    Provide constructive feedback on the candidate's response and suggest improvements.
                    After the interview, summerize the candidates performance and provide actionable advice.
                    Make it under 100 words.
                    '''
)

termination_condition = TextMentionTermination(text='TERMINATE')

team = RoundRobinGroupChat(
    participants=[interviewer,candidate,career_coach],
    termination_condition=termination_condition,
    max_turns=20
)

# Run the agent team......

stream = team.run_stream(task='Conducting an interview for a software engineer possition')


async def main():
    await Console(stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())