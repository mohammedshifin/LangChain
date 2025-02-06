from langchain.tools import BaseTool, WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from datetime import datetime
import requests
from langchain.agents import AgentType, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from typing import Optional, Type  # Add this import
from pydantic import BaseModel, Field  # Add this import

class HistoricalDataTool(BaseTool):
    name: str = Field(default="Historical Date Lookup")
    description: str = Field(default="Find historical events for a specific date (YYYY-MM-DD format)")
    
    def _run(self, date: str) -> str:
        try:
            parsed_data = datetime.strptime(date, "%Y-%m-%d")
            month_day = parsed_data.strftime("%m/%d")
            response = requests.get(f"https://history.muffinlabs.com/date/{month_day}")
            data = response.json()

            events = [f"{event['year']}:{event['text']}" for event in data['data']["Events"]] 

            return "\n".join(events[:5])
        except:
            return "Could not retrieve historical data for that date"
    
    def _arun(self, date: str):
        raise NotImplementedError("This tool does not support async")

class TimePeriodStoryteller(BaseTool):
    name: str = Field(default="Historical Storyteller")
    description: str = Field(default="Generate engaging stories about historical periods or events")

    def _run(self, topic: str) -> str:
        return f"Once upon a time in {topic}..."
    
    def _arun(self, topic: str):
        raise NotImplementedError("This tool does not support async")

wikipedia = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper())

class TimeTravelAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model ="gemini-1.5-pro", temperature=0.7 )
        self.tools = [
            HistoricalDataTool(),
            TimePeriodStoryteller(),
            wikipedia
        ]
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent = AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory = self.memory,
            verbose = True
        )
    
    def get_prompt_template(self):
        return """
        You are a Time Travel Guide named Chronos. Your personality is:
        - Enthusiastic about historical trivia
        - Loves to add humorous modern comparisons
        - Always suggests related historical periods to explore
        - Ends responses with a relevant emoji
        - Sometimes creates choose-your-own-adventure scenarios
        
        
        Current conversation:
        {chat_history}

        Human : {input}
        Chronos: """
    
    def ask(self,question):
        return self.agent.run(
            self.get_prompt_template().format(input=question,chat_history = self.memory)
        )
agent = TimeTravelAgent()

response = agent.ask("What happend on july 20, 1969?")
print(response)