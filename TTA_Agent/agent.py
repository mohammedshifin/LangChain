import os
from typing import Optional
import requests
from datetime import datetime

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Function to get API key
def get_google_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("Google API Key not found in .env file")
    return api_key

class HistoricalDataTool(BaseTool):
    name: str = "Historical Date Lookup"
    description: str = "Find historical events for a specific date (YYYY-MM-DD format)"

    def _run(self, date: str) -> str:
        try:
            parsed_data = datetime.strptime(date, "%Y-%m-%d")
            month_day = parsed_data.strftime("%m/%d")
            response = requests.get(f"https://history.muffinlabs.com/date/{month_day}")
            data = response.json()

            events = [f"{event['year']}: {event['text']}" for event in data['data']["Events"]][:5]
            return "\n".join(events) if events else "No historical events found for this date."
        except Exception as e:
            return f"Error retrieving historical data: {str(e)}"
    
    def _arun(self, date: str):
        raise NotImplementedError("Async not supported")

class TimePeriodStoryteller(BaseTool):
    name: str = "Historical Storyteller"
    description: str = "Generate engaging stories about historical periods or events"

    def _run(self, topic: str) -> str:
        return f"A fascinating narrative about {topic} unfolds through the annals of history..."
    
    def _arun(self, topic: str):
        raise NotImplementedError("Async not supported")

class TimeTravelAgent:
    def __init__(self, api_key: Optional[str] = None):
        # Use environment variable or passed API key
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API Key is required")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0.7, 
            google_api_key=api_key
        )
        
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        
        self.tools = [
            HistoricalDataTool(),
            TimePeriodStoryteller(),
            wikipedia
        ]
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )
    
    def ask(self, question: str) -> str:
        return self.agent.run(question)

# Usage example
def main():
    # Ensure you have set GOOGLE_API_KEY environment variable
    agent = TimeTravelAgent()
    response = agent.ask("What happened on July 20, 1969?")
    print(response)

if __name__ == "__main__":
    main()