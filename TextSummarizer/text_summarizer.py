import os
import textwrap
from typing import List, Optional

import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

class TextSummarizer:
    def __init__(self, google_api_key: Optional[str] = None):
        self.google_api_key = (
            google_api_key or
            os.getenv('GOOGLE_API_KEY')
        )

        if not self.google_api_key:
            raise ValueError("Google API key required in .env or as argument")
        genai.configure(api_key = self.google_api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def split_text(self,text:str,max_chunks_size:int = 10000)->List[str]:
        chunks = []
        while len(text)>max_chunks_size:
            split_index  =text.rfind(' ',0,max_chunks_size)
            if split_index==-1:
                split_index = max_chunks_size
            
            chunks.append(text[:split_index])
            text = text[split_index:].strip()
        chunks.append(text)
        return chunks

    def summarize(self,
                  text:str,
                  max_length: int=250,
                  tone:str = 'objective')->str:
        prompt = f"""Summarize this text. {tone} tone, {max_length} words max:
        {text}"""    

        chunks = self.split_text(text)
        summaries = []

        for chunk in chunks:
            response = self.model.generate_content(prompt)
            summaries.append(response.text)

        final_text = " ".join(summaries)   
        final_summary = self.model.generate_content(
        f"Consolidate summaries, {tone} tone, {max_length} words: {final_text}"    
        ) 

        words = final_summary.text.split()
        return ' '.join(words[:max_length])    
    def summarize_youtube_transcript(self, 
                                     video_id: str, 
                                     language: str = 'en') -> str:
        """Summarize YouTube video transcript."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
            full_text = ' '.join([entry['text'] for entry in transcript])
            return self.summarize(full_text, max_length=300)
        
        except Exception as e:
            return f"Transcript error: {str(e)}"                  