import os
import sys
import json
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

class EmotionScore(BaseModel):
    emotion: int = Field(description="Overall emotion state at the moment: 0-100, where 0 is very negative and 100 is elated")


# Global debug flag
DEBUG = False

class Interviewer:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.system_prompt = (
            "You are an interviewer conducting a job assessment interview on a candidate's Product management skills. "
            "Please focus on the following areas of interest: "
            "(a) market positioning of the new product, "
            "(b) competitive analysis, "
            "(c) TAM calculation, "
            "(d) MRD and PRD creation, "
            "(e) engineering, "
            "(f) pre-launch and launch, "
            "(g) maintenance and EOL cycles. "
            "Make it general enough to test the candidate's knowledge and ask them to provide specific examples."
            "Keep the interview conversational and engaging and to the point. When all areas are covered, ask the candidate if they have any questions."
            "Do not output bullet points, markdown titles, or other formatting. Just output the text in a clear and easy to read format."
            "Use your thoughts on this candidate as a reference. They are marked as 'thoughts' in assistant messages."
            "Also consider your emotional state changes and how they affect your assessment of the candidate."
            "They are marked as 'emotions' in assistant messages. You are allowed to be emotional and let it show."
        )
        self.conversation_history = []
        self.messages = []

    def call_anthropic_api(self, messages, system_prompt=None):
        # Debug: Print accumulated context before API call
        if DEBUG:
            print("\n----- DEBUG: LATEST CONTEXT BEING SENT TO API -----")
            print("Messages:")
            for msg in messages[-10:]:
                print(f"  {msg['role']}: {msg['content']}")
            print("---------------------------------------------\n")
        
        # Use provided system prompt or default to self.system_prompt
        prompt_to_use = system_prompt if system_prompt else self.system_prompt
        
        try:
            client = Anthropic(api_key=self.api_key)
            message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                system=prompt_to_use,
                messages=messages
            )
            
            # Check if content exists and has elements
            if message.content and len(message.content) > 0:
                return message.content[0].text
            else:
                # Handle empty response
                print("Warning: Received empty response from API")
                return "I'm sorry, I'm having trouble formulating a response. Let's continue with the interview."
        except Exception as e:
            # Handle any API errors
            print(f"Error calling Anthropic API: {str(e)}")
            return "I apologize for the technical difficulties. Let's proceed with the interview."

    def generate_internal_emotions(self):
        """Generate interviewer's emotional state during the interview"""
        emotions_prompt = (
            "You are impersonating an emotional plane of an interviewer. "
            "Based on the conversation so far, express your current emotional state "
            "and your feelings about the candidate. Be authentic and raw with your emotions. "
            "For example: 'Im feeling really now excited about the candidate's experience', or "
            "'Im getting increasingly frustrated because the candidate is avoding my questions.' "
            "Consider your previous emotional state to gauge the change and conclude with the final state, e.g. 'I am sad now'"
            "Only print your assessment of emotional state and nothing else – no tags, no markdown, no formatting, just the statement."
        )
        
        # Call API with the conversation history and the emotions prompt
        return self.call_anthropic_api(self.messages, emotions_prompt)

    def generate_emotion_score(self, text):
        """Generate an emotion score for a given text"""
        emotion_score_schema = EmotionScore.model_json_schema()
 
        tools = [
            {
                "name": "emotion_score_result",
                "description": "build the emotion score object",
                "input_schema": emotion_score_schema
            }
        ]
        client = Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1200,
            temperature=0.2,
            system="You are calculating the integer emotion score for a given text (0-100).",
            messages=[
                {
                    "role": "user",
                    "content": f"{text}"
                }
            ],
            tools=tools,
            tool_choice={"type": "tool", "name": "emotion_score_result"}
        )
        function_call = message.content[0].input
        return EmotionScore(**function_call).emotion

    def get_response(self, user_input):
        """Function mode: Get a single response from the interviewer"""
        # Initialize conversation if this is the first interaction
        if not self.messages:
            if user_input:
                # If user provided an opening message, use it
                self.messages.append({"role": "user", "content": user_input})
                
                # Generate internal emotions first
                internal_emotions = self.generate_internal_emotions().strip()
                if "[emotions]" in internal_emotions and "[/emotions]" in internal_emotions:
                    internal_emotions = internal_emotions.split("[emotions]")[1].split("[/emotions]")[0]
                # Add internal emotions to messages for the model to see
                self.messages.append({"role": "assistant", "content": f"[emotions]{internal_emotions}[/emotions]"})
                
                emotion_score = self.generate_emotion_score(internal_emotions)
                if DEBUG:
                    print(f"Emotion score: {emotion_score}")
                # Generate internal monologue
                internal_thoughts = self.generate_internal_monologue().strip()
                if "[thoughts]" in internal_thoughts and "[/thoughts]" in internal_thoughts:
                    internal_thoughts = internal_thoughts.split("[thoughts]")[1].split("[/thoughts]")[0]
                # Add internal thoughts to messages for the model to see
                self.messages.append({"role": "assistant", "content": f"[thoughts]{internal_thoughts}[/thoughts]"})
                
                # Get response from API
                interviewer_response = self.call_anthropic_api(self.messages).strip()
                
                # Add the actual response to messages for future context
                self.messages.append({"role": "assistant", "content": interviewer_response})
                
                # Store the complete conversation history separately if needed
                self.conversation_history = self.messages.copy()
                
                return (internal_emotions, internal_thoughts, interviewer_response, emotion_score)
            else:
                # Otherwise start with an assistant message
                initial_message = self.call_anthropic_api([{"role": "user", "content": "Hello, I'm here for the interview."}])
                
                # No thoughts for the initial message since there's no context yet
                self.messages.append({"role": "user", "content": "Hello, I'm here for the interview."})
                self.messages.append({"role": "assistant", "content": initial_message})
                
                # Store the complete conversation history separately if needed
                self.conversation_history = self.messages.copy()
                
                return (None, None, initial_message, None)
        else:
            # Add user input to messages
            self.messages.append({"role": "user", "content": user_input})
        
            # Generate internal emotions first
            internal_emotions = self.generate_internal_emotions().strip()
            # Strip the answer and only return the content between [emotions]..[/emotions] or the whole string if there are no tags
            if "[emotions]" in internal_emotions and "[/emotions]" in internal_emotions:
                internal_emotions = internal_emotions.split("[emotions]")[1].split("[/emotions]")[0]

            # Add internal emotions to messages for the model to see
            self.messages.append({"role": "assistant", "content": f"[emotions]{internal_emotions}[/emotions]"})
            
            # Generate emotion score
            emotion_score = self.generate_emotion_score(internal_emotions)
            if DEBUG:
                print(f"Emotion score: {emotion_score}")
            # Generate internal monologue
            internal_thoughts = self.generate_internal_monologue().strip()
            # Strip the answer and only return the content between [thoughts]..[/thoughts] or the whole string if there are no tags
            if "[thoughts]" in internal_thoughts and "[/thoughts]" in internal_thoughts:
                internal_thoughts = internal_thoughts.split("[thoughts]")[1].split("[/thoughts]")[0]

            # Add internal thoughts to messages for the model to see
            self.messages.append({"role": "assistant", "content": f"[thoughts]{internal_thoughts}[/thoughts]"})
            
            # Get response from API
            interviewer_response = self.call_anthropic_api(self.messages)
            
            # Add the actual response to messages for future context
            self.messages.append({"role": "assistant", "content": interviewer_response})
            
            # Store the complete conversation history separately if needed
            self.conversation_history = self.messages.copy()
            
            return (internal_emotions, internal_thoughts, interviewer_response, emotion_score)

    def generate_internal_monologue(self):
        """Generate interviewer's internal thoughts about the candidate"""
        internal_monologue_prompt = (
            "You are an interviewer conducting a job assessment interview on a candidate's Product management skills. "
            "Your job is to assess where you are in this conversation given the following context: "
            "(a) what the interviewee said so far (b) how you candidly assessed this internally and (c) what you said back to him. "
            "Your assessment should be straightforward and similar to what you would say to your good colleague about this candidate, "
            "without covering anything up. It will never be heard by a candidate. For example, if you observe that the candidate "
            "is making great claims but lacks on examples to support them, you may tell your colleague: 'this guy likes to make "
            "bold statements but he is a bit thin on substance and experience' "
            "Only print your assessment and nothing else – no tags, no markdown, no formatting, just the statement."
        )
        
        # Call API with the conversation history and the internal monologue prompt
        return self.call_anthropic_api(self.messages, internal_monologue_prompt)

    def conduct_interview(self, opening_message=None, function_mode=False):
        """
        Conduct the interview either in CLI mode or function mode
        
        Args:
            opening_message: Optional initial message from the candidate
            function_mode: If True, just process the opening message and return
                          If False, run in interactive CLI mode
        """
        # Function mode - just process one message and return
        if function_mode:
            emotions, thoughts, response, emotion_score = self.get_response(opening_message)
            return (emotions, thoughts, response, emotion_score)
        
        # CLI mode - interactive session
        user_input = ""
        
        # If no opening message is provided, use a default one
        if not opening_message:
            opening_message = "Hello, I'm here for the product management interview"
            
        # Process the opening message
        print("Candidate:", opening_message)
        print("\n")
        emotions, thoughts, interviewer_response, emotion_score = self.get_response(opening_message)
        print(f"Interviewer emotions: {emotions}\n")
        print(f"Emotion score: {emotion_score}\n")
        print(f"Interviewer thoughts: {thoughts}\n")
        print(f"Interviewer response: {interviewer_response}\n")
        
        # Continue with interactive loop
        while user_input.lower() != "exit":
            user_input = input("Candidate: ")
            if user_input.lower() == "exit":
                break
                
            emotions, thoughts, interviewer_response, emotion_score = self.get_response(user_input)
            print(f"Interviewer emotions: {emotions}\n")
            print(f"Emotion score: {emotion_score}\n")
            print(f"Interviewer thoughts: {thoughts}\n")
            print(f"Interviewer response: {interviewer_response}\n")

    def main(self):
        print("Welcome to the Product Management Interview! Type responses, or print 'exit' to end it.")
        
        # Regular CLI mode
        opening_message = None
        if len(sys.argv) > 1:
            opening_message = " ".join(sys.argv[1:])
        
        print("Opening message:", opening_message)
        self.conduct_interview(opening_message)

if __name__ == "__main__":
    # Check for debug flag in environment
    if os.getenv("DEBUG", "").lower() in ("true", "1", "yes"):
        DEBUG = True
        print("Debug mode enabled")
    
    interviewer = Interviewer()
    interviewer.main()

