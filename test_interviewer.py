from emotional_interviewer import Interviewer
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import statistics

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

client = Anthropic(api_key=api_key)
N_TURNS = 10
N_SIM = 10

def main():
    print("Testing the Interviewee's EIQ via interviewer's emotional score")
    print("---------------------------------------------")
    
    personas = [
        {"name": "Alex (High EQ)", "eq_level": "High", "description": "Strong leadership, empathetic, and excellent communicator."},
        {"name": "Jordan (Low EQ)", "eq_level": "Low", "description": "Struggles with collaboration, dismissive of feedback, poor communication."},
        {"name": "Taylor (Mid EQ)", "eq_level": "Mid", "description": "Good communication but lacks empathy and adaptability."},
        {"name": "Morgan (High EQ)", "eq_level": "High", "description": "Inspiring leader, strong interpersonal skills."},
        {"name": "Casey (Low EQ)", "eq_level": "Low", "description": "Avoids responsibility, struggles with emotional awareness."},
    ]

    for persona in personas:
        scores = []

        for _ in range(N_SIM):
            interviewer = Interviewer()
            conversation_history = []
            total_emotion_score = 0
            interviewee_response = None

            for _ in range(N_TURNS):
                # Start with the interviewer asking a question
                result = interviewer.conduct_interview(interviewee_response, function_mode=True)
                emotions, thoughts, interviewer_response, emotion_score = result
                if not isinstance(emotion_score, int):
                    emotion_score = 50
                # Print emotions and thoughts
                print(f"Interviewer emotions: {emotions}")
                print(f"Emotion score: {emotion_score}")
                print(f"Interviewer thoughts: {thoughts}")
                print(f"Interviewer response: {interviewer_response}")
                
                # Accumulate the emotion score
                total_emotion_score += emotion_score
                
                # Create the interviewee's prompt based on the interviewer's question
                interviewee_prompt = f"""
                You are {persona["name"]}, a product management candidate.
                Your emotional intelligence (EQ) level is {persona["eq_level"]}. {persona["description"]}
                You are a product manager with 3 years of experience working in two AI startups. You are very good technically 
                but are less exposed to business side of things, which you know theoretically but not practically.
                You are taking a job interview for a product manager position. Answer questions based on your personality traits
                and your job experience. You are allowed to state any facts which fit your personality and your job history, 
                but stay consistent with the conversation history. Only output the response relevant to your persona and nothing else.
                Conversation history so far: {conversation_history}
                Next interviewer question: {interviewer_response}
                """
                
                try:
                    interviewee_response = client.messages.create(
                        model="claude-3-7-sonnet-20250219",
                        max_tokens=300,
                        messages=[{"role": "user", "content": interviewee_prompt}]
                    )
                    interviewee_response = interviewee_response.content[0].text.strip()
                except Exception as e:
                    print(f"Error during API call: {e}")
                    interviewee_response = "Sorry, I couldn't process that."

                print(f"\nCandidate: {interviewee_response}")
                conversation_history.append(f"Interviewer: {interviewer_response}.")
                conversation_history.append(f"You answered: {interviewee_response}.")
            
            # Calculate the average emotion score for this simulation
            average_emotion_score = total_emotion_score / N_TURNS
            scores.append(average_emotion_score)

        # Calculate statistics for the persona
        avg_score = statistics.mean(scores)
        min_score = min(scores)
        max_score = max(scores)
        std_dev = statistics.stdev(scores)

        # Print final statistics for the persona
        print(f"\nStatistics for {persona['name']} in {N_TURNS} conversation turns in {N_SIM} simulations:")
        print(f"Average Emotion Score: {avg_score}")
        print(f"Minimum Emotion Score: {min_score}")
        print(f"Maximum Emotion Score: {max_score}")
        print(f"Standard Deviation: {std_dev}")

if __name__ == "__main__":
    main() 



