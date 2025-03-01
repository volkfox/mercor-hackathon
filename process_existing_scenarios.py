import os
import time
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from anthropic import Anthropic, APIError, APIStatusError, RateLimitError

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Define personas with varying levels of EQ
personas = [
    "Alexis: Limited Emotional Awareness - Struggles to recognize emotions in themselves and others, misses social cues, prefers structured environments and logical problems.",
    "Morgan: Emerging Emotional Recognition - Can identify basic emotions in clear contexts but struggles with nuance, applies emotional rules mechanically.",
    "Jordan: Functional Emotional Intelligence - Has adequate EQ for everyday interactions, can recognize common emotional expressions and respond appropriately.",
    "Taylor: Developed Emotional Intelligence - Possesses strong emotional awareness, can identify subtle emotional shifts, adjusts communication style based on others' states.",
    "Riley: Advanced Emotional Intelligence - Demonstrates exceptional EQ, quickly perceives complex emotional states even when concealed, navigates conflicts with skill.",
    "Quinn: Exceptional Emotional Intelligence - Possesses extraordinary EQ that seems intuitive, can read rooms instantly, understands complex emotional patterns."
]

# Map persona names to their full descriptions
persona_map = {p.split(':')[0]: p for p in personas}

def generate_conversation_history_prompt(scenario, conversation_needed):
    return f"""Based on the following scenario and conversation requirements, generate a conversation history summary and current emotional state:

SCENARIO:
{scenario}

CONVERSATION NEEDED:
{conversation_needed}

Generate a JSON response with:
1. A conversation objective: The specific goal that needs to be achieved through this conversation
2. A summary of what has happened so far in the conversation (3-4 exchanges)
3. The current emotional state of the other party
4. The current point in the conversation where the user needs to respond

Format your response as JSON with these fields:
- conversation_objective: The specific goal to achieve through this conversation
- conversation_history: A summary of what has happened in the conversation so far (3-4 exchanges)
- current_emotional_state: A description of the current emotional state of the other party
- conversation_point: The current point in the conversation where the user needs to respond

IMPORTANT: Your entire response must be valid JSON that can be parsed. Do not include any text before or after the JSON.
"""

def generate_optimal_response_prompt(scenario, conversation_objective, conversation_history, emotional_state, conversation_point, persona):
    return f"""Given the following scenario, conversation history, and emotional intelligence profile, generate the optimal next response to achieve the objective:

PERSONA:
{persona}

SCENARIO:
{scenario}

CONVERSATION OBJECTIVE:
{conversation_objective}

CONVERSATION HISTORY:
{conversation_history}

CURRENT EMOTIONAL STATE OF OTHER PARTY:
{emotional_state}

CURRENT CONVERSATION POINT:
{conversation_point}

Generate a JSON response with:
- optimal_response: The best next thing to say to achieve the objective while demonstrating emotional intelligence
- reasoning: Why this response is effective given the scenario, history, and emotional state
- eq_skills_demonstrated: The specific emotional intelligence skills being demonstrated in this response

IMPORTANT: Your entire response must be valid JSON that can be parsed. Do not include any text before or after the JSON.
"""

def extract_json_from_response(response_text):
    """Extract JSON from the response text, handling potential formatting issues."""
    # Print the first 200 characters of the response for debugging
    print(f"\n--- Response Preview (first 200 chars) ---")
    print(response_text[:200] + "..." if len(response_text) > 200 else response_text)
    print("--- End Preview ---\n")
    
    try:
        # First try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If that fails, try to find JSON in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                print(f"Extracted JSON substring: {json_str[:100]}...")
                return json.loads(json_str)
            else:
                print(f"No JSON found in response")
                return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from response: {e}")
        return None

def api_call(prompt, system_message, attempt=1, max_attempts=3):
    """Make an API call with retry logic."""
    print(f"\n--- Prompt Preview (first 200 chars) ---")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print("--- End Prompt ---\n")
    
    print(f"Making API call (attempt {attempt}/{max_attempts})")
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.7,
            system=system_message,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except RateLimitError as e:
        print(f"Rate limit error: {e}")
        if attempt < max_attempts:
            wait_time = min(2 ** attempt * 5, 60)  # Exponential backoff
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            return api_call(prompt, system_message, attempt+1, max_attempts)
        return None
        
    except APIStatusError as e:
        if e.status_code == 529:  # Overloaded
            print(f"API overloaded. Waiting before retry...")
            if attempt < max_attempts:
                wait_time = min(2 ** attempt * 10, 120)  # Longer exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                return api_call(prompt, system_message, attempt+1, max_attempts)
        else:
            print(f"API error: {e}")
        return None
        
    except Exception as e:
        print(f"Error making API call: {e}")
        return None

def generate_conversation_history(scenario, conversation_needed):
    """Generate conversation history and current emotional state based on scenario."""
    prompt = generate_conversation_history_prompt(scenario, conversation_needed)
    
    system_message = "You are an expert in emotional intelligence and interpersonal dynamics. Your task is to generate realistic conversation histories and emotional states for challenging scenarios. IMPORTANT: Your response must be valid JSON that can be parsed directly."
    
    response_text = api_call(prompt, system_message)
    if not response_text:
        return None
    
    data = extract_json_from_response(response_text)
    
    required_keys = ["conversation_objective", "conversation_history", "current_emotional_state", "conversation_point"]
    if data and all(k in data for k in required_keys):
        print("Successfully generated conversation history")
        return data
    else:
        print("Failed to extract valid conversation history data")
        return None

def generate_optimal_response(scenario, conversation_data, persona_desc):
    """Generate the optimal next response based on scenario, conversation history, and persona."""
    prompt = generate_optimal_response_prompt(
        scenario,
        conversation_data["conversation_objective"],
        conversation_data["conversation_history"],
        conversation_data["current_emotional_state"],
        conversation_data["conversation_point"],
        persona_desc
    )
    
    system_message = "You are an expert in emotional intelligence and interpersonal dynamics. Your task is to generate optimal responses that demonstrate emotional intelligence and help achieve conversation objectives. IMPORTANT: Your response must be valid JSON that can be parsed directly."
    
    response_text = api_call(prompt, system_message)
    if not response_text:
        return None
    
    data = extract_json_from_response(response_text)
    
    if data and all(k in data for k in ["optimal_response", "reasoning", "eq_skills_demonstrated"]):
        print("Successfully generated optimal response")
        return data
    else:
        print("Failed to extract valid optimal response data")
        return None

def process_scenarios(input_file, output_file=None, persona_to_process=None, max_scenarios=None):
    """Process existing scenarios to generate conversation histories and optimal responses."""
    # Read the existing scenarios
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} scenarios from {input_file}")
    
    # If a specific persona is requested, filter for it
    if persona_to_process:
        df = df[df["persona"] == persona_to_process]
        print(f"Filtered to {len(df)} scenarios for persona {persona_to_process}")
    
    # If max_scenarios is specified, limit the number of scenarios
    if max_scenarios and max_scenarios < len(df):
        df = df.sample(max_scenarios, random_state=42)
        print(f"Sampled {len(df)} scenarios")
    
    # Create a list to store the processed data
    processed_data = []
    
    # Create a temporary file to save progress
    temp_output_file = output_file or f"data/eq_training_data_temp_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    
    # Process each scenario
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing scenarios"):
        scenario = row["scenario"]
        conversation_needed = row["conversation_needed"]
        persona = row.get("persona", "Unknown")  # Use "Unknown" if persona is not in the data
        
        print(f"\nProcessing scenario {idx+1}/{len(df)} for persona {persona}")
        
        # Get the full persona description
        persona_desc = persona_map.get(persona, persona)
        
        # Generate conversation history
        conversation_data = generate_conversation_history(scenario, conversation_needed)
        
        if conversation_data:
            # Generate optimal response
            response_data = generate_optimal_response(scenario, conversation_data, persona_desc)
            
            if response_data:
                # Combine all data
                combined_data = {
                    "persona": persona,
                    "scenario": scenario,
                    "conversation_needed": conversation_needed,
                    "conversation_objective": conversation_data["conversation_objective"],
                    "conversation_history": conversation_data["conversation_history"],
                    "current_emotional_state": conversation_data["current_emotional_state"],
                    "conversation_point": conversation_data["conversation_point"],
                    "optimal_response": response_data["optimal_response"],
                    "reasoning": response_data["reasoning"],
                    "eq_skills_demonstrated": response_data["eq_skills_demonstrated"]
                }
                
                processed_data.append(combined_data)
                
                # Save progress
                temp_df = pd.DataFrame(processed_data)
                temp_df.to_csv(temp_output_file, index=False)
                print(f"Progress saved to {temp_output_file}")
        
        # Rate limiting - be nice to the API
        wait_time = 10
        print(f"Waiting {wait_time} seconds before next scenario...")
        time.sleep(wait_time)
    
    # Generate final output filename if not provided
    if not output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"data/eq_training_data_{timestamp}.csv"
    
    # Save to CSV
    if processed_data:
        final_df = pd.DataFrame(processed_data)
        final_df.to_csv(output_file, index=False)
        print(f"\nProcessed {len(processed_data)} scenarios and saved to {output_file}")
    else:
        print("No data was processed successfully.")
    
    return processed_data

if __name__ == "__main__":
    # File paths
    input_file = "data/eq_scenarios_20250227-161517.csv"
    output_file = f"data/eq_training_data_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    
    # Process all scenarios or specify parameters to process a subset
    # Example: process_scenarios(input_file, output_file, persona_to_process="Taylor", max_scenarios=2)
    process_scenarios(input_file, output_file) 