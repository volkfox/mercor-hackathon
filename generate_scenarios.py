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

def generate_scenario_prompt(persona):
    return f"""Generate a challenging scenario that would be difficult for someone with the following emotional intelligence profile to navigate:

{persona}

The scenario should:
1. Be realistic and specific
2. Involve interpersonal dynamics
3. Require emotional intelligence to handle effectively
4. Be challenging but not impossible for this persona
5. Have a clear objective that needs to be achieved

Format your response as JSON with these fields:
- scenario: A detailed description of the situation
- conversation_needed: A description of the conversation required to address the issue, including:
  * The specific objective/goal that needs to be achieved
  * The emotional challenges that make this conversation difficult
  * The key emotional intelligence skills needed to navigate it successfully

IMPORTANT: Your entire response must be valid JSON that can be parsed. Do not include any text before or after the JSON.

Example format:
{{
  "scenario": "A detailed description of the challenging situation...",
  "conversation_needed": "A description of what kind of conversation would be needed to resolve this, including the specific objective (e.g., getting team agreement, resolving a conflict, delivering difficult feedback while maintaining the relationship), the emotional challenges involved, and the EQ skills required."
}}
"""

def extract_json_from_response(response_text, persona_name, attempt_number):
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
                print(f"No JSON found in response for {persona_name}")
                return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from response for {persona_name}: {e}")
        return None

def generate_scenario(persona, attempt=1, max_attempts=3):
    """Generate a scenario and required conversation for a given persona."""
    persona_name = persona.split(':')[0]
    prompt = generate_scenario_prompt(persona)
    
    # Print prompt for debugging
    print(f"\n--- Prompt for {persona_name} ---")
    print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    print("--- End Prompt ---\n")
    
    print(f"\nGenerating scenario for {persona_name} (attempt {attempt}/{max_attempts})")
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0.7,
            system="You are an expert in emotional intelligence and interpersonal dynamics. Your task is to generate realistic, challenging scenarios that test emotional intelligence. Each scenario must have a clear objective that requires specific EQ skills to achieve. The conversation needed should outline the goal, challenges, and required skills. IMPORTANT: Your response must be valid JSON that can be parsed directly.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract JSON from the response
        data = extract_json_from_response(response.content[0].text, persona_name, attempt)
        
        if data and "scenario" in data and "conversation_needed" in data:
            print(f"Successfully generated scenario for {persona_name}")
            # Print a preview of the extracted data
            print(f"Scenario preview: {data['scenario'][:100]}...")
            print(f"Conversation needed preview: {data['conversation_needed'][:100]}...")
            return data
        else:
            print(f"Failed to extract valid data for persona: {persona_name}")
            if attempt < max_attempts:
                print(f"Retrying ({attempt+1}/{max_attempts})...")
                time.sleep(5)  # Wait longer between retries
                return generate_scenario(persona, attempt+1, max_attempts)
            return None
            
    except RateLimitError as e:
        print(f"Rate limit error: {e}")
        if attempt < max_attempts:
            wait_time = min(2 ** attempt * 5, 60)  # Exponential backoff
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            return generate_scenario(persona, attempt+1, max_attempts)
        return None
        
    except APIStatusError as e:
        if e.status_code == 529:  # Overloaded
            print(f"API overloaded. Waiting before retry...")
            if attempt < max_attempts:
                wait_time = min(2 ** attempt * 10, 120)  # Longer exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                return generate_scenario(persona, attempt+1, max_attempts)
        else:
            print(f"API error: {e}")
        return None
        
    except Exception as e:
        print(f"Error generating scenario: {e}")
        return None

def main():
    """Main function to generate scenarios for all personas and save to CSV."""
    all_scenarios = []
    
    print(f"Generating scenarios for {len(personas)} personas...")
    
    # Save any successful scenarios as we go
    temp_df_path = "data/temp_scenarios.csv"
    
    for persona in tqdm(personas, desc="Personas"):
        persona_scenarios = []
        
        # Generate 2 scenarios per persona (reduced from 3)
        for i in range(2):
            print(f"\nGenerating scenario {i+1}/2 for {persona.split(':')[0]}")
            data = generate_scenario(persona)
            
            if data:
                # Add persona information to the data
                data["persona"] = persona.split(':')[0]
                all_scenarios.append(data)
                persona_scenarios.append(data)
                
                # Save progress after each successful generation
                temp_df = pd.DataFrame(all_scenarios)
                temp_df.to_csv(temp_df_path, index=False)
                print(f"Progress saved to {temp_df_path}")
                
            # Rate limiting - be nice to the API
            wait_time = 10  # Increased wait time between requests
            print(f"Waiting {wait_time} seconds before next request...")
            time.sleep(wait_time)
        
        print(f"Completed {len(persona_scenarios)} scenarios for {persona.split(':')[0]}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_scenarios)
    
    # Save only the required columns
    output_df = df[["scenario", "conversation_needed"]]
    
    # Generate timestamp for filename
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"data/eq_scenarios_{timestamp}.csv"
    
    # Save to CSV
    output_df.to_csv(filename, index=False)
    print(f"\nGenerated {len(all_scenarios)} scenarios and saved to {filename}")

if __name__ == "__main__":
    main() 