import os
import time
import json
import pandas as pd
import argparse
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

def generate_diverse_conversation_histories_prompt(scenario, conversation_needed, num_variations=10):
    return f"""Based on the following scenario and conversation requirements, generate {num_variations} DIVERSE conversation history variations:

SCENARIO:
{scenario}

CONVERSATION NEEDED:
{conversation_needed}

Generate {num_variations} different conversation histories that represent DIVERSE points in the conversation with DIFFERENT emotional states and conversation points. Each variation should be at a different stage with different emotional dynamics.

For each variation, provide:
1. A variation_description: A short phrase describing this variation (e.g., "Initial approach - no prior contact", "Multiple failed attempts - growing frustration", "Extensive history - tried various approaches")
2. A conversation objective
3. A conversation history that could be:
   - No prior exchanges (first approach)
   - A few recent exchanges (2-3)
   - Multiple past attempts with what worked/didn't work
   - Extensive history of different approaches tried
4. The current emotional state of the other party (make these VERY DIVERSE across variations)
5. The current point in the conversation where the user needs to respond (what the other person just said or did)

Format your response as a JSON array with {num_variations} objects, each containing:
- variation_id: A number from 1 to {num_variations}
- variation_description: A short descriptive phrase for this variation
- conversation_objective: The specific goal to achieve through this conversation
- conversation_history: The history of interactions (ranging from none to extensive)
- current_emotional_state: A description of the current emotional state of the other party
- conversation_point: The current point in the conversation where the user needs to respond

IMPORTANT: 
- Include variations with NO prior exchanges (first time addressing the issue)
- Include variations with EXTENSIVE history (multiple past attempts, what worked/didn't)
- Include variations with DIFFERENT amounts of prior interaction
- Make each variation TRULY DIFFERENT in terms of conversation progress
- Include variations where previous approaches failed
- Make the conversation_point specific about what the other person just said/did
- Your entire response must be valid JSON that can be parsed

Example variations:
1. No prior exchanges: "No previous discussions about this issue"
2. Brief history: "Two previous attempts to discuss, both met with deflection"
3. Multiple attempts: "Five previous conversations, tried direct approach, then sympathetic, then involving HR"
4. Extensive history: "Month-long pattern of discussions, tried various strategies including..."
"""

def generate_optimal_response_prompt(scenario, conversation_data, persona):
    return f"""Given the following scenario, conversation history, and emotional intelligence profile, generate the optimal next response to achieve the objective:

PERSONA:
{persona}

SCENARIO:
{scenario}

CONVERSATION OBJECTIVE:
{conversation_data["conversation_objective"]}

CONVERSATION HISTORY:
{conversation_data["conversation_history"]}

CURRENT EMOTIONAL STATE OF OTHER PARTY:
{conversation_data["current_emotional_state"]}

CURRENT CONVERSATION POINT:
{conversation_data["conversation_point"]}

Generate a JSON response with:
- optimal_response: The best next thing to say to achieve the objective while demonstrating emotional intelligence
- reasoning: Why this response is effective given the scenario, history, and emotional state

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
            start_idx = response_text.find('[')
            if start_idx == -1:
                start_idx = response_text.find('{')
            
            end_idx = response_text.rfind(']')
            if end_idx == -1:
                end_idx = response_text.rfind('}') + 1
            else:
                end_idx += 1
            
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
            max_tokens=4000,  # Increased for multiple variations
            temperature=0.8,  # Slightly increased for diversity
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

def generate_diverse_conversation_histories(scenario, conversation_needed, num_variations=10):
    """Generate multiple diverse conversation histories for a scenario."""
    prompt = generate_diverse_conversation_histories_prompt(scenario, conversation_needed, num_variations)
    
    system_message = "You are an expert in emotional intelligence and interpersonal dynamics. Your task is to generate diverse and realistic conversation histories and emotional states for challenging scenarios. Each variation should be truly different in terms of emotional dynamics and conversation progress. IMPORTANT: Your response must be valid JSON that can be parsed directly."
    
    response_text = api_call(prompt, system_message)
    if not response_text:
        return None
    
    data = extract_json_from_response(response_text)
    
    if isinstance(data, list) and len(data) > 0:
        required_keys = ["conversation_objective", "conversation_history", "current_emotional_state", "conversation_point", "variation_description"]
        valid_variations = [v for v in data if all(k in v for k in required_keys)]
        
        if valid_variations:
            print(f"Successfully generated {len(valid_variations)} conversation history variations")
            return valid_variations
    
    print("Failed to extract valid conversation history variations")
    return None

def generate_optimal_response(scenario, conversation_data, persona_desc):
    """Generate the optimal next response based on scenario, conversation history, and persona."""
    prompt = generate_optimal_response_prompt(scenario, conversation_data, persona_desc)
    
    system_message = "You are an expert in emotional intelligence and interpersonal dynamics. Your task is to generate optimal responses that demonstrate emotional intelligence and help achieve conversation objectives. IMPORTANT: Your response must be valid JSON that can be parsed directly."
    
    response_text = api_call(prompt, system_message)
    if not response_text:
        return None
    
    data = extract_json_from_response(response_text)
    
    if data and all(k in data for k in ["optimal_response", "reasoning"]):
        print("Successfully generated optimal response")
        return data
    else:
        print("Failed to extract valid optimal response data")
        return None

def process_scenarios_with_variations(input_file, output_file=None, persona_to_process=None, max_scenarios=None, variations_per_scenario=10, resume_from=None):
    """Process existing scenarios to generate multiple conversation variations and optimal responses."""
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
    
    # If resuming from a previous run, load existing data
    if resume_from and os.path.exists(resume_from):
        try:
            existing_df = pd.read_csv(resume_from)
            processed_data = existing_df.to_dict('records')
            print(f"Loaded {len(processed_data)} existing samples from {resume_from}")
            
            # Get the scenarios we've already processed
            processed_scenarios = set()
            for item in processed_data:
                scenario_key = f"{item['persona']}_{item['scenario'][:50]}"
                processed_scenarios.add(scenario_key)
            
            # Filter out scenarios we've already processed
            df_filtered = []
            for _, row in df.iterrows():
                scenario_key = f"{row['persona']}_{row['scenario'][:50]}"
                if scenario_key not in processed_scenarios:
                    df_filtered.append(row)
            
            if df_filtered:
                df = pd.DataFrame(df_filtered)
                print(f"Filtered to {len(df)} unprocessed scenarios")
            else:
                print("All scenarios have been processed already")
                return processed_data
                
        except Exception as e:
            print(f"Error loading existing data: {e}")
            print("Starting from scratch")
    
    # Create a temporary file to save progress
    temp_output_file = output_file or f"data/eq_training_data_diverse_temp_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    
    # Process each scenario
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing scenarios"):
        scenario = row["scenario"]
        conversation_needed = row["conversation_needed"]
        persona = row.get("persona", "Unknown")  # Use "Unknown" if persona is not in the data
        
        print(f"\nProcessing scenario {idx+1}/{len(df)} for persona {persona}")
        
        # Get the full persona description
        persona_desc = persona_map.get(persona, persona)
        
        # Generate diverse conversation histories
        conversation_variations = generate_diverse_conversation_histories(
            scenario, 
            conversation_needed,
            num_variations=variations_per_scenario
        )
        
        if conversation_variations:
            # Process each variation
            for variation in tqdm(conversation_variations, desc="Processing variations"):
                # Generate optimal response for this variation
                response_data = generate_optimal_response(scenario, variation, persona_desc)
                
                if response_data:
                    # Combine all data - REMOVED persona and eq_skills_demonstrated
                    combined_data = {
                        "scenario": scenario,
                        "conversation_needed": conversation_needed,
                        "variation_id": variation.get("variation_id", 0),
                        "variation_description": variation.get("variation_description", "Unknown variation"),
                        "conversation_objective": variation["conversation_objective"],
                        "conversation_history": variation["conversation_history"],
                        "current_emotional_state": variation["current_emotional_state"],
                        "conversation_point": variation["conversation_point"],
                        "optimal_response": response_data["optimal_response"],
                        "reasoning": response_data["reasoning"]
                    }
                    
                    processed_data.append(combined_data)
                    
                    # Save progress after each variation
                    temp_df = pd.DataFrame(processed_data)
                    temp_df.to_csv(temp_output_file, index=False)
                    print(f"Progress saved to {temp_output_file} ({len(processed_data)} samples)")
                
                # Small pause between variations to be nice to the API
                time.sleep(3)
            
            # Longer pause between scenarios
            wait_time = 10
            print(f"Waiting {wait_time} seconds before next scenario...")
            time.sleep(wait_time)
    
    # Generate final output filename if not provided
    if not output_file:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = f"data/eq_training_data_diverse_{timestamp}.csv"
    
    # Save to CSV
    if processed_data:
        final_df = pd.DataFrame(processed_data)
        final_df.to_csv(output_file, index=False)
        print(f"\nProcessed {len(processed_data)} total samples across {len(df)} scenarios and saved to {output_file}")
    else:
        print("No data was processed successfully.")
    
    return processed_data

if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate diverse EQ training data from scenarios')
    parser.add_argument('--input', type=str, default="data/eq_scenarios_20250227-161517.csv", 
                        help='Input CSV file with scenarios')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for training data (default: auto-generated filename)')
    parser.add_argument('--persona', type=str, default=None,
                        help='Filter to process only scenarios for this persona')
    parser.add_argument('--max_scenarios', type=int, default=None,
                        help='Maximum number of scenarios to process')
    parser.add_argument('--variations', type=int, default=10,
                        help='Number of variations to generate per scenario')
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode (1 scenario, 3 variations)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous run by loading this CSV file')
    
    args = parser.parse_args()
    
    # If test mode is enabled, override other settings
    if args.test:
        print("Running in TEST mode - processing 1 scenario with 3 variations")
        args.max_scenarios = 1
        args.variations = 3
        if not args.output:
            args.output = f"data/eq_training_data_TEST_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    
    # Process scenarios with variations
    process_scenarios_with_variations(
        input_file=args.input,
        output_file=args.output,
        persona_to_process=args.persona,
        max_scenarios=args.max_scenarios,
        variations_per_scenario=args.variations,
        resume_from=args.resume
    ) 