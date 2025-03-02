# EQ Scenario Generator

HuggingFace model links:
https://huggingface.co/sonyashijin/phi_4_interviewee_EQ_reasoning_LoRA
https://huggingface.co/sonyashijin/phi_4_bad_EQ_interviewee_16bit_merged

# Frontend live demo with HeyGen:

<img width="713" alt="image" src="https://github.com/user-attachments/assets/6b413907-8c30-403f-9e1f-0924d44d8c61" />



## A synthetic data generation pipeline for creating emotional intelligence scenarios and multi-turn conversations.

The "interviewer" part is hardcoded for a job interview for Product Management Position and consists of two files:
- `emotional_interviewer.py`: The main file that defines the Interviewer class and the conversation flow.
The interview is stateful (accumulating conversation history) and at every turn it generates:
- emotional interviewer's monologue
- score for emotional state (integer range from 0 to 100)
- internal interviewer's monologue
- external interviewer's monologue

- `test_interviewer.py`: A sample test script that demonstrates how to use the interviewer.

we generated synthetic data to mimic data collected from our platform and finetuned on filtered platform data to show data viability for improving LLM EQ - **our platform offers a way to crowd-source valuable human EQ data for LLM training**

## Setup

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file with your API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

Run the main script to generate EQ scenarios and conversations:
