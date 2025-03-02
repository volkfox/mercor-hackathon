# a platform for crowdsourcing human EQ data from job candidates

HuggingFace model links:
https://huggingface.co/sonyashijin/phi_4_interviewee_EQ_reasoning_LoRA
https://huggingface.co/sonyashijin/phi_4_bad_EQ_interviewee_16bit_merged

# Frontend live demo with HeyGen:

<img width="713" alt="image" src="https://github.com/user-attachments/assets/6b413907-8c30-403f-9e1f-0924d44d8c61" />
https://github.com/aadinash/mercor-frontend
Problem:
Lack of quality data to improve LLMs on soft skills (EIQ): Most AI models lack real-world, high-quality data for training in emotional intelligence and nuanced human communication, leading to poor soft-skill reasoning.
Lack of opportunities to improve human soft skills for job interviews: Job candidates struggle to get structured, personalized feedback on soft skills, making it difficult to improve for high-stakes remote interviews.

Our solution: An EQ training platform that helps candidates practice and improve their soft skills for remote job interviews. It provides real-time feedback on emotional intelligence and communication—while also collecting high-quality data to enhance LLM emotional intelligence. To demonstrate the value of the collected data, we fine-tuned a model on high-EQ responses from the platform, showing measurable improvements in soft skill reasoning with our EQ scorer.

we generated synthetic data to mimic data collected from our platform and finetuned models via GRPO, with our platform EQ scorer as the reward function, on filtered platform data to show data viability for improving LLM EQ - **our platform offers a way to crowd-source valuable human EQ data for LLM training**


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
