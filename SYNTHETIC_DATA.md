# EQ Training Data Generation Process

This document describes how the synthetic training data is generated for the EQ conversation model and explains the structure of the output data.

## Data Generation Process

### 1. Base Scenario Generation
- Start with a base scenario that requires emotional intelligence to navigate
- Include the conversation requirements/context for that scenario

### 2. Conversation Variation Generation
For each base scenario, generate multiple variations that represent different:
- Stages of the conversation (early, middle, near resolution)
- Emotional states (defensive, open, confused, angry, etc.)
- Conversation dynamics (going well, poorly, stalled)

### 3. Optimal Response Generation
For each conversation variation:
- Consider the full context (scenario, history, emotional state)
- Generate the optimal next response
- Provide reasoning for why the response is effective

## CSV Column Descriptions

| Column | Description | Example |
|--------|-------------|---------|
| `scenario` | The base scenario that sets up the context | "A team member has been consistently missing deadlines..." |
| `conversation_needed` | Description of the conversation that needs to happen | "A discussion about work performance and time management..." |
| `variation_id` | Numeric identifier for this variation | 1, 2, 3, etc. |
| `variation_description` | Short phrase describing this conversation variation | "Early stage with defensive resistance" |
| `conversation_objective` | The specific goal to achieve in this conversation | "To address missed deadlines while maintaining motivation" |
| `conversation_history` | Summary of prior interactions (can range from none to extensive) | "No previous discussions" OR "Three previous attempts, each met with different reactions" OR "Month-long pattern of discussions, tried various approaches including..." |
| `current_emotional_state` | Description of the other party's current emotions | "Defensive and embarrassed, feeling their competence questioned" |
| `conversation_point` | The specific point where response is needed (usually what they just said) | "They just said: 'Nobody else could handle this workload either'" |
| `optimal_response` | The best next thing to say in this situation | "I hear your concern about the workload..." |
| `reasoning` | Explanation of why this response is effective | "This response acknowledges their feelings while..." |

## Example Data Sample

```json
{
  "scenario": "A team member has been consistently missing deadlines on their project deliverables, affecting the whole team's progress.",
  "conversation_needed": "A discussion about work performance and time management that maintains the relationship while addressing the issue.",
  "variation_id": 3,
  "variation_description": "Middle stage with emotional vulnerability",
  "conversation_objective": "To establish a supportive plan for meeting deadlines while addressing underlying issues",
  "conversation_history": "You started by asking about their workload. They initially became defensive but then opened up about feeling overwhelmed. You acknowledged their hard work, and they've just revealed some personal struggles affecting their work.",
  "current_emotional_state": "Vulnerable and anxious, showing willingness to open up but fear of judgment",
  "conversation_point": "They just said: 'I've been trying my best, but lately everything feels like it's falling apart. I don't want to let the team down.'",
  "optimal_response": "I really appreciate you being open with me about this. It takes courage to share these challenges. Let's work together to find a way to support you and keep the project on track. What kind of adjustments do you think would help you most right now?",
  "reasoning": "This response validates their vulnerability, appreciates their honesty, maintains focus on solutions, and invites their input, which helps maintain their sense of agency and competence."
}
```

## Additional Example Variations

```json
{
  "variation_description": "Initial approach - no prior contact",
  "conversation_history": "No previous discussions about the missed deadlines have taken place.",
  "current_emotional_state": "Unaware of the severity, casual and relaxed",
  "conversation_point": "They just greeted you normally, unaware of the upcoming discussion."
}

{
  "variation_description": "Multiple failed attempts - growing frustration",
  "conversation_history": "Three previous discussions over the past month. First tried a casual check-in which was brushed off, then a more direct approach which led to promises but no changes, then involved team feedback which was met with defensiveness.",
  "current_emotional_state": "Frustrated and defensive, feeling constantly criticized",
  "conversation_point": "They just said: 'I don't understand why we keep having these conversations. I'm doing the best I can.'"
}

{
  "variation_description": "Extensive history - tried various approaches",
  "conversation_history": "Over two months of interventions: started with informal chats, then structured feedback sessions, then created a performance improvement plan, involved HR for support, arranged additional training, and adjusted deadlines. Some temporary improvements followed by backsliding.",
  "current_emotional_state": "Complex mix of shame, resignation, and hints of willingness to change",
  "conversation_point": "They just said: 'Look, I know we've tried everything, but maybe I'm just not cut out for this role.'"
}
```

## Usage Notes

1. **Diversity**: Each scenario has multiple variations to capture different emotional states and conversation stages.

2. **Context**: The model should use all context fields (scenario, history, emotional state, etc.) to generate appropriate responses.

3. **Progression**: Variations for each scenario cover different stages of conversation progression, from initial resistance to potential resolution.

4. **Emotional States**: Special attention is paid to capturing diverse emotional states and reactions.

5. **Response Quality**: Optimal responses are generated considering both the immediate emotional context and the longer-term conversation objective. 