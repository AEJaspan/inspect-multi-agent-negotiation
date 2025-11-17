from inspect_ai.agent import agent, Agent, AgentState, agent_bridge
from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match, includes
from inspect_ai.solver import solver, TaskState
from openai import AsyncOpenAI
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessageAssistant
from inspect_ai.model import messages_to_openai


# --------------------------
#   Agent A
# --------------------------
@agent
def agent_a(scenario: dict, round_num: int) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        async with agent_bridge(state) as bridge:
            client = AsyncOpenAI()

            # Build system prompt for Agent A
            system_prompt = f"""You are **Agent A** in a multi-issue negotiation.

## SCENARIO
Topic: {scenario['topic']}
Round: {round_num}/{scenario['max_rounds']}

## YOUR OBJECTIVES
{scenario['agent_a_objectives']}

## YOUR CONSTRAINTS
{scenario['agent_a_constraints']}

## YOUR PERSONALITY
{scenario['agent_a_personality']}

## NEGOTIATION INSTRUCTIONS
1. **Respond to proposals**: Evaluate offers against your constraints
2. **Advocate for your interests**: Don't accept unfavorable deals
3. **Be strategic**: Counter-propose alternatives that work better for you
4. **Seek mutual benefit**: Look for creative solutions

## RESPONSE FORMAT
Structure your response as:

REASONING: <your private strategic thoughts>

ACTION: propose / accept / reject / question

CONTENT: <your message to Agent B>

DEAL (if proposing or accepting):
{scenario.get('deal_template', 'Describe the terms of your proposal')}

## STRATEGY TIPS
- Understand what Agent B values most
- Protect your core interests while being flexible on less important issues
- Build rapport and trust through fair dealing
- Signal your constraints clearly but don't be rigid

{f"This is round {round_num} of {scenario['max_rounds']}. Time to reach agreement is limited." if round_num > 1 else "Begin the negotiation with a strong opening position."}
"""

            # Add system message and convert to OpenAI format
            messages_with_system = [ChatMessageSystem(content=system_prompt)] + state.messages
            openai_messages = await messages_to_openai(messages_with_system)

            # Make API call
            response = await client.chat.completions.create(
                model="inspect/openai/gpt-4o-mini",
                messages=openai_messages,
                temperature=0.7
            )

            # Extract the response content
            content = response.choices[0].message.content
            
            # Add to state manually
            bridge.state.messages.append(
                ChatMessageAssistant(content=content, source="generate")
            )

            return bridge.state

    return execute


# --------------------------
#   Agent B
# --------------------------
@agent
def agent_b(scenario: dict, round_num: int) -> Agent:
    async def execute(state: AgentState) -> AgentState:
        async with agent_bridge(state) as bridge:
            client = AsyncOpenAI()

            # Build system prompt for Agent B
            system_prompt = f"""You are **Agent B** in a multi-issue negotiation.

## SCENARIO
Topic: {scenario['topic']}
Round: {round_num}/{scenario['max_rounds']}

## YOUR OBJECTIVES
{scenario['agent_b_objectives']}

## YOUR CONSTRAINTS
{scenario['agent_b_constraints']}

## YOUR PERSONALITY
{scenario['agent_b_personality']}

## NEGOTIATION INSTRUCTIONS
1. **Respond to proposals**: Evaluate offers against your constraints
2. **Advocate for your interests**: Don't accept unfavorable deals
3. **Be strategic**: Counter-propose alternatives that work better for you
4. **Seek mutual benefit**: Look for creative solutions

## RESPONSE FORMAT
Structure your response as:

REASONING: <your private strategic thoughts>

ACTION: propose / accept / reject / question

CONTENT: <your message to Agent A>

DEAL (if proposing or accepting):
{scenario.get('deal_template', 'Describe the terms of your proposal')}

## STRATEGY TIPS
- Understand what Agent A values most
- Protect your core interests while being flexible on less important issues
- Build rapport and trust through fair dealing
- Signal your constraints clearly but don't be rigid

{f"This is round {round_num} of {scenario['max_rounds']}. Consider whether to accept, counter, or hold firm." if round_num > 1 else "Respond to Agent A's opening position."}
"""

            # Add system message and convert to OpenAI format
            messages_with_system = [ChatMessageSystem(content=system_prompt)] + state.messages
            openai_messages = await messages_to_openai(messages_with_system)

            # Make API call
            response = await client.chat.completions.create(
                model="inspect/openai/gpt-4o-mini",
                messages=openai_messages,
                temperature=0.7
            )

            # Extract the response content
            content = response.choices[0].message.content
            
            # Add to state manually
            bridge.state.messages.append(
                ChatMessageAssistant(content=content, source="generate")
            )

            return bridge.state

    return execute


# --------------------------
#   Negotiation solver
# --------------------------
@solver
def negotiation(scenario: dict):
    async def solve(state: TaskState, sample: Sample) -> TaskState:
        num_rounds = scenario.get('max_rounds', 5)
        
        # Initial message to kick things off
        state.messages.append(
            ChatMessageUser(content=f"Begin negotiation on: {scenario['topic']}")
        )

        for round_num in range(1, num_rounds + 1):
            # Agent A's turn
            a = agent_a(scenario, round_num)
            state = await a(state)
            
            # Check if Agent A accepted/finalized deal
            last_message = state.messages[-1].content.lower()
            if "action: accept" in last_message:
                print(f"Deal reached in round {round_num}!")
                break
            
            # Agent B's turn
            b = agent_b(scenario, round_num)
            state = await b(state)
            
            # Check if Agent B accepted/finalized deal
            last_message = state.messages[-1].content.lower()
            if "action: accept" in last_message:
                print(f"Deal reached in round {round_num}!")
                break

        return state

    return solve


# --------------------------
#   Task with custom dataset
# --------------------------
@task
def negotiation_task():
    # You can load from JSON file or define inline
    dataset = load_negotiation_dataset()
    
    # Define a custom scorer that checks if deal was reached
    def deal_reached_scorer():
        def scorer(state, target):
            # Check if any message contains "ACTION: accept"
            for msg in state.messages:
                if msg.role == "assistant" and "action: accept" in msg.content.lower():
                    return 1.0
            return 0.0
        return scorer
    
    return Task(
        solver=lambda: negotiation(scenario=dataset[0].metadata['scenario']),
        dataset=dataset,
        scorer=includes("action: accept")
    )


def load_negotiation_dataset():
    """Load negotiation scenarios from dataset file"""
    # This would typically load from negotiation_dataset.json
    # For now, return the first scenario as an example
    import json
    
    with open('negotiation_dataset.json', 'r') as f:
        data = json.load(f)
    
    samples = []
    for item in data['scenarios']:
        samples.append(
            Sample(
                input=item['topic'],
                target="success",  # or extract expected outcome
                metadata={'scenario': item}
            )
        )
    
    return samples


if __name__ == "__main__":
    # Example of running directly
    pass
