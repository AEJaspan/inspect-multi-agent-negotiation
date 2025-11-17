"""
Simple example of running a negotiation task with the fixed implementation.

This script demonstrates:
1. Loading a negotiation scenario from the dataset
2. Running agents through multiple rounds
3. Tracking negotiation outcomes
"""

from negotiation_fixed import (
    agent_a, agent_b, negotiation, 
    load_negotiation_dataset, negotiation_task
)
from inspect_ai import eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState
import json
import asyncio
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


async def run_single_negotiation_example():
    """Run a single negotiation scenario manually"""
    
    # Load a scenario
    with open('negotiation_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Pick the salary negotiation scenario
    scenario = data['scenarios'][0]  # salary negotiation
    
    print(f"\n{'='*60}")
    print(f"NEGOTIATION SCENARIO: {scenario['topic']}")
    print(f"{'='*60}\n")
    
    # Initialize state
    state = TaskState(
        messages=[ChatMessageUser(content=f"Begin negotiation on: {scenario['topic']}")],
        input=scenario['topic'],
        sample_id=scenario['id'],
        epoch=0,
        model="inspect/openai/gpt-4o-mini",
        # sample=Sample(input=scenario['topic'], target="success")
    )
    
    max_rounds = scenario['max_rounds']
    
    for round_num in range(1, max_rounds + 1):
        print(f"\n--- ROUND {round_num} ---\n")
        
        # Agent A's turn
        print("Agent A (Hiring Manager) thinking...")
        a = agent_a(scenario, round_num)
        state = await a(state)
        print(f"\nAgent A: {state.messages[-1].content[:200]}...\n")
        
        # Check for acceptance
        if "action: accept" in state.messages[-1].content.lower():
            print("✅ Deal reached by Agent A!")
            break
        
        # Agent B's turn
        print("Agent B (Candidate) thinking...")
        b = agent_b(scenario, round_num)
        state = await b(state)
        print(f"\nAgent B: {state.messages[-1].content[:200]}...\n")
        
        # Check for acceptance
        if "action: accept" in state.messages[-1].content.lower():
            print("✅ Deal reached by Agent B!")
            break
    
    print(f"\n{'='*60}")
    print("NEGOTIATION COMPLETE")
    print(f"{'='*60}\n")
    
    # Print summary
    print("Full conversation:")
    for i, msg in enumerate(state.messages[1:], 1):  # Skip initial message
        role = "🏢 Agent A" if msg.role == "assistant" and i % 2 == 1 else "👤 Agent B"
        print(f"\n{role}:")
        print(f"{msg.content[:300]}...")


async def run_all_scenarios():
    """Run all negotiation scenarios from the dataset"""
    
    with open('negotiation_dataset.json', 'r') as f:
        data = json.load(f)
    
    results = []
    
    for scenario in data['scenarios']:
        print(f"\n\n{'='*60}")
        print(f"Running: {scenario['topic']}")
        print(f"{'='*60}")
        
        state = TaskState(
            messages=[ChatMessageUser(content=f"Begin negotiation on: {scenario['topic']}")],
            sample=Sample(input=scenario['topic'], target="success")
        )
        
        deal_reached = False
        rounds_taken = 0
        
        for round_num in range(1, scenario['max_rounds'] + 1):
            rounds_taken = round_num
            
            # Agent A
            a = agent_a(scenario, round_num)
            state = await a(state)
            if "action: accept" in state.messages[-1].content.lower():
                deal_reached = True
                break
            
            # Agent B
            b = agent_b(scenario, round_num)
            state = await b(state)
            if "action: accept" in state.messages[-1].content.lower():
                deal_reached = True
                break
        
        results.append({
            'scenario': scenario['topic'],
            'deal_reached': deal_reached,
            'rounds': rounds_taken,
            'total_messages': len(state.messages) - 1
        })
        
        print(f"✅ Deal: {deal_reached} | Rounds: {rounds_taken}")
    
    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY OF ALL NEGOTIATIONS")
    print(f"{'='*60}\n")
    
    for result in results:
        status = "✅ SUCCESS" if result['deal_reached'] else "❌ NO DEAL"
        print(f"{status} - {result['scenario']}")
        print(f"   Rounds: {result['rounds']} | Messages: {result['total_messages']}")
    
    success_rate = sum(1 for r in results if r['deal_reached']) / len(results) * 100
    print(f"\n Overall Success Rate: {success_rate:.1f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("Running all scenarios...")
        asyncio.run(run_all_scenarios())
    else:
        print("Running single example...")
        print("(Use --all to run all scenarios)")
        asyncio.run(run_single_negotiation_example())
