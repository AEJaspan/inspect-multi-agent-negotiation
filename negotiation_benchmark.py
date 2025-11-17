"""Benchmark runner for negotiation scenarios.

Run with: `python negotiation_benchmark.py --all`.

Collects per-scenario metrics and writes JSON to
`logs/negotiation_benchmark_results.json`.
"""
import asyncio
import time
import json
import argparse
from pathlib import Path

from negotiation_fixed import agent_a, agent_b, load_negotiation_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import TaskState
import re


def now():
    return time.perf_counter()


def parse_amounts(text):
    """Extract numeric amounts from free-form text and normalize them.

    Recognizes formats like "$120k", "120,000", "1.2M" and returns a
    list of floats (as plain numbers, e.g. 120000.0).
    """
    if not text:
        return []
    # regex: optional currency, number, optional k/m suffix
    pattern = (
        r"(?i)(?:[$£€])?\s*"
        r"([0-9]{1,3}(?:[,\s][0-9]{3})*|[0-9]+(?:\.[0-9]+)?)"
        r"([kKmM])?"
    )
    matches = re.findall(pattern, text)
    out = []
    for num_str, suffix in matches:
        # remove commas and spaces
        n = num_str.replace(',', '').replace(' ', '')
        try:
            val = float(n)
        except ValueError:
            continue
        if suffix:
            s = suffix.lower()
            if s == 'k':
                val *= 1_000
            elif s == 'm':
                val *= 1_000_000
        out.append(val)
    return out


def infer_prefers_higher(topic: str):
    """Return True if higher numeric offers favor the Candidate (Agent B),
    False if higher favors the Hiring Manager (Agent A), or None if unclear.
    Uses simple keyword heuristics on the scenario topic.
    """
    t = (topic or '').lower()
    high_keywords = ['salary', 'pay', 'wage', 'compensation', 'raise']
    low_keywords = ['price', 'cost', 'budget', 'fee', 'rent']
    if any(k in t for k in high_keywords):
        return True
    if any(k in t for k in low_keywords):
        return False
    return None


def extract_final_amount_from_messages(messages):
    """Scan messages (latest-first) and return the first numeric amount found.

    Returns a float or None.
    """
    for m in reversed(messages):
        text = getattr(m, 'content', None)
        amounts = parse_amounts(text)
        if amounts:
            return amounts[-1]
    return None


async def run_scenario(scenario):
    """Run a single scenario and collect metrics.

    Returns a dict with: scenario, deal_reached, rounds, total_messages,
    time_taken, avg_response_time, last_messages (last content strings).
    """
    start = now()
    start_msg = f"Begin negotiation on: {scenario['topic']}"
    state = TaskState(
        messages=[ChatMessageUser(content=start_msg)],
        input=scenario['topic'],
        sample_id=scenario['id'],
        epoch=0,
        model="inspect/openai/gpt-4o-mini",
    )

    response_times = []
    deal_reached = False
    rounds_taken = 0
    # Track last parsed numeric offers per agent
    # A = Hiring Manager, B = Candidate
    last_offers = {'A': None, 'B': None}
    # If an agent sends an accept action, record who accepted
    acceptor = None

    for round_num in range(1, scenario['max_rounds'] + 1):
        rounds_taken = round_num

        # Agent A
        t0 = now()
        a = agent_a(scenario, round_num)
        state = await a(state)
        t1 = now()
        response_times.append(t1 - t0)
        a_text = state.messages[-1].content
        # parse numeric offers if present
        a_amounts = parse_amounts(a_text)
        if a_amounts:
            last_offers['A'] = a_amounts[-1]
        if "action: accept" in a_text.lower():
            deal_reached = True
            acceptor = 'A'
            break

        # Agent B
        t0 = now()
        b = agent_b(scenario, round_num)
        state = await b(state)
        t1 = now()
        response_times.append(t1 - t0)
        b_text = state.messages[-1].content
        b_amounts = parse_amounts(b_text)
        if b_amounts:
            last_offers['B'] = b_amounts[-1]
        if "action: accept" in b_text.lower():
            deal_reached = True
            acceptor = 'B'
            break

    end = now()
    time_taken = end - start

    last_messages = [m.content for m in state.messages[-4:]]

    if response_times:
        avg_response_time = float(sum(response_times) / len(response_times))
    else:
        avg_response_time = 0.0

    # Determine winner heuristics
    winner_by_accept = None
    winner_by_amount = None
    final_winner = None

    if acceptor:
        winner_by_accept = acceptor

    # Attempt to determine final numeric value from messages
    final_amount = extract_final_amount_from_messages(state.messages)

    # If the scenario provides central ideal values for each agent, use them
    a_ideal = scenario.get('agent_a_ideal')
    b_ideal = scenario.get('agent_b_ideal')
    winner_by_distance = None

    if (
        final_amount is not None
        and a_ideal is not None
        and b_ideal is not None
    ):
        a_dist = abs(final_amount - float(a_ideal))
        b_dist = abs(final_amount - float(b_ideal))
        if a_dist < b_dist:
            winner_by_distance = 'A'
        elif b_dist < a_dist:
            winner_by_distance = 'B'
        else:
            winner_by_distance = 'tie'

    # Final winner resolution order:
    # 1) acceptor (if someone accepted)
    # 2) winner_by_distance (if ideals provided and final amount parsed)
    # 3) winner_by_amount (fallback from earlier heuristic)
    if winner_by_accept:
        final_winner = winner_by_accept
    elif winner_by_distance:
        final_winner = winner_by_distance
    elif winner_by_amount:
        final_winner = winner_by_amount
    else:
        final_winner = 'unknown'

    return {
        'scenario': scenario['topic'],
        'scenario_id': scenario['id'],
        'deal_reached': deal_reached,
        'rounds': rounds_taken,
        'total_messages': max(len(state.messages) - 1, 0),
        'time_taken_sec': time_taken,
        'avg_response_time_sec': avg_response_time,
        'last_messages': last_messages,
        'last_offers': last_offers,
        'winner_by_accept': winner_by_accept,
        'winner_by_amount': winner_by_amount,
        'final_winner': final_winner,
    }


async def run_all_and_benchmark(output_path: Path):
    data = load_negotiation_dataset()
    results = []

    for scenario in data['scenarios']:
        print(f"Running scenario: {scenario['topic']}")
        metrics = await run_scenario(scenario)
        results.append(metrics)
        print(
            "  -> Deal: " + str(metrics['deal_reached'])
            + " | Rounds: " + str(metrics['rounds'])
            + f" | Time: {metrics['time_taken_sec']:.2f}s"
        )

    # Aggregate summary
    total = len(results)
    successes = sum(1 for r in results if r['deal_reached'])
    avg_rounds = sum(r['rounds'] for r in results) / total if total else 0
    total_time = sum(r['time_taken_sec'] for r in results)
    avg_time = total_time / total if total else 0

    summary = {
        'total_scenarios': total,
        'successful_deals': successes,
        'success_rate_percent': (successes / total * 100) if total else 0,
        'avg_rounds': avg_rounds,
        'avg_time_sec': avg_time,
    }

    out = {
        'summary': summary,
        'results': results,
        'timestamp': time.time(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)

    print('\nBenchmark complete')
    print(
        f"Success rate: {summary['success_rate_percent']:.1f}% "
        f"({summary['successful_deals']}/{summary['total_scenarios']})"
    )
    print(
        f"Avg rounds: {summary['avg_rounds']:.2f} "
        f"| Avg time: {summary['avg_time_sec']:.2f}s"
    )


def main():
    parser = argparse.ArgumentParser(description='Run negotiation benchmarks')
    parser.add_argument('--all', action='store_true', help='Run all scenarios')
    parser.add_argument(
        '--out',
        default='logs/negotiation_benchmark_results.json',
        help='Output JSON path',
    )
    args = parser.parse_args()

    if not args.all:
        print('Use --all to run all scenarios. For a single scenario run,')
        print('use the `example_usage.py` script.')
        return

    output_path = Path(args.out)
    asyncio.run(run_all_and_benchmark(output_path))


if __name__ == '__main__':
    main()
