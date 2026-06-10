import csv
import datetime
import json
import sys
from pathlib import Path


TEST_SET2_START = '2023-11-01'
TEST_SET2_END = '2023-11-15'
OUTPUT_DIR = Path('run/test_set2_november')


# Make parameter.py initialize the simulator date filter for this standalone eval.
sys.argv = [
    sys.argv[0],
    '--experiment_start_date',
    TEST_SET2_START,
    '--experiment_end_date',
    TEST_SET2_END,
]

from agent.dispatch.fastest_agent import FastestDispatchAgent
from agent.reposition.scratch_rl_agent import (
    ScratchCostOnlyRLRepositionAgent,
    ScratchDispatchAwareNoCostRLRepositionAgent,
    ScratchDispatchAwareNoShapingRLRepositionAgent,
    ScratchDispatchAwareRLRepositionAgent,
    ScratchPerceptionOnlyRLRepositionAgent,
    ScratchRLRepositionAgent,
)
from agent.reposition.urgent_agent import UrgentRepositionAgent
from main_process import collect_metrics, run_one_day, set_random_seed
from parameter import SEED
from perception import PerceptionLayer
from simulator import Environment


METHODS = [
    ('Urgent', UrgentRepositionAgent),
    ('ScratchRL', ScratchRLRepositionAgent),
    ('ScratchDispatchAwareRL', ScratchDispatchAwareRLRepositionAgent),
    ('ScratchDispatchAwareNoCostRL', ScratchDispatchAwareNoCostRLRepositionAgent),
    ('ScratchDispatchAwareNoShapingRL', ScratchDispatchAwareNoShapingRLRepositionAgent),
    ('ScratchCostOnlyRL', ScratchCostOnlyRLRepositionAgent),
    ('ScratchPerceptionOnlyRL', ScratchPerceptionOnlyRLRepositionAgent),
]


def get_dates(start: str, end: str):
    current = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    last = datetime.datetime.strptime(end, '%Y-%m-%d').date()
    dates = []
    while current <= last:
        dates.append(current)
        current += datetime.timedelta(days=1)
    return dates


def evaluate_method(method_name, agent_cls, dates):
    set_random_seed(SEED)
    env = Environment()
    env.reset_statistic()
    dispatch_agent = FastestDispatchAgent()
    reposition_agent = agent_cls(train_mode=False)
    perception = PerceptionLayer()

    next_episode = reposition_agent.model_initialization()
    loaded_checkpoint_epoch = next_episode - 1 if next_episode > 0 else None
    for day in dates:
        run_one_day(env, day, dispatch_agent, reposition_agent, perception)

    metrics = collect_metrics(env)
    metrics.update(
        {
            'method': method_name,
            'loaded_checkpoint_epoch': loaded_checkpoint_epoch,
            'date_start': str(dates[0]),
            'date_end': str(dates[-1]),
            'num_days': len(dates),
            'delivered_quantity': env.total_delivered_quantity,
            'revenue': env.revenue,
            'cost_ratio': metrics['cost'] / env.revenue if env.revenue else 0.0,
            'fail_dispatch': env.fail_dispatch,
            'fail_order': env.fail_order,
        }
    )
    return metrics


def write_outputs(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / 'metrics.json'
    csv_path = OUTPUT_DIR / 'metrics.csv'
    md_path = OUTPUT_DIR / 'metrics.md'

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')

    fieldnames = [
        'method',
        'loaded_checkpoint_epoch',
        'num_days',
        'date_start',
        'date_end',
        'cost',
        'fuel_cost',
        'overtime',
        'discontinuity',
        'station_ot',
        'project_ot',
        'go_km',
        'return_km',
        'delivered_quantity',
        'revenue',
        'cost_ratio',
        'fail_dispatch',
        'fail_order',
    ]
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    urgent_cost = next(row['cost'] for row in rows if row['method'] == 'Urgent')
    best_cost = min(row['cost'] for row in rows)

    lines = [
        '# Test Set2 November Metrics',
        '',
        f'- Date range: `{TEST_SET2_START} ~ {TEST_SET2_END}`',
        '- Dispatch method: `Fastest`',
        '- Mode: evaluation only, no training, no checkpoint saving',
        '- RL checkpoints: existing `best_checkpoint.json` in each method directory',
        '',
        '| method | loaded checkpoint | cost RMB | vs Urgent | fuel | overtime | discontinuity | station OT | project OT | go km | return km | delivered m3 | cost/revenue |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sorted(rows, key=lambda item: item['cost']):
        vs_urgent = row['cost'] - urgent_cost
        lines.append(
            f"| `{row['method']}` | {row['loaded_checkpoint_epoch']} | "
            f"{row['cost']:,.2f} | {vs_urgent:+,.2f} | "
            f"{row['fuel_cost']:,.2f} | {row['overtime']:,.2f} | "
            f"{row['discontinuity']:,.2f} | {row['station_ot']:,.2f} | "
            f"{row['project_ot']:,.2f} | {row['go_km']:,.2f} | "
            f"{row['return_km']:,.2f} | {row['delivered_quantity']:,.2f} | "
            f"{row['cost_ratio']:.2%} |"
        )

    lines.extend(
        [
            '',
            f'Best method on test set2: `{min(rows, key=lambda item: item["cost"])["method"]}`.',
            f'Best cost: `{best_cost:,.2f} RMB`.',
            '',
            'Artifacts:',
            '',
            f'- `{json_path.as_posix()}`',
            f'- `{csv_path.as_posix()}`',
        ]
    )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return json_path, csv_path, md_path


def main():
    dates = get_dates(TEST_SET2_START, TEST_SET2_END)
    rows = []
    for method_name, agent_cls in METHODS:
        print(f'Evaluating {method_name} on {TEST_SET2_START} ~ {TEST_SET2_END}')
        rows.append(evaluate_method(method_name, agent_cls, dates))

    json_path, csv_path, md_path = write_outputs(rows)
    print(f'Wrote {json_path}')
    print(f'Wrote {csv_path}')
    print(f'Wrote {md_path}')
    print()
    for row in sorted(rows, key=lambda item: item['cost']):
        print(
            f"{row['method']}: cost={row['cost']:.2f}, "
            f"loaded_checkpoint_epoch={row['loaded_checkpoint_epoch']}"
        )


if __name__ == '__main__':
    main()
