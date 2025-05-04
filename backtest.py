import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import product

ORDER_SIZE = 5000
STEP_SIZE = 100

def load_data(path='l1_day.csv'):
    df = pd.read_csv(path)
    df = df.dropna(subset=['ask_px_00', 'ask_sz_00'])
    df = df.sort_values('ts_event')
    df = df.drop_duplicates(subset=['ts_event', 'publisher_id'], keep='first')
    return list(df.groupby('ts_event'))

def allocator(venues, lambda_over, lambda_under, theta_queue):
    N = len(venues)
    step = STEP_SIZE
    best_cost = float('inf')
    best_split = None

    def cost(alloc):
        used = sum(alloc)
        over = max(0, used - ORDER_SIZE)
        under = max(0, ORDER_SIZE - used)
        queue_penalty = sum([alloc[i] / venues[i]['ask_size'] if venues[i]['ask_size'] > 0 else 0 for i in range(N)])
        exec_cost = sum([alloc[i] * venues[i]['ask'] for i in range(N)])
        return (
            exec_cost +
            lambda_over * over +
            lambda_under * under +
            theta_queue * queue_penalty
        )

    splits = [[]]
    for i in range(N):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(ORDER_SIZE - used + 500, venues[i]['ask_size'])
            for qty in range(0, int(max_v) + 1, step):
                new_splits.append(alloc + [qty])
        splits = new_splits

    for alloc in splits:
        c = cost(alloc)
        if c < best_cost:
            best_cost = c
            best_split = alloc

    return best_split, best_cost

def simulate_with_cost_trace(snapshots, lambda_over, lambda_under, theta_queue):
    filled = 0
    cost = 0
    cost_trace = []
    for ts, group in snapshots:
        if filled >= ORDER_SIZE:
            break
        venues = []
        for _, row in group.iterrows():
            venues.append({
                'ask': row['ask_px_00'],
                'ask_size': row['ask_sz_00'],
                'fee': 0.0,
                'rebate': 0.0
            })

        alloc, _ = allocator(venues, lambda_over, lambda_under, theta_queue)
        for i, venue in enumerate(venues):
            qty = min(venue['ask_size'], alloc[i], ORDER_SIZE - filled)
            cost += qty * venue['ask']
            filled += qty
            cost_trace.append(cost)
            if filled >= ORDER_SIZE:
                break

    avg_fill_price = cost / filled if filled > 0 else 0
    return cost, avg_fill_price, cost_trace

def plot_cost_trace(trace):
    plt.figure(figsize=(10, 6))
    plt.plot(trace, label='Cumulative Cost')
    plt.xlabel('Fill Event Count')
    plt.ylabel('Total Cost ($)')
    plt.title('Cumulative Cost of Order Execution')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results.png')

def baseline_best_ask(snapshots):
    filled = 0
    cost = 0
    for ts, group in snapshots:
        best_row = group.loc[group['ask_px_00'].idxmin()]
        px = best_row['ask_px_00']
        sz = best_row['ask_sz_00']
        qty = min(sz, ORDER_SIZE - filled)
        cost += qty * px
        filled += qty
        if filled >= ORDER_SIZE:
            break
    return cost, cost / filled if filled > 0 else 0

def baseline_twap(snapshots):
    n_buckets = 6
    bucket_size = len(snapshots) // n_buckets
    order_per_bucket = ORDER_SIZE // n_buckets
    total_cost = 0
    filled_total = 0

    for i in range(n_buckets):
        filled = 0
        bucket = snapshots[i * bucket_size : (i + 1) * bucket_size]
        for ts, group in bucket:
            for _, row in group.iterrows():
                px = row['ask_px_00']
                sz = row['ask_sz_00']
                qty = min(sz, order_per_bucket - filled)
                total_cost += qty * px
                filled += qty
                if filled >= order_per_bucket:
                    break
            if filled >= order_per_bucket:
                break
        filled_total += filled

    return total_cost, total_cost / filled_total if filled_total > 0 else 0

def baseline_vwap(snapshots):
    filled = 0
    cost = 0
    for ts, group in snapshots:
        group = group[group['ask_sz_00'] > 0]
        if group.empty:
            continue
        weighted_avg_price = (group['ask_px_00'] * group['ask_sz_00']).sum() / group['ask_sz_00'].sum()
        total_qty = min(ORDER_SIZE - filled, group['ask_sz_00'].sum())
        cost += total_qty * weighted_avg_price
        filled += total_qty
        if filled >= ORDER_SIZE:
            break
    return cost, cost / ORDER_SIZE

def compute_bps(base, custom):
    return 10000 * (base - custom) / base

def grid_search(snapshots):
    best_params = None
    best_total_cost = float('inf')
    best_avg_price = None
    best_trace = None

    lambdas = [1e-6, 1e-5, 1e-4]
    thetas = [1e-6, 1e-4, 1e-2]

    for lo, lu, tq in product(lambdas, lambdas, thetas):
        cost, avg, trace = simulate_with_cost_trace(snapshots, lo, lu, tq)
        if cost < best_total_cost:
            best_total_cost = cost
            best_avg_price = avg
            best_params = (lo, lu, tq)
            best_trace = trace

    return best_params, best_total_cost, best_avg_price, best_trace

def main():
    snapshots = load_data()
    best_params, total_cost, avg_fill, trace = grid_search(snapshots)
    lo, lu, tq = best_params

    plot_cost_trace(trace)

    best_cost, best_price = baseline_best_ask(snapshots)
    twap_cost, twap_price = baseline_twap(snapshots)
    vwap_cost, vwap_price = baseline_vwap(snapshots)

    result = {
        "best_parameters": {
            "lambda_over": lo,
            "lambda_under": lu,
            "theta_queue": tq
        },
        "our_strategy": {
            "total_cost": total_cost,
            "average_fill_price": avg_fill
        },
        "baseline_best_ask": {
            "total_cost": best_cost,
            "average_fill_price": best_price
        },
        "baseline_twap": {
            "total_cost": twap_cost,
            "average_fill_price": twap_price
        },
        "baseline_vwap": {
            "total_cost": vwap_cost,
            "average_fill_price": vwap_price
        },
        "savings_vs_best_ask_bps": compute_bps(best_price, avg_fill),
        "savings_vs_twap_bps": compute_bps(twap_price, avg_fill),
        "savings_vs_vwap_bps": compute_bps(vwap_price, avg_fill)
    }

    print(json.dumps(result, indent=2))
    print("\nFilled 5000 shares using optimal parameters — results.png saved.")

if __name__ == "__main__":
    main()
