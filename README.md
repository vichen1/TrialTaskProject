# Cont & Kukanov Static Smart Order Router – Backtest

## Overview

This project implements and back-tests a static smart order router based on the model from Cont & Kukanov (2013). The router allocates a 5000-share buy order across venues by minimizing a cost function with penalties for queue risk, overfill, and underfill.

## Files

- `backtest.py`: Main script that runs allocation, parameter tuning, and baseline comparison.
- `results.png`: Cumulative cost plot for the optimal strategy.
- `README.md`: Project overview and explanation.

## How It Works

1. **Data Preprocessing**:
   - Loads `l1_day.csv`, keeping the first message per venue at each timestamp.
   - Constructs a time-ordered stream of Level-1 snapshots.

2. **Static Allocator**:
   - Exhaustive search over share splits in 100-share steps.
   - Cost = execution + λ_over × overfill + λ_under × underfill + θ_queue × queue_penalty

3. **Execution Simulation**:
   - Fills orders across time using the allocator output.
   - Tracks cost over time for plotting.

4. **Baselines**:
   - **Best Ask**: Always takes from the lowest price.
   - **TWAP**: Uniform fill across time windows.
   - **VWAP**: Weighted by venue ask size.

5. **Parameter Grid Search**:
   - λ_over, λ_under ∈ {1e-6, 1e-5, 1e-4}
   - θ_queue ∈ {1e-6, 1e-4, 1e-2}

## Output

- JSON object with:
  - Best parameters
  - Total cost and fill price
  - Cost and price for all baselines
  - Savings vs each baseline (bps)
- Plot of cumulative cost saved as `results.png`

## Suggested Improvement

To increase realism, the model could incorporate **queue position**:
- Add a fill-probability function based on queue depth.
- Penalize large orders that exceed top-of-book visibility.

## Notes

- Runs in under 2 minutes on standard hardware.
- Only uses `pandas`, `numpy`, and `matplotlib`.
