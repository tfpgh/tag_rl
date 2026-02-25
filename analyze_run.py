#!/usr/bin/env python3
"""
Comprehensive TensorBoardX run analysis script.
Usage: python analyze_run.py [--run-dir RUNS/DIR]

Reads TensorBoard event files and produces detailed diagnostics
for debugging training issues in the tag RL environment.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)


def load_scalars(run_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load all scalar time series from a TensorBoard run directory.
    Returns {tag: (steps, values)} with sorted arrays.
    """
    ea = EventAccumulator(run_dir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(f"ERROR: No scalar data found in {run_dir}")
        print(f"  Available data types: {list(ea.Tags().keys())}")
        sys.exit(1)

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        order = np.argsort(steps)
        data[tag] = (steps[order], values[order])
    return data


# ─── Statistical helpers ────────────────────────────────────────────────

def describe(values: np.ndarray, name: str = "") -> str:
    if len(values) == 0:
        return f"  {name}: (empty)"
    lines = [
        f"  {name}:" if name else "",
        f"    count    = {len(values)}",
        f"    mean     = {np.mean(values):.6f}",
        f"    std      = {np.std(values):.6f}",
        f"    min      = {np.min(values):.6f}",
        f"    25%      = {np.percentile(values, 25):.6f}",
        f"    median   = {np.median(values):.6f}",
        f"    75%      = {np.percentile(values, 75):.6f}",
        f"    max      = {np.max(values):.6f}",
    ]
    return "\n".join(lines)


def windowed_stats(steps: np.ndarray, values: np.ndarray, n_windows: int = 5):
    """Split a time series into n_windows equal parts and summarize each."""
    if len(values) < n_windows:
        return [(steps, values)]
    chunks = np.array_split(np.arange(len(values)), n_windows)
    results = []
    for idx in chunks:
        results.append((steps[idx], values[idx]))
    return results


def compute_trend(steps: np.ndarray, values: np.ndarray):
    """Linear regression slope and R² for a time series."""
    if len(values) < 2:
        return 0.0, 0.0
    x = steps.astype(np.float64)
    y = values.astype(np.float64)
    x_norm = (x - x.mean()) / (x.std() + 1e-12)
    slope, intercept = np.polyfit(x_norm, y, 1)
    y_pred = slope * x_norm + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    # Convert slope back to per-step units
    real_slope = slope / (x.std() + 1e-12)
    return real_slope, r2


def detect_plateaus(values: np.ndarray, window: int = 50, threshold: float = 1e-5):
    """Detect if a metric has plateaued (rolling std below threshold)."""
    if len(values) < window:
        return False, 0.0
    rolling_std = np.array([
        np.std(values[max(0, i - window):i + 1])
        for i in range(len(values))
    ])
    recent_std = np.mean(rolling_std[-window:])
    return recent_std < threshold, recent_std


def detect_spikes(values: np.ndarray, z_threshold: float = 4.0):
    """Detect outlier spikes using z-score method."""
    if len(values) < 10:
        return [], 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std < 1e-12:
        return [], 0.0
    z_scores = np.abs((values - mean) / std)
    spike_indices = np.where(z_scores > z_threshold)[0]
    return spike_indices.tolist(), float(np.max(z_scores))


def compute_ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average."""
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
    return ema


# ─── Analysis sections ──────────────────────────────────────────────────

def analyze_overview(data: dict):
    print("=" * 80)
    print("1. OVERVIEW")
    print("=" * 80)
    print(f"\nAvailable metrics ({len(data)} total):")
    for tag in sorted(data.keys()):
        steps, vals = data[tag]
        print(f"  {tag:40s}  {len(vals):6d} points, steps [{steps[0]:>12,.0f} .. {steps[-1]:>12,.0f}]")

    # Determine training progress
    any_steps = next(iter(data.values()))[0]
    total_steps = any_steps[-1]
    target = 1e9
    print(f"\n  Latest step: {total_steps:,.0f}")
    print(f"  Target steps: {target:,.0f}")
    print(f"  Progress: {100 * total_steps / target:.2f}%")
    print(f"  Number of updates logged: {len(any_steps)}")
    if len(any_steps) >= 2:
        step_size = int(np.median(np.diff(any_steps)))
        print(f"  Steps per update (median): {step_size:,}")


def analyze_rewards(data: dict):
    print("\n" + "=" * 80)
    print("2. REWARD ANALYSIS")
    print("=" * 80)

    for agent in ["chaser", "evader"]:
        tag = f"{agent}/mean_reward"
        if tag not in data:
            print(f"\n  WARNING: {tag} not found!")
            continue

        steps, vals = data[tag]
        print(f"\n--- {agent.upper()} Rewards ---")
        print(describe(vals, "Full run"))

        # Trend
        slope, r2 = compute_trend(steps, vals)
        direction = "INCREASING" if slope > 0 else "DECREASING" if slope < 0 else "FLAT"
        print(f"\n  Trend: {direction} (slope={slope:.2e}/step, R²={r2:.4f})")

        # Windowed progression
        windows = windowed_stats(steps, vals, n_windows=5)
        print(f"\n  Progression over {len(windows)} windows:")
        for i, (ws, wv) in enumerate(windows):
            label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
            print(f"    [{label}] steps {ws[0]:>12,.0f}-{ws[-1]:>12,.0f}:  "
                  f"mean={np.mean(wv):+.6f}  std={np.std(wv):.6f}  "
                  f"min={np.min(wv):+.6f}  max={np.max(wv):+.6f}")

        # Plateau detection
        is_plateau, recent_std = detect_plateaus(vals)
        if is_plateau:
            print(f"\n  ⚠ PLATEAU DETECTED: recent rolling std = {recent_std:.2e}")

        # Spike detection
        spikes, max_z = detect_spikes(vals)
        if spikes:
            print(f"\n  ⚠ {len(spikes)} SPIKES detected (max z-score = {max_z:.1f})")
            for idx in spikes[:10]:  # Show first 10
                print(f"      step {steps[idx]:>12,.0f}: value = {vals[idx]:+.6f}")

    # Reward balance
    if "chaser/mean_reward" in data and "evader/mean_reward" in data:
        _, c_vals = data["chaser/mean_reward"]
        _, e_vals = data["evader/mean_reward"]
        n = min(len(c_vals), len(e_vals))
        ratio = np.mean(c_vals[:n]) / (np.mean(e_vals[:n]) + 1e-12)
        diff = np.mean(c_vals[:n]) - np.mean(e_vals[:n])
        print(f"\n--- Reward Balance ---")
        print(f"  Mean chaser reward:  {np.mean(c_vals):+.6f}")
        print(f"  Mean evader reward:  {np.mean(e_vals):+.6f}")
        print(f"  Difference (C - E):  {diff:+.6f}")
        print(f"  Ratio (C / E):       {ratio:+.4f}")
        # Correlation
        corr = np.corrcoef(c_vals[:n], e_vals[:n])[0, 1]
        print(f"  Correlation:         {corr:+.4f}")
        if abs(corr + 1) < 0.1:
            print("  → Rewards are strongly anti-correlated (expected for zero-sum)")
        elif corr > 0.5:
            print("  ⚠ Rewards are positively correlated — unexpected for adversarial game")


def analyze_losses(data: dict):
    print("\n" + "=" * 80)
    print("3. LOSS ANALYSIS")
    print("=" * 80)

    for agent in ["chaser", "evader"]:
        print(f"\n--- {agent.upper()} ---")

        for loss_type in ["total_loss", "actor_loss", "value_loss"]:
            tag = f"{agent}/{loss_type}"
            if tag not in data:
                print(f"  {loss_type}: NOT FOUND")
                continue
            steps, vals = data[tag]
            print(f"\n  {loss_type}:")
            print(f"    mean={np.mean(vals):.6f}  std={np.std(vals):.6f}  "
                  f"min={np.min(vals):.6f}  max={np.max(vals):.6f}")

            slope, r2 = compute_trend(steps, vals)
            print(f"    trend: slope={slope:.2e}/step  R²={r2:.4f}")

            # Check for NaN/Inf
            n_nan = np.sum(np.isnan(vals))
            n_inf = np.sum(np.isinf(vals))
            if n_nan > 0:
                print(f"    ⚠ {n_nan} NaN values detected!")
            if n_inf > 0:
                print(f"    ⚠ {n_inf} Inf values detected!")

            # Check for loss explosion
            if len(vals) > 20:
                early_mean = np.mean(vals[:len(vals)//5])
                late_mean = np.mean(vals[-len(vals)//5:])
                if abs(late_mean) > 10 * abs(early_mean) and abs(early_mean) > 1e-8:
                    print(f"    ⚠ LOSS EXPLOSION: early mean={early_mean:.4f} → late mean={late_mean:.4f} "
                          f"({late_mean/early_mean:.1f}x increase)")

            # Windowed progression
            windows = windowed_stats(steps, vals, n_windows=5)
            for i, (ws, wv) in enumerate(windows):
                label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
                print(f"    [{label}] mean={np.mean(wv):+.6f}  std={np.std(wv):.6f}")

            # Spike detection
            spikes, max_z = detect_spikes(vals)
            if spikes:
                print(f"    ⚠ {len(spikes)} spikes (max z={max_z:.1f})")

        # Value loss diagnostic
        vl_tag = f"{agent}/value_loss"
        if vl_tag in data:
            _, vl_vals = data[vl_tag]
            print(f"\n  Value loss magnitude check:")
            print(f"    If value_loss >> 1.0, the critic is poorly calibrated.")
            print(f"    Current mean value_loss: {np.mean(vl_vals):.6f}")
            if np.mean(vl_vals) > 1.0:
                print(f"    ⚠ Value loss is HIGH — critic may be struggling to fit returns")
            elif np.mean(vl_vals) < 0.001:
                print(f"    ⚠ Value loss is very LOW — critic may have collapsed")


def analyze_entropy(data: dict):
    print("\n" + "=" * 80)
    print("4. ENTROPY ANALYSIS (Exploration vs Exploitation)")
    print("=" * 80)

    for agent in ["chaser", "evader"]:
        tag = f"{agent}/entropy"
        if tag not in data:
            print(f"\n  {agent}: entropy NOT FOUND")
            continue
        steps, vals = data[tag]
        print(f"\n--- {agent.upper()} Entropy ---")
        print(describe(vals, "Full run"))

        slope, r2 = compute_trend(steps, vals)
        print(f"\n  Trend: slope={slope:.2e}/step, R²={r2:.4f}")

        # Windowed
        windows = windowed_stats(steps, vals, n_windows=5)
        print(f"\n  Progression:")
        for i, (ws, wv) in enumerate(windows):
            label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
            print(f"    [{label}] mean={np.mean(wv):.6f}  std={np.std(wv):.6f}")

        # Entropy collapse check
        if len(vals) > 0:
            recent_entropy = np.mean(vals[-max(1, len(vals)//10):])
            initial_entropy = np.mean(vals[:max(1, len(vals)//10)])
            print(f"\n  Initial entropy: {initial_entropy:.6f}")
            print(f"  Recent entropy:  {recent_entropy:.6f}")
            if initial_entropy > 0:
                pct_change = (recent_entropy - initial_entropy) / abs(initial_entropy) * 100
                print(f"  Change:          {pct_change:+.1f}%")
            if recent_entropy < 0.1:
                print(f"  ⚠ ENTROPY COLLAPSE — policy is nearly deterministic!")
                print(f"    Consider increasing ent_coef (currently 0.03)")
            elif recent_entropy < initial_entropy * 0.3:
                print(f"  ⚠ Significant entropy drop — exploration may be insufficient")

    # Compare agents
    if "chaser/entropy" in data and "evader/entropy" in data:
        _, c_ent = data["chaser/entropy"]
        _, e_ent = data["evader/entropy"]
        print(f"\n--- Entropy Comparison ---")
        print(f"  Chaser mean: {np.mean(c_ent):.6f}  Evader mean: {np.mean(e_ent):.6f}")
        if len(c_ent) > 0 and len(e_ent) > 0:
            c_recent = np.mean(c_ent[-max(1, len(c_ent)//10):])
            e_recent = np.mean(e_ent[-max(1, len(e_ent)//10):])
            print(f"  Chaser recent: {c_recent:.6f}  Evader recent: {e_recent:.6f}")
            if abs(c_recent - e_recent) > 0.5:
                print(f"  ⚠ Large entropy gap between agents — one may be dominating")


def analyze_environment(data: dict):
    print("\n" + "=" * 80)
    print("5. ENVIRONMENT METRICS")
    print("=" * 80)

    # Tag rate
    if "env/tag_rate" in data:
        steps, vals = data["env/tag_rate"]
        print(f"\n--- Tag Rate ---")
        print(describe(vals, "Full run"))
        slope, r2 = compute_trend(steps, vals)
        print(f"\n  Trend: slope={slope:.2e}/step, R²={r2:.4f}")

        windows = windowed_stats(steps, vals, n_windows=5)
        print(f"\n  Progression:")
        for i, (ws, wv) in enumerate(windows):
            label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
            print(f"    [{label}] mean={np.mean(wv):.4f}  std={np.std(wv):.4f}")

        recent_tag_rate = np.mean(vals[-max(1, len(vals)//10):])
        print(f"\n  Recent tag rate: {recent_tag_rate:.4f}")
        if recent_tag_rate < 0.05:
            print(f"  ⚠ Very low tag rate — chaser is failing to catch evader")
            print(f"    Possible causes: reward too sparse, evader too strong, chaser not exploring")
        elif recent_tag_rate > 0.95:
            print(f"  ⚠ Very high tag rate — evader is failing to escape")
            print(f"    Possible causes: evader policy collapsed, chaser too strong")
        elif 0.3 <= recent_tag_rate <= 0.7:
            print(f"  ✓ Tag rate in healthy competitive range")

    # Mean distance
    if "env/mean_distance" in data:
        steps, vals = data["env/mean_distance"]
        print(f"\n--- Mean Distance ---")
        print(describe(vals, "Full run"))
        slope, r2 = compute_trend(steps, vals)
        print(f"\n  Trend: slope={slope:.2e}/step, R²={r2:.4f}")

        windows = windowed_stats(steps, vals, n_windows=5)
        print(f"\n  Progression:")
        for i, (ws, wv) in enumerate(windows):
            label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
            print(f"    [{label}] mean={np.mean(wv):.4f}  std={np.std(wv):.4f}")

        # Interpret distance in context of arena (1.5m × 1.5m, diagonal ≈ 2.12m)
        recent_dist = np.mean(vals[-max(1, len(vals)//10):])
        arena_diag = np.sqrt(1.5**2 + 1.5**2)
        print(f"\n  Recent mean distance: {recent_dist:.4f}")
        print(f"  Arena diagonal:       {arena_diag:.4f}")
        print(f"  Relative distance:    {recent_dist/arena_diag:.2%} of diagonal")

    # Episode length
    if "env/mean_episode_length" in data:
        steps, vals = data["env/mean_episode_length"]
        print(f"\n--- Mean Episode Length ---")
        print(describe(vals, "Full run"))
        slope, r2 = compute_trend(steps, vals)
        print(f"\n  Trend: slope={slope:.2e}/step, R²={r2:.4f}")

        windows = windowed_stats(steps, vals, n_windows=5)
        print(f"\n  Progression:")
        for i, (ws, wv) in enumerate(windows):
            label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
            print(f"    [{label}] mean={np.mean(wv):.2f}  std={np.std(wv):.2f}")

        recent_ep_len = np.mean(vals[-max(1, len(vals)//10):])
        max_ep_len = 600  # 30s × 20Hz
        print(f"\n  Recent mean episode length: {recent_ep_len:.1f} steps ({recent_ep_len/20:.1f}s)")
        print(f"  Max episode length:         {max_ep_len} steps ({max_ep_len/20:.1f}s)")
        if recent_ep_len > max_ep_len * 0.95:
            print(f"  ⚠ Episodes almost always timing out — chaser never catches evader")
        elif recent_ep_len < max_ep_len * 0.1:
            print(f"  ⚠ Episodes very short — chaser catches evader almost immediately")


def analyze_actions(data: dict):
    print("\n" + "=" * 80)
    print("6. ACTION ANALYSIS")
    print("=" * 80)

    for agent in ["chaser", "evader"]:
        tag = f"{agent}/mean_action_magnitude"
        if tag not in data:
            print(f"\n  {agent}: action magnitude NOT FOUND")
            continue
        steps, vals = data[tag]
        print(f"\n--- {agent.upper()} Action Magnitude ---")
        print(describe(vals, "Full run"))

        slope, r2 = compute_trend(steps, vals)
        print(f"\n  Trend: slope={slope:.2e}/step, R²={r2:.4f}")

        windows = windowed_stats(steps, vals, n_windows=5)
        print(f"\n  Progression:")
        for i, (ws, wv) in enumerate(windows):
            label = "FIRST" if i == 0 else "LAST" if i == len(windows) - 1 else f"  {i+1} "
            print(f"    [{label}] mean={np.mean(wv):.4f}  std={np.std(wv):.4f}")

        # Check for action saturation (actions clipped to [-1, 1])
        recent_mag = np.mean(vals[-max(1, len(vals)//10):])
        if recent_mag > 0.95:
            print(f"  ⚠ Actions near saturation (mean mag={recent_mag:.3f})")
            print(f"    Policy may be bang-bang — consider if this is desired")
        elif recent_mag < 0.05:
            print(f"  ⚠ Actions near zero — agent may be frozen/not learning")


def analyze_stability(data: dict):
    print("\n" + "=" * 80)
    print("7. TRAINING STABILITY ANALYSIS")
    print("=" * 80)

    # Check for NaN/Inf in all metrics
    print("\n--- NaN/Inf Check ---")
    has_problems = False
    for tag in sorted(data.keys()):
        _, vals = data[tag]
        n_nan = np.sum(np.isnan(vals))
        n_inf = np.sum(np.isinf(vals))
        if n_nan > 0 or n_inf > 0:
            has_problems = True
            print(f"  ⚠ {tag}: {n_nan} NaN, {n_inf} Inf")
    if not has_problems:
        print(f"  ✓ No NaN or Inf values in any metric")

    # Variance analysis
    print(f"\n--- Metric Variance (coefficient of variation) ---")
    for tag in sorted(data.keys()):
        _, vals = data[tag]
        mean = np.mean(vals)
        std = np.std(vals)
        cv = std / (abs(mean) + 1e-12)
        status = ""
        if cv > 5.0:
            status = " ⚠ EXTREMELY NOISY"
        elif cv > 2.0:
            status = " ⚠ VERY NOISY"
        elif cv > 1.0:
            status = " (noisy)"
        print(f"  {tag:40s}  CV={cv:.4f}{status}")

    # Smoothed trend analysis
    print(f"\n--- Smoothed Recent Trends (last 20% of training) ---")
    for tag in sorted(data.keys()):
        steps, vals = data[tag]
        n = len(vals)
        if n < 10:
            continue
        cutoff = max(1, int(n * 0.8))
        recent_steps = steps[cutoff:]
        recent_vals = vals[cutoff:]
        slope, r2 = compute_trend(recent_steps, recent_vals)
        ema_vals = compute_ema(recent_vals, alpha=0.1)
        ema_start = ema_vals[0]
        ema_end = ema_vals[-1]
        pct = (ema_end - ema_start) / (abs(ema_start) + 1e-12) * 100
        direction = "↑" if pct > 1 else "↓" if pct < -1 else "→"
        print(f"  {tag:40s}  {direction} EMA: {ema_start:+.6f} → {ema_end:+.6f} ({pct:+.1f}%)")


def analyze_learning_dynamics(data: dict):
    print("\n" + "=" * 80)
    print("8. LEARNING DYNAMICS & COMPETITIVE BALANCE")
    print("=" * 80)

    # Check if one agent is dominating
    c_reward_tag = "chaser/mean_reward"
    e_reward_tag = "evader/mean_reward"
    tag_rate_tag = "env/tag_rate"

    if tag_rate_tag in data:
        _, tag_rate = data[tag_rate_tag]
        print(f"\n--- Competitive Balance Over Time ---")
        windows = windowed_stats(*data[tag_rate_tag], n_windows=10)
        print(f"  Window | Tag Rate | Interpretation")
        print(f"  -------+----------+------------------")
        for i, (ws, wv) in enumerate(windows):
            rate = np.mean(wv)
            interp = ""
            if rate < 0.2:
                interp = "Evader dominates"
            elif rate < 0.4:
                interp = "Evader advantage"
            elif rate <= 0.6:
                interp = "Balanced"
            elif rate <= 0.8:
                interp = "Chaser advantage"
            else:
                interp = "Chaser dominates"
            print(f"    {i+1:2d}   |  {rate:.4f}  | {interp}")

    # Reward volatility comparison
    if c_reward_tag in data and e_reward_tag in data:
        _, c_vals = data[c_reward_tag]
        _, e_vals = data[e_reward_tag]

        print(f"\n--- Reward Volatility ---")
        # Rolling std with window of ~10% of data
        def rolling_std(vals, window):
            if len(vals) < window:
                return np.array([np.std(vals)])
            return np.array([np.std(vals[max(0,i-window):i+1]) for i in range(len(vals))])

        window = max(10, len(c_vals) // 10)
        c_vol = rolling_std(c_vals, window)
        e_vol = rolling_std(e_vals, window)
        print(f"  Chaser reward volatility (recent):  {np.mean(c_vol[-window:]):.6f}")
        print(f"  Evader reward volatility (recent):  {np.mean(e_vol[-window:]):.6f}")

    # Learning rate schedule diagnostic
    print(f"\n--- Learning Rate Schedule Diagnostic ---")
    print(f"  Config: lr=3e-4, anneal_lr=True")
    any_steps = next(iter(data.values()))[0]
    if len(any_steps) > 0:
        current_step = any_steps[-1]
        total = 1e9
        frac_remaining = 1.0 - current_step / total
        current_lr = 3e-4 * frac_remaining
        print(f"  Estimated current LR: {current_lr:.2e} (started at 3.00e-04)")
        print(f"  LR decay progress:    {(1-frac_remaining)*100:.1f}%")


def analyze_correlations(data: dict):
    print("\n" + "=" * 80)
    print("9. CROSS-METRIC CORRELATIONS")
    print("=" * 80)

    # Key pairs to check
    pairs = [
        ("chaser/mean_reward", "env/tag_rate"),
        ("evader/mean_reward", "env/tag_rate"),
        ("chaser/entropy", "chaser/mean_reward"),
        ("evader/entropy", "evader/mean_reward"),
        ("chaser/value_loss", "chaser/mean_reward"),
        ("evader/value_loss", "evader/mean_reward"),
        ("chaser/actor_loss", "chaser/entropy"),
        ("evader/actor_loss", "evader/entropy"),
        ("env/tag_rate", "env/mean_distance"),
        ("env/tag_rate", "env/mean_episode_length"),
        ("chaser/mean_action_magnitude", "chaser/mean_reward"),
        ("evader/mean_action_magnitude", "evader/mean_reward"),
    ]

    print(f"\n  {'Metric A':40s}  {'Metric B':40s}  Corr")
    print(f"  {'-'*40}  {'-'*40}  ------")
    for a, b in pairs:
        if a in data and b in data:
            _, va = data[a]
            _, vb = data[b]
            n = min(len(va), len(vb))
            if n > 1:
                corr = np.corrcoef(va[:n], vb[:n])[0, 1]
                flag = " ⚠" if abs(corr) > 0.8 else ""
                print(f"  {a:40s}  {b:40s}  {corr:+.4f}{flag}")


def generate_diagnosis(data: dict):
    print("\n" + "=" * 80)
    print("10. AUTOMATED DIAGNOSIS SUMMARY")
    print("=" * 80)

    issues = []
    good = []

    # Check rewards improving
    for agent in ["chaser", "evader"]:
        tag = f"{agent}/mean_reward"
        if tag in data:
            steps, vals = data[tag]
            slope, r2 = compute_trend(steps, vals)
            if slope < -1e-12 and r2 > 0.1:
                issues.append(f"{agent} reward is DECREASING over training (slope={slope:.2e}, R²={r2:.3f})")
            elif abs(slope) < 1e-12 or r2 < 0.01:
                is_plateau, _ = detect_plateaus(vals)
                if is_plateau:
                    issues.append(f"{agent} reward has PLATEAUED — not improving")
            else:
                good.append(f"{agent} reward is improving (slope={slope:.2e})")

    # Check entropy
    for agent in ["chaser", "evader"]:
        tag = f"{agent}/entropy"
        if tag in data:
            _, vals = data[tag]
            recent = np.mean(vals[-max(1, len(vals)//10):])
            initial = np.mean(vals[:max(1, len(vals)//10)])
            if recent < 0.1:
                issues.append(f"{agent} entropy has COLLAPSED ({recent:.4f}) — policy nearly deterministic")
            elif recent < initial * 0.3:
                issues.append(f"{agent} entropy dropped significantly ({initial:.4f} → {recent:.4f})")

    # Check tag rate
    if "env/tag_rate" in data:
        _, vals = data["env/tag_rate"]
        recent = np.mean(vals[-max(1, len(vals)//10):])
        if recent < 0.05:
            issues.append(f"Tag rate extremely low ({recent:.4f}) — chaser is failing")
        elif recent > 0.95:
            issues.append(f"Tag rate extremely high ({recent:.4f}) — evader has collapsed")
        elif 0.3 <= recent <= 0.7:
            good.append(f"Tag rate in competitive range ({recent:.4f})")

    # Check value loss
    for agent in ["chaser", "evader"]:
        tag = f"{agent}/value_loss"
        if tag in data:
            _, vals = data[tag]
            recent = np.mean(vals[-max(1, len(vals)//10):])
            if recent > 5.0:
                issues.append(f"{agent} value loss is very high ({recent:.4f}) — critic poorly calibrated")
            n_nan = np.sum(np.isnan(vals))
            if n_nan > 0:
                issues.append(f"{agent} value loss has {n_nan} NaN values!")

    # Check episode length
    if "env/mean_episode_length" in data:
        _, vals = data["env/mean_episode_length"]
        recent = np.mean(vals[-max(1, len(vals)//10):])
        if recent > 570:  # 95% of max 600
            issues.append(f"Episodes almost always timing out ({recent:.0f}/600 steps) — chaser can't tag")
        elif recent < 60:
            issues.append(f"Episodes extremely short ({recent:.0f}/600 steps) — evader caught instantly")

    # Check for NaN/Inf anywhere
    for tag in data:
        _, vals = data[tag]
        if np.any(np.isnan(vals)):
            issues.append(f"NaN values in {tag}")
        if np.any(np.isinf(vals)):
            issues.append(f"Inf values in {tag}")

    # Check loss explosion
    for agent in ["chaser", "evader"]:
        for lt in ["total_loss", "actor_loss", "value_loss"]:
            tag = f"{agent}/{lt}"
            if tag in data:
                _, vals = data[tag]
                if len(vals) > 20:
                    early = np.mean(vals[:len(vals)//5])
                    late = np.mean(vals[-len(vals)//5:])
                    if abs(late) > 10 * abs(early) and abs(early) > 1e-6:
                        issues.append(f"{agent}/{lt} has EXPLODED: {early:.4f} → {late:.4f}")

    # Print results
    if issues:
        print(f"\n  ISSUES FOUND ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. ⚠ {issue}")
    else:
        print(f"\n  No critical issues detected.")

    if good:
        print(f"\n  POSITIVE SIGNALS ({len(good)}):")
        for i, g in enumerate(good, 1):
            print(f"    {i}. ✓ {g}")

    # Suggestions
    print(f"\n--- Suggested Actions ---")
    if any("entropy" in i.lower() and "collapse" in i.lower() for i in issues):
        print(f"  • Increase ent_coef (currently 0.03) to encourage more exploration")
    if any("plateau" in i.lower() for i in issues):
        print(f"  • Consider increasing learning rate or changing reward shaping")
        print(f"  • Try different discount factor (gamma) or GAE lambda")
    if any("tag rate extremely low" in i.lower() for i in issues):
        print(f"  • Increase distance_shaping_scale to give chaser stronger approach signal")
        print(f"  • Consider curriculum: start with smaller arena or slower evader")
    if any("tag rate extremely high" in i.lower() for i in issues):
        print(f"  • Increase distance_shaping for evader to encourage fleeing")
        print(f"  • Check if evader policy has collapsed (entropy too low)")
    if any("value loss" in i.lower() and "high" in i.lower() for i in issues):
        print(f"  • Decrease vf_coef or increase num_minibatches for more stable value updates")
        print(f"  • Check if reward scale is appropriate")
    if any("exploded" in i.lower() for i in issues):
        print(f"  • Reduce learning rate")
        print(f"  • Increase max_grad_norm clipping (currently 0.5)")
    if any("nan" in i.lower() for i in issues):
        print(f"  • NaN detected — check for division by zero in reward or loss")
        print(f"  • Try reducing learning rate or increasing gradient clipping")
    if not issues:
        print(f"  • Training appears healthy — continue monitoring")
        print(f"  • If performance is unsatisfactory despite healthy metrics, consider:")
        print(f"    - Increasing network capacity (hidden_size)")
        print(f"    - Longer rollouts (num_steps)")
        print(f"    - Different reward shaping parameters")


def dump_raw_recent(data: dict, n_points: int = 20):
    """Dump the most recent N data points for each metric."""
    print("\n" + "=" * 80)
    print(f"11. RAW RECENT DATA (last {n_points} points)")
    print("=" * 80)

    for tag in sorted(data.keys()):
        steps, vals = data[tag]
        recent = min(n_points, len(vals))
        print(f"\n  {tag}:")
        print(f"    {'Step':>14s}  {'Value':>14s}")
        for i in range(-recent, 0):
            print(f"    {steps[i]:14,.0f}  {vals[i]:+14.6f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze TensorBoardX training run")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="runs/Feb25_12-36-44_node002",
        help="Path to the TensorBoard run directory",
    )
    parser.add_argument(
        "--raw-points",
        type=int,
        default=20,
        help="Number of recent raw data points to dump (default: 20)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not Path(run_dir).exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        print(f"Available runs:")
        runs_dir = Path("runs")
        if runs_dir.exists():
            for d in sorted(runs_dir.iterdir()):
                print(f"  {d}")
        else:
            print(f"  (no 'runs' directory found)")
        sys.exit(1)

    print(f"Loading TensorBoard data from: {run_dir}")
    data = load_scalars(run_dir)
    print(f"Loaded {len(data)} metrics.\n")

    analyze_overview(data)
    analyze_rewards(data)
    analyze_losses(data)
    analyze_entropy(data)
    analyze_environment(data)
    analyze_actions(data)
    analyze_stability(data)
    analyze_learning_dynamics(data)
    analyze_correlations(data)
    generate_diagnosis(data)
    dump_raw_recent(data, n_points=args.raw_points)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nCopy-paste the full output above and send it back for further analysis.")


if __name__ == "__main__":
    main()
