"""
Plot saved optimization results.
Usage:  uv run python Energy/plot_results.py [results_file.npz]
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_results(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # (a) Generator dispatch
    ax = axes[0, 0]
    gen_p = data['gen_power']
    gen_on = data['gen_on']
    p_min = data['gen_params'].item()['p_min']
    p_max = data['gen_params'].item()['p_max']
    n_gen = len(gen_p)
    x = np.arange(n_gen)
    bars = ax.bar(x, gen_p, color=['steelblue' if o else 'lightgray' for o in gen_on])
    ax.plot(x, p_max, 'r_', ms=8, label='p_max')
    ax.plot(x, p_min, 'g_', ms=8, label='p_min')
    ax.set_xlabel('Generator'); ax.set_ylabel('Power (MW)')
    ax.set_title(f'Generator Dispatch ({np.sum(gen_on)}/{n_gen} on)')
    ax.legend(); ax.grid(alpha=0.3)

    # (b) Power mix pie
    ax = axes[0, 1]
    thermal = float(np.sum(gen_p * gen_on))
    wind = float(data['wind_power'])
    solar = float(data['solar_power'])
    battery = float(data['total_deterministic']) - thermal - wind - solar
    labels = ['Thermal', 'Wind', 'Solar', 'Battery']
    sizes = [thermal, wind, solar, max(0, battery)]
    colors = ['steelblue', 'limegreen', 'gold', 'coral']
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.0f%%',
        startangle=90)
    ax.set_title(f'Power Mix ({float(data["total_deterministic"]):.0f} MW)')

    # (c) Battery SoC
    ax = axes[0, 2]
    batt_soc = data['batt_soc']
    batt_params = data['battery_params']
    for i, soc in enumerate(batt_soc):
        b = batt_params[i]
        ax.bar(i, soc, color='coral', width=0.4)
        ax.axhline(y=0.5, color='gray', ls='--', alpha=0.5, label='init' if i==0 else '')
        ax.axhline(y=b['capacity']*0.1/b['capacity'], color='red', ls=':', alpha=0.5)
    ax.set_xticks(range(len(batt_soc)))
    ax.set_xticklabels([f'Batt {i}' for i in range(len(batt_soc))])
    ax.set_ylim(0, 1); ax.set_ylabel('SoC'); ax.set_title('Battery State of Charge')
    if len(batt_soc) == 2:
        ax.legend(fontsize=7)

    # (d) MC surplus histogram
    ax = axes[1, 0]
    surplus = data['mc_surplus']
    ax.hist(surplus, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='red', ls='--', lw=2, label='Demand met')
    ax.axvline(x=np.percentile(surplus, 100*data['alpha']),
               color='orange', ls='--', lw=2,
               label=f"VaR({1-data['alpha']:.0%})")
    ax.set_xlabel('Surplus (MW)'); ax.set_ylabel('Frequency')
    ax.set_title(f"MC: {data['violation_rate']:.1%} violations (target ≤{data['alpha']:.0%})")
    ax.legend()

    # (e) Generator cost curves
    ax = axes[1, 1]
    a = data['gen_params'].item()['a']
    b = data['gen_params'].item()['b']
    c = data['gen_params'].item()['c']
    for i in range(min(n_gen, 8)):  # show first 8
        p_range = np.linspace(p_min[i], p_max[i], 50)
        cost = a[i]*p_range**2 + b[i]*p_range + c[i]
        alpha_curve = 0.8 if gen_on[i] else 0.2
        ax.plot(p_range, cost / p_range,  # $/MWh
                color=f'C{i}', alpha=alpha_curve, lw=2 if gen_on[i] else 0.5,
                label=f'G{i} {"on" if gen_on[i] else "off"}')
        if gen_on[i]:
            ax.plot(gen_p[i], (a[i]*gen_p[i]**2 + b[i]*gen_p[i] + c[i]) / gen_p[i],
                    'o', color=f'C{i}', ms=6)
    ax.set_xlabel('Power (MW)'); ax.set_ylabel('Avg Cost ($/MWh)')
    ax.set_title('Generator Cost Curves'); ax.legend(fontsize=5, ncol=2)
    ax.grid(alpha=0.3)

    # (f) Key metrics + baseline comparison
    ax = axes[1, 2]
    ax.axis('off')
    greedy_cost = float(data.get('baseline_greedy_cost', 0))
    igo_cost = float(data.get('igo_cost', 0))
    metrics = [
        f"System: {n_gen} gen + {len(batt_soc)} batt",
        f"Wind: {data['wind_rated_mw']:.0f}MW Solar: {data['solar_rated_mw']:.0f}MW",
        f"Demand: {data['demand_nominal']:.0f} MW ±{25:.0f}",
        f"Chance: P≥{1-data['alpha']:.0%}, MC={data['mc_samples']}",
        "",
        f"--- Dispatch ---",
        f"Thermal: {thermal:.1f} MW ({np.sum(gen_on):.0f}/{n_gen})",
        f"Slack: {data['total_deterministic']-data['demand_nominal']:+.1f} MW",
        f"Batt discharge: {float(np.sum(data['batt_soc'] < 0.49))}/{len(batt_soc)}",
        "",
        f"--- Economics ---",
        f"IGO cost: ${igo_cost:.0f}",
        f"Greedy baseline: ${greedy_cost:.0f}",
        f"Δ: ${igo_cost-greedy_cost:+.0f}",
        "",
        f"--- Reliability ---",
        f"MC violation: {data['violation_rate']:.1%} (≤{data['alpha']:.0%})",
        f"Solve: {data['solve_time']:.1f}s",
    ]
    for i, m in enumerate(metrics):
        ax.text(0.05, 0.97 - i*0.047, m, transform=ax.transAxes,
                fontsize=8, family='monospace', verticalalignment='top')

    fig.suptitle('Stochastic UC + Storage — Black-Box IGO with Chance Constraint',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()

    out = Path(npz_path).parent / 'stochastic_uc_plot.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = Path(__file__).resolve().parent / "results" / "stochastic_uc.npz"
    plot_results(path)
