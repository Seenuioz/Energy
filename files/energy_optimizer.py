#!/usr/bin/env python3
"""
Energy Consumption Optimization System using Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import json
import os
from datetime import datetime, timedelta

print("=" * 70)
print("   ENERGY CONSUMPTION OPTIMIZATION SYSTEM - ML PIPELINE")
print("=" * 70)

# ─────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ─────────────────────────────────────────────
print("\n[1/7] Generating synthetic energy dataset...")

np.random.seed(42)
n_samples = 8760  # 1 year of hourly data

timestamps = pd.date_range(start='2024-01-01', periods=n_samples, freq='h')
hours = timestamps.hour
days = timestamps.dayofweek
months = timestamps.month

# Realistic energy patterns
base_load = 200  # kWh
hour_pattern = 50 * np.sin((hours - 6) * np.pi / 12) + 30
day_pattern = np.where(days < 5, 40, -20)  # weekday vs weekend
seasonal = 30 * np.cos((months - 1) * np.pi / 6)
temperature = 22 + 12 * np.cos((months - 7) * np.pi / 6) + np.random.normal(0, 3, n_samples)
occupancy = np.clip(np.random.normal(0.6, 0.2, n_samples), 0, 1)
occupancy[hours < 7] *= 0.1
occupancy[hours > 22] *= 0.1

noise = np.random.normal(0, 15, n_samples)
energy = (base_load + hour_pattern + day_pattern + seasonal +
          25 * occupancy + 3 * (temperature - 22) + noise)
energy = np.clip(energy, 50, 500)
energy = np.array(energy, dtype=float)

# Inject anomalies
anomaly_idx = np.random.choice(n_samples, 80, replace=False)
energy[anomaly_idx] *= np.random.uniform(2.0, 3.5, 80)

df = pd.DataFrame({
    'timestamp': timestamps,
    'energy_kwh': energy,
    'temperature': temperature,
    'occupancy': occupancy,
    'hour': hours,
    'day_of_week': days,
    'month': months,
    'is_weekend': (days >= 5).astype(int),
    'is_peak_hour': ((hours >= 9) & (hours <= 18)).astype(int),
    'humidity': np.clip(60 + 15 * np.sin(hours * np.pi / 12) + np.random.normal(0, 5, n_samples), 20, 95),
    'solar_irradiance': np.clip(800 * np.maximum(0, np.sin((hours - 6) * np.pi / 12)) + np.random.normal(0, 50, n_samples), 0, 1000),
})

print(f"   ✓ Generated {n_samples:,} hourly records | Date range: {timestamps[0].date()} → {timestamps[-1].date()}")
print(f"   ✓ Energy: min={energy.min():.1f} kWh, max={energy.max():.1f} kWh, mean={energy.mean():.1f} kWh")

# ─────────────────────────────────────────────
# 2. ANOMALY DETECTION
# ─────────────────────────────────────────────
print("\n[2/7] Running anomaly detection (Isolation Forest)...")

features_anomaly = df[['energy_kwh', 'temperature', 'occupancy', 'hour', 'day_of_week']].copy()
scaler_anomaly = StandardScaler()
features_scaled = scaler_anomaly.fit_transform(features_anomaly)

iso_forest = IsolationForest(contamination=0.02, random_state=42, n_estimators=150)
df['anomaly'] = iso_forest.fit_predict(features_scaled)
df['anomaly_score'] = iso_forest.decision_function(features_scaled)
df['is_anomaly'] = (df['anomaly'] == -1).astype(int)

n_anomalies = df['is_anomaly'].sum()
anomaly_energy_waste = df[df['is_anomaly'] == 1]['energy_kwh'].sum() - \
                       df[df['is_anomaly'] == 1]['energy_kwh'].count() * df[df['is_anomaly'] == 0]['energy_kwh'].mean()
print(f"   ✓ Detected {n_anomalies} anomalies ({n_anomalies/n_samples*100:.1f}% of data)")
print(f"   ✓ Estimated anomaly energy waste: {anomaly_energy_waste:.0f} kWh/year")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3/7] Engineering features...")

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['temp_sq'] = df['temperature'] ** 2
df['occ_temp'] = df['occupancy'] * df['temperature']
df['energy_lag1'] = df['energy_kwh'].shift(1).bfill()
df['energy_lag24'] = df['energy_kwh'].shift(24).bfill()
df['energy_rolling_mean_6h'] = df['energy_kwh'].rolling(6, min_periods=1).mean()
df['energy_rolling_std_24h'] = df['energy_kwh'].rolling(24, min_periods=1).std().fillna(0)

feature_cols = ['temperature', 'occupancy', 'hour_sin', 'hour_cos',
                'month_sin', 'month_cos', 'day_sin', 'day_cos',
                'is_weekend', 'is_peak_hour', 'humidity', 'solar_irradiance',
                'temp_sq', 'occ_temp', 'energy_lag1', 'energy_lag24',
                'energy_rolling_mean_6h', 'energy_rolling_std_24h']

clean_df = df[df['is_anomaly'] == 0].copy()
X = clean_df[feature_cols]
y = clean_df['energy_kwh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"   ✓ {len(feature_cols)} features engineered | Train: {len(X_train):,} | Test: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n[4/7] Training ML models...")

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
    'Ridge Regression': Ridge(alpha=1.0),
}

results = {}
for name, model in models.items():
    if name == 'Ridge Regression':
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results[name] = {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape, 'predictions': y_pred, 'model': model}
    print(f"   ✓ {name:22s} → MAE: {mae:6.2f} kWh | RMSE: {rmse:6.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")

best_model_name = max(results, key=lambda k: results[k]['r2'])
best_model = results[best_model_name]['model']
best_preds = results[best_model_name]['predictions']
print(f"\n   ★ Best model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")

# ─────────────────────────────────────────────
# 5. USAGE PATTERN CLUSTERING
# ─────────────────────────────────────────────
print("\n[5/7] Clustering energy usage patterns (K-Means)...")

daily_profiles = []
daily_dates = []
for date, group in clean_df.groupby(clean_df['timestamp'].dt.date):
    if len(group) == 24:
        daily_profiles.append(group['energy_kwh'].values)
        daily_dates.append(date)

daily_profiles = np.array(daily_profiles)
scaler_km = MinMaxScaler()
profiles_scaled = scaler_km.fit_transform(daily_profiles)

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(profiles_scaled)
cluster_names = ['Low Demand', 'Peak Demand', 'Weekend Pattern', 'High Industrial']

cluster_stats = {}
for i in range(n_clusters):
    mask = cluster_labels == i
    cluster_stats[i] = {
        'name': cluster_names[i],
        'count': mask.sum(),
        'mean_daily': daily_profiles[mask].sum(axis=1).mean(),
        'profile': daily_profiles[mask].mean(axis=0)
    }
    print(f"   ✓ Cluster {i} ({cluster_names[i]:20s}): {mask.sum():3d} days | Avg daily: {cluster_stats[i]['mean_daily']:.0f} kWh")

# ─────────────────────────────────────────────
# 6. OPTIMIZATION RECOMMENDATIONS
# ─────────────────────────────────────────────
print("\n[6/7] Generating optimization recommendations...")

feature_importance = None
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

# Calculate savings potential
peak_mask = (clean_df['is_peak_hour'] == 1) & (clean_df['is_weekend'] == 0)
off_peak_mask = ~peak_mask
peak_avg = clean_df.loc[peak_mask, 'energy_kwh'].mean()
off_peak_avg = clean_df.loc[off_peak_mask, 'energy_kwh'].mean()
shift_potential = (peak_avg - off_peak_avg) * peak_mask.sum() * 0.15

anomaly_saving = abs(anomaly_energy_waste)
occupancy_saving = clean_df.loc[clean_df['occupancy'] < 0.2, 'energy_kwh'].mean() * \
                   (clean_df['occupancy'] < 0.2).sum() * 0.3
total_saving = shift_potential + anomaly_saving + occupancy_saving
total_cost_saving = total_saving * 0.12  # $0.12/kWh

recommendations = [
    {"id": 1, "category": "Load Shifting",       "saving_kwh": shift_potential,  "priority": "HIGH"},
    {"id": 2, "category": "Anomaly Elimination", "saving_kwh": anomaly_saving,   "priority": "CRITICAL"},
    {"id": 3, "category": "Low-Occupancy Trim",  "saving_kwh": occupancy_saving, "priority": "MEDIUM"},
]

for rec in recommendations:
    print(f"   [{rec['priority']:8s}] {rec['category']:22s}: Save {rec['saving_kwh']:,.0f} kWh/year (${rec['saving_kwh']*0.12:,.0f})")

print(f"\n   💰 TOTAL POTENTIAL SAVINGS: {total_saving:,.0f} kWh/year = ${total_cost_saving:,.0f}/year")

# ─────────────────────────────────────────────
# 7. VISUALIZATION DASHBOARD
# ─────────────────────────────────────────────
print("\n[7/7] Generating visualization dashboard...")

# Dark theme
plt.rcParams.update({
    'figure.facecolor': '#0a0e1a',
    'axes.facecolor': '#0d1117',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.5,
    'font.family': 'monospace',
})

NEON_GREEN   = '#00ff9f'
NEON_BLUE    = '#00b4ff'
NEON_ORANGE  = '#ff6e00'
NEON_PINK    = '#ff2d78'
NEON_YELLOW  = '#ffe600'
DARK_BG      = '#0a0e1a'
CARD_BG      = '#0d1117'
ACCENT       = '#161b22'

fig = plt.figure(figsize=(22, 16), facecolor=DARK_BG)
fig.suptitle('⚡  ENERGY CONSUMPTION OPTIMIZATION SYSTEM  ⚡',
             fontsize=18, fontweight='bold', color=NEON_GREEN,
             y=0.97, fontfamily='monospace')

gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)

# ── Plot 1: Weekly Energy Timeline ──
ax1 = fig.add_subplot(gs[0, :3])
week_sample = df.iloc[:168]
colors_ts = [NEON_PINK if a == 1 else NEON_BLUE for a in week_sample['is_anomaly']]
ax1.fill_between(range(len(week_sample)), week_sample['energy_kwh'],
                 alpha=0.15, color=NEON_BLUE)
ax1.scatter(range(len(week_sample)), week_sample['energy_kwh'],
            c=colors_ts, s=10, zorder=3)
ax1.plot(range(len(week_sample)), week_sample['energy_kwh'],
         color=NEON_BLUE, lw=1.2, alpha=0.7)
anomaly_positions = [i for i, a in enumerate(week_sample['is_anomaly']) if a == 1]
if anomaly_positions:
    ax1.scatter(anomaly_positions, week_sample['energy_kwh'].iloc[anomaly_positions],
                color=NEON_PINK, s=60, zorder=5, marker='X', label='Anomaly')
ax1.set_title('ENERGY TIMELINE — WEEK 1 (with Anomalies)', color=NEON_GREEN, fontsize=10, pad=8)
ax1.set_xlabel('Hour', fontsize=8)
ax1.set_ylabel('kWh', fontsize=8)
ax1.legend(fontsize=7, facecolor=CARD_BG, edgecolor='#30363d')
ax1.grid(True, alpha=0.3)
ax1.set_facecolor(CARD_BG)

# ── Plot 2: KPI Cards ──
ax_kpi = fig.add_subplot(gs[0, 3])
ax_kpi.set_facecolor(CARD_BG)
ax_kpi.axis('off')
kpis = [
    ('TOTAL RECORDS', f'{n_samples:,}', NEON_BLUE),
    ('ANOMALIES', f'{n_anomalies}', NEON_PINK),
    ('BEST R²', f'{results[best_model_name]["r2"]:.4f}', NEON_GREEN),
    ('SAVINGS/YR', f'${total_cost_saving:,.0f}', NEON_YELLOW),
]
for idx, (label, value, color) in enumerate(kpis):
    y_pos = 0.85 - idx * 0.22
    ax_kpi.add_patch(FancyBboxPatch((0.02, y_pos - 0.07), 0.96, 0.18,
                                    boxstyle="round,pad=0.02",
                                    facecolor=ACCENT, edgecolor=color, lw=1.5,
                                    transform=ax_kpi.transAxes))
    ax_kpi.text(0.5, y_pos + 0.06, label, ha='center', va='center',
                color='#8b949e', fontsize=6.5, transform=ax_kpi.transAxes)
    ax_kpi.text(0.5, y_pos - 0.01, value, ha='center', va='center',
                color=color, fontsize=11, fontweight='bold', transform=ax_kpi.transAxes)
ax_kpi.set_title('KEY METRICS', color=NEON_GREEN, fontsize=9, pad=8)

# ── Plot 3: Model Comparison ──
ax2 = fig.add_subplot(gs[1, :2])
model_names = list(results.keys())
r2_vals = [results[m]['r2'] for m in model_names]
mae_vals = [results[m]['mae'] for m in model_names]
x = np.arange(len(model_names))
w = 0.35
bars1 = ax2.bar(x - w/2, r2_vals, w, label='R² Score', color=NEON_GREEN, alpha=0.85)
ax2_r = ax2.twinx()
bars2 = ax2_r.bar(x + w/2, mae_vals, w, label='MAE (kWh)', color=NEON_ORANGE, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=7)
ax2.set_ylabel('R²', color=NEON_GREEN, fontsize=8)
ax2_r.set_ylabel('MAE (kWh)', color=NEON_ORANGE, fontsize=8)
ax2.set_title('MODEL PERFORMANCE COMPARISON', color=NEON_GREEN, fontsize=10, pad=8)
ax2.set_facecolor(CARD_BG)
ax2.yaxis.label.set_color(NEON_GREEN)
ax2_r.tick_params(colors=NEON_ORANGE)
lines = [plt.Rectangle((0,0),1,1, fc=NEON_GREEN), plt.Rectangle((0,0),1,1, fc=NEON_ORANGE)]
ax2.legend(lines, ['R²', 'MAE'], fontsize=7, facecolor=CARD_BG, edgecolor='#30363d')
ax2.grid(axis='y', alpha=0.3)

# ── Plot 4: Predicted vs Actual ──
ax3 = fig.add_subplot(gs[1, 2:])
sample_n = min(200, len(y_test))
ax3.scatter(y_test.values[:sample_n], best_preds[:sample_n],
            c=NEON_BLUE, alpha=0.6, s=15, edgecolors='none')
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
ax3.plot(lims, lims, color=NEON_PINK, lw=1.5, linestyle='--', label='Perfect Fit')
ax3.set_xlabel('Actual (kWh)', fontsize=8)
ax3.set_ylabel('Predicted (kWh)', fontsize=8)
ax3.set_title(f'PREDICTED vs ACTUAL — {best_model_name}', color=NEON_GREEN, fontsize=10, pad=8)
ax3.legend(fontsize=7, facecolor=CARD_BG, edgecolor='#30363d')
ax3.grid(True, alpha=0.3)
ax3.set_facecolor(CARD_BG)

# ── Plot 5: Daily Cluster Profiles ──
ax4 = fig.add_subplot(gs[2, :2])
cluster_colors = [NEON_GREEN, NEON_ORANGE, NEON_BLUE, NEON_PINK]
for i in range(n_clusters):
    ax4.plot(cluster_stats[i]['profile'], color=cluster_colors[i],
             lw=2, label=f"{cluster_names[i]} (n={cluster_stats[i]['count']})")
    ax4.fill_between(range(24), cluster_stats[i]['profile'],
                     alpha=0.08, color=cluster_colors[i])
ax4.set_xlabel('Hour of Day', fontsize=8)
ax4.set_ylabel('kWh', fontsize=8)
ax4.set_title('USAGE PATTERN CLUSTERS (K-Means)', color=NEON_GREEN, fontsize=10, pad=8)
ax4.legend(fontsize=6.5, facecolor=CARD_BG, edgecolor='#30363d')
ax4.grid(True, alpha=0.3)
ax4.set_facecolor(CARD_BG)
ax4.set_xticks(range(0, 24, 3))
ax4.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 3)], fontsize=6.5, rotation=30)

# ── Plot 6: Feature Importance ──
ax5 = fig.add_subplot(gs[2, 2:])
if feature_importance is not None:
    top_n = 10
    fi_top = feature_importance.head(top_n)
    colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))[::-1]
    bars = ax5.barh(range(top_n), fi_top.values, color=colors_fi, alpha=0.9)
    ax5.set_yticks(range(top_n))
    ax5.set_yticklabels(fi_top.index, fontsize=7)
    ax5.invert_yaxis()
    ax5.set_xlabel('Importance Score', fontsize=8)
    ax5.set_title('TOP FEATURE IMPORTANCES', color=NEON_GREEN, fontsize=10, pad=8)
    for i, (bar, val) in enumerate(zip(bars, fi_top.values)):
        ax5.text(val + 0.002, i, f'{val:.3f}', va='center', fontsize=6.5, color='#8b949e')
ax5.grid(axis='x', alpha=0.3)
ax5.set_facecolor(CARD_BG)

# ── Plot 7: Monthly Energy Heatmap ──
ax6 = fig.add_subplot(gs[3, :2])
monthly_hourly = clean_df.pivot_table(values='energy_kwh',
                                       index='month', columns='hour', aggfunc='mean')
im = ax6.imshow(monthly_hourly.values, aspect='auto',
                cmap='RdYlGn_r', interpolation='bilinear')
ax6.set_yticks(range(12))
ax6.set_yticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=7)
ax6.set_xticks(range(0, 24, 3))
ax6.set_xticklabels([f'{h:02d}h' for h in range(0, 24, 3)], fontsize=7)
ax6.set_title('ENERGY HEATMAP (Month × Hour)', color=NEON_GREEN, fontsize=10, pad=8)
plt.colorbar(im, ax=ax6, label='kWh', shrink=0.8)
ax6.set_facecolor(CARD_BG)

# ── Plot 8: Savings Breakdown ──
ax7 = fig.add_subplot(gs[3, 2:])
categories = [r['category'] for r in recommendations]
savings = [r['saving_kwh'] for r in recommendations]
bar_colors = [NEON_PINK, NEON_YELLOW, NEON_BLUE]
bars_sav = ax7.bar(categories, savings, color=bar_colors, alpha=0.85, width=0.5)
for bar, val in zip(bars_sav, savings):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,.0f}\nkWh', ha='center', va='bottom', fontsize=7.5,
             color='#c9d1d9', fontweight='bold')
ax7.set_ylabel('kWh / Year', fontsize=8)
ax7.set_title('OPTIMIZATION SAVINGS POTENTIAL', color=NEON_GREEN, fontsize=10, pad=8)
ax7.set_xticklabels([c.replace(' ', '\n') for c in categories], fontsize=8)
ax7.grid(axis='y', alpha=0.3)
ax7.set_facecolor(CARD_BG)

# Timestamp footer
fig.text(0.5, 0.01,
         f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}  |  '
         f'Best Model: {best_model_name}  |  Total Savings: ${total_cost_saving:,.0f}/yr',
         ha='center', fontsize=7.5, color='#484f58')

output_path = output_path = 'energy_optimization_dashboard.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight',
            facecolor=DARK_BG, edgecolor='none')
plt.close()
print(f"   ✓ Dashboard saved → {output_path}")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("   OPTIMIZATION REPORT SUMMARY")
print("=" * 70)
print(f"   Dataset        : {n_samples:,} hourly records (1 year)")
print(f"   Anomalies Found: {n_anomalies} ({n_anomalies/n_samples*100:.1f}%)")
print(f"   Best ML Model  : {best_model_name}")
print(f"   Model R²       : {results[best_model_name]['r2']:.4f}")
print(f"   Model MAPE     : {results[best_model_name]['mape']:.2f}%")
print(f"   Usage Clusters : {n_clusters} distinct patterns identified")
print(f"   Energy Savings : {total_saving:,.0f} kWh/year")
print(f"   Cost Savings   : ${total_cost_saving:,.0f}/year (@ $0.12/kWh)")
print(f"   CO₂ Reduction  : {total_saving * 0.386 / 1000:.1f} tonnes/year")
print("=" * 70)
print("   ✅ Pipeline complete. Dashboard exported successfully.")
print("=" * 70)
