import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/mnt/user-data/uploads/heart_ccv.csv')

# ── Feature label maps ──────────────────────────────────────────────────────
cp_map   = {0: 'Typical\nAngina', 1: 'Atypical\nAngina', 2: 'Non-anginal\nPain', 3: 'Asymptomatic'}
thal_map = {0: 'No thal', 1: 'Normal', 2: 'Fixed\nDefect', 3: 'Reversable\nDefect'}
slope_map= {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}

df['age_group'] = pd.cut(df['age'], bins=[29,40,50,60,70,80],
                          labels=['30s','40s','50s','60s','70s+'])
df['cp_label']   = df['cp'].map(cp_map)
df['thal_label'] = df['thal'].map(thal_map)

# ── Colour palette ──────────────────────────────────────────────────────────
BG       = '#0D1117'
CARD     = '#161B22'
BORDER   = '#21262D'
POSITIVE = '#F87171'   # heart disease present
NEGATIVE = '#34D399'   # no heart disease
BLUE     = '#58A6FF'
PURPLE   = '#BC8CFF'
AMBER    = '#FFA657'
TEXT_PRI = '#E6EDF3'
TEXT_SEC = '#8B949E'
GRID_CLR = '#21262D'

# ── Figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor=BG)
gs  = gridspec.GridSpec(5, 3, figure=fig,
                        hspace=0.52, wspace=0.35,
                        top=0.95, bottom=0.03,
                        left=0.06, right=0.97)

# ── Helpers ──────────────────────────────────────────────────────────────────
def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT_SEC, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SEC)
    ax.yaxis.label.set_color(TEXT_SEC)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']:
        ax.spines[sp].set_color(BORDER)
    ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.5, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT_PRI, fontsize=11,
                     fontweight='bold', pad=11, loc='left')
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_SEC, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_SEC, fontsize=9)

def metric_card(ax, value, label, sub='', color=BLUE):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.6)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.60, value, ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    ax.text(0.5, 0.30, label, ha='center', va='center',
            fontsize=11, color=TEXT_PRI)
    if sub:
        ax.text(0.5, 0.13, sub, ha='center', va='center',
                fontsize=9, color=TEXT_SEC)

def legend_patches(labels, colors):
    return [mpatches.Patch(color=c, label=l) for l, c in zip(labels, colors)]

# ═══════════════════════════════════════════════════════════════════════════
# Title bar
# ═══════════════════════════════════════════════════════════════════════════
fig.text(0.06, 0.972, '❤  Heart Disease — Clinical Data Dashboard',
         fontsize=21, fontweight='bold', color=TEXT_PRI, va='top')
fig.text(0.06, 0.958, '302 patients · 14 clinical features · UCI Heart Disease Dataset',
         fontsize=11, color=TEXT_SEC, va='top')

# ═══════════════════════════════════════════════════════════════════════════
# Row 0 — Metric cards
# ═══════════════════════════════════════════════════════════════════════════
disease_rate = df['target'].mean() * 100
mean_age     = df['age'].mean()
mean_chol    = df['chol'].mean()

ax_m1 = fig.add_subplot(gs[0, 0])
metric_card(ax_m1, f'{disease_rate:.1f}%', 'Heart disease rate',
            f'{df["target"].sum()} of {len(df)} patients', POSITIVE)

ax_m2 = fig.add_subplot(gs[0, 1])
metric_card(ax_m2, f'{mean_age:.1f} yrs', 'Mean patient age',
            f'Range: {df["age"].min()}–{df["age"].max()} yrs', AMBER)

ax_m3 = fig.add_subplot(gs[0, 2])
metric_card(ax_m3, f'{mean_chol:.0f}', 'Mean cholesterol (mg/dl)',
            f'Max: {df["chol"].max()} · Min: {df["chol"].min()}', PURPLE)

# ═══════════════════════════════════════════════════════════════════════════
# Row 1 — Age distribution by target + Sex breakdown
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[1, :2])
bins_age = range(29, 80, 3)
d0 = df[df['target'] == 0]['age']
d1 = df[df['target'] == 1]['age']
ax1.hist(d0, bins=bins_age, color=NEGATIVE, alpha=0.75, label='No disease', edgecolor='none')
ax1.hist(d1, bins=bins_age, color=POSITIVE, alpha=0.75, label='Heart disease', edgecolor='none')
ax1.axvline(d0.mean(), color=NEGATIVE, linewidth=1.5, linestyle='--', alpha=0.9)
ax1.axvline(d1.mean(), color=POSITIVE, linewidth=1.5, linestyle='--', alpha=0.9)
style_ax(ax1, 'Age distribution by diagnosis', 'Age (years)', 'Number of patients')
ax1.legend(handles=legend_patches(['No disease','Heart disease'], [NEGATIVE, POSITIVE]),
           frameon=False, labelcolor=TEXT_SEC, fontsize=9, loc='upper left')

ax2 = fig.add_subplot(gs[1, 2])
sex_target = df.groupby(['sex', 'target']).size().unstack(fill_value=0)
sex_labels = ['Female', 'Male']
x = np.arange(2)
w = 0.38
b1 = ax2.bar(x - w/2, sex_target[0].values, w, color=NEGATIVE, edgecolor='none', label='No disease')
b2 = ax2.bar(x + w/2, sex_target[1].values, w, color=POSITIVE, edgecolor='none', label='Heart disease')
for bars in [b1, b2]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 str(int(bar.get_height())), ha='center', fontsize=9, color=TEXT_SEC)
ax2.set_xticks(x); ax2.set_xticklabels(sex_labels, color=TEXT_PRI, fontsize=10)
style_ax(ax2, 'Disease by sex', ylabel='Patients')
ax2.legend(handles=legend_patches(['No disease','Heart disease'], [NEGATIVE, POSITIVE]),
           frameon=False, labelcolor=TEXT_SEC, fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════
# Row 2 — Chest pain type + Max heart rate violin
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2, :2])
cp_order = [0, 1, 2, 3]
cp_labels_x = [cp_map[c] for c in cp_order]
cp_table = pd.crosstab(df['cp'], df['target'])
x = np.arange(len(cp_order))
w = 0.38
b1 = ax3.bar(x - w/2, [cp_table.loc[c, 0] if 0 in cp_table.columns else 0 for c in cp_order],
             w, color=NEGATIVE, edgecolor='none')
b2 = ax3.bar(x + w/2, [cp_table.loc[c, 1] if 1 in cp_table.columns else 0 for c in cp_order],
             w, color=POSITIVE, edgecolor='none')
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, h + 1,
                     str(int(h)), ha='center', fontsize=8.5, color=TEXT_SEC)
ax3.set_xticks(x); ax3.set_xticklabels(cp_labels_x, color=TEXT_PRI, fontsize=9.5)
style_ax(ax3, 'Chest pain type vs diagnosis', ylabel='Patients')
ax3.legend(handles=legend_patches(['No disease','Heart disease'], [NEGATIVE, POSITIVE]),
           frameon=False, labelcolor=TEXT_SEC, fontsize=9)

ax4 = fig.add_subplot(gs[2, 2])
thal_data_0 = df[df['target'] == 0]['thalach']
thal_data_1 = df[df['target'] == 1]['thalach']
vp = ax4.violinplot([thal_data_0, thal_data_1], positions=[0, 1],
                     showmedians=True, showextrema=False)
for i, (body, color) in enumerate(zip(vp['bodies'], [NEGATIVE, POSITIVE])):
    body.set_facecolor(color); body.set_alpha(0.7); body.set_edgecolor('none')
vp['cmedians'].set_color(TEXT_PRI); vp['cmedians'].set_linewidth(2)
ax4.set_xticks([0, 1])
ax4.set_xticklabels(['No disease', 'Heart disease'], color=TEXT_PRI, fontsize=9)
style_ax(ax4, 'Max heart rate\n(thalach)', ylabel='BPM')
for i, (data, color) in enumerate([(thal_data_0, NEGATIVE), (thal_data_1, POSITIVE)]):
    ax4.text(i, data.median() + 2, f'{data.median():.0f}', ha='center',
             fontsize=9, color=color, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════
# Row 3 — Cholesterol scatter + Oldpeak by slope
# ═══════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[3, :2])
for target, color, label in [(0, NEGATIVE, 'No disease'), (1, POSITIVE, 'Heart disease')]:
    sub = df[df['target'] == target]
    ax5.scatter(sub['age'], sub['chol'], c=color, alpha=0.55,
                s=35, edgecolors='none', label=label)
# Reference lines
ax5.axhline(200, color=AMBER, linewidth=1, linestyle=':', alpha=0.8)
ax5.axhline(240, color=POSITIVE, linewidth=1, linestyle=':', alpha=0.8)
ax5.text(76, 202, 'Borderline (200)', color=AMBER, fontsize=8, va='bottom')
ax5.text(76, 242, 'High risk (240)', color=POSITIVE, fontsize=8, va='bottom')
style_ax(ax5, 'Age vs cholesterol by diagnosis',
         'Age (years)', 'Cholesterol (mg/dl)')
ax5.legend(handles=legend_patches(['No disease','Heart disease'], [NEGATIVE, POSITIVE]),
           frameon=False, labelcolor=TEXT_SEC, fontsize=9)

ax6 = fig.add_subplot(gs[3, 2])
slope_labels = ['Upsloping\n(0)', 'Flat\n(1)', 'Downsloping\n(2)']
for s_val in [0, 1, 2]:
    for t_val, color in [(0, NEGATIVE), (1, POSITIVE)]:
        sub = df[(df['slope'] == s_val) & (df['target'] == t_val)]['oldpeak']
        if len(sub):
            ax6.scatter([s_val + (0.1 if t_val else -0.1)] * len(sub),
                        sub, c=color, alpha=0.5, s=25, edgecolors='none')
    for t_val, color in [(0, NEGATIVE), (1, POSITIVE)]:
        sub = df[(df['slope'] == s_val) & (df['target'] == t_val)]['oldpeak']
        if len(sub):
            ax6.plot([s_val + (0.1 if t_val else -0.1)], [sub.mean()],
                     marker='D', markersize=8, color=color,
                     markeredgewidth=1.5, markeredgecolor='white')
ax6.set_xticks([0,1,2]); ax6.set_xticklabels(slope_labels, color=TEXT_PRI, fontsize=8.5)
style_ax(ax6, 'ST depression (oldpeak)\nby slope', ylabel='Oldpeak value')
ax6.legend(handles=legend_patches(['No disease','Heart disease'], [NEGATIVE, POSITIVE]),
           frameon=False, labelcolor=TEXT_SEC, fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════
# Row 4 — Key stats table + Correlation bar chart
# ═══════════════════════════════════════════════════════════════════════════
ax7 = fig.add_subplot(gs[4, :2])
ax7.set_facecolor(CARD)
ax7.axis('off')
ax7.set_title('Mean clinical values by diagnosis', color=TEXT_PRI,
              fontsize=11, fontweight='bold', pad=11, loc='left')

stats = df.groupby('target')[['age','trestbps','chol','thalach','oldpeak']].mean().round(1)
col_labels = ['Diagnosis', 'Age (yrs)', 'Blood pressure', 'Cholesterol', 'Max HR', 'ST depression']
col_vals_0 = ['No disease'] + list(stats.loc[0])
col_vals_1 = ['Heart disease'] + list(stats.loc[1])
col_widths  = [0.22, 0.15, 0.17, 0.15, 0.14, 0.16]

header_y = 0.84; row_h = 0.22; x_start = 0.01

# Header
x = x_start
for label, w in zip(col_labels, col_widths):
    ax7.text(x, header_y, label, color=TEXT_SEC,
             fontsize=9.5, fontweight='bold', va='center', transform=ax7.transAxes)
    x += w

ax7.plot([0.01, 0.99], [header_y - 0.08, header_y - 0.08],
         color=BORDER, linewidth=1, transform=ax7.transAxes, clip_on=False)

for row_i, (vals, row_color, txt_color) in enumerate([
    (col_vals_0, '#0D1F17', NEGATIVE),
    (col_vals_1, '#1F0D0D', POSITIVE)
]):
    y = header_y - 0.12 - row_i * row_h
    rect = FancyBboxPatch((x_start - 0.01, y - 0.09), 0.99, row_h - 0.02,
                          boxstyle='round,pad=0.01', linewidth=0,
                          facecolor=row_color, transform=ax7.transAxes, clip_on=False)
    ax7.add_patch(rect)
    x = x_start
    for vi, (val, w) in enumerate(zip(vals, col_widths)):
        color = txt_color if vi == 0 else TEXT_PRI
        ax7.text(x, y, str(val), color=color,
                 fontsize=9.5, va='center', transform=ax7.transAxes,
                 fontweight='bold' if vi == 0 else 'normal')
        x += w

ax8 = fig.add_subplot(gs[4, 2])
corr = df.select_dtypes(include=['number']).drop(columns='target').corrwith(df['target']).sort_values()
colors_corr = [POSITIVE if v > 0 else NEGATIVE for v in corr]
ax8.barh(range(len(corr)), corr.values, color=colors_corr,
         edgecolor='none', height=0.7)
ax8.axvline(0, color=BORDER, linewidth=1)
ax8.set_yticks(range(len(corr)))
ax8.set_yticklabels(corr.index, color=TEXT_PRI, fontsize=8.5)
ax8.tick_params(axis='x', colors=TEXT_SEC, labelsize=8)
style_ax(ax8, 'Feature correlation\nwith target', xlabel='Pearson r')
ax8.yaxis.grid(False)
ax8.xaxis.grid(True, color=GRID_CLR, linewidth=0.5, linestyle='--')

plt.savefig('/mnt/user-data/outputs/heart_dashboard.png',
            dpi=160, bbox_inches='tight', facecolor=BG)
print("Heart disease dashboard saved!")
