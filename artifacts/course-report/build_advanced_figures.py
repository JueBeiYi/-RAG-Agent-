"""Derive paired/ordinal analyses and render four supplementary journal figures.

Only the frozen report-data.json is read. No model or network is used.
The build_advanced(D, save) entry is invoked by build_figures.py.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from figure_style import heading

BASE = Path(__file__).resolve().parent
NAVY, TEAL, BLUE, OCHRE, CORAL = '#193A56', '#168980', '#5688AD', '#C7923A', '#B85B61'
INK, MUTED, GRID = '#263D4E', '#6E7E8A', '#E1E8EE'
MODES = ['vector', 'hybrid', 'hybrid_rerank']
MODE_NAMES = ['Dense', 'Hybrid', 'Rerank']
CATS = ['factual', 'comparison', 'synthesis', 'reasoning']
CAT_NAMES = ['Factual', 'Comparison', 'Synthesis', 'Reasoning']
METRICS = ['hit_at_5', 'mrr', 'recall_at_5']
QFIELDS = ['correctness', 'completeness', 'citation_quality', 'fluency']
QNAMES = ['Correctness', 'Completeness', 'Citation', 'Fluency']


def average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def derive_statistics(data):
    rows = data['tables']['retrieval_records']['rows']
    by = {m: {r['case_id']: r for r in rows if r['split'] == 'test' and r['mode'] == m} for m in MODES}
    case_ids = sorted(by['vector'])
    assert len(case_ids) == 60
    assert all(set(by[m]) == set(case_ids) for m in MODES)
    assert all(len({by[m][cid]['category'] for m in MODES}) == 1 for cid in case_ids)
    rng = np.random.default_rng(20261017)
    indices = rng.integers(0, 60, size=(10000, 60))
    pairs = [('vector', 'hybrid'), ('hybrid', 'hybrid_rerank'), ('vector', 'hybrid_rerank')]
    effects = []
    for old, new in pairs:
        for metric in METRICS:
            delta = np.array([by[new][cid][metric] - by[old][cid][metric] for cid in case_ids], dtype=float)
            low, high = np.percentile(delta[indices].mean(axis=1), [2.5, 97.5])
            effects.append({'from': old, 'to': new, 'metric': metric, 'n': 60,
                            'mean_delta': float(delta.mean()), 'ci95': [float(low), float(high)],
                            'improved_cases': int((delta > 0).sum()), 'unchanged_cases': int((delta == 0).sum()),
                            'worsened_cases': int((delta < 0).sum())})
    ranks = []
    for category in CATS:
        for cid in case_ids:
            if by['vector'][cid]['category'] != category:
                continue
            ranks.append({'case_id': cid, 'category': category,
                          'rank': {m: by[m][cid]['first_relevant_rank'] for m in MODES},
                          'hit_at_5': {m: by[m][cid]['hit_at_5'] for m in MODES}})
    score_rows = data['tables']['baseline_ai_scores']['rows']
    assert len(score_rows) == 60 and {r['case_id'] for r in score_rows} == set(case_ids)
    scores = np.array([[float(r[q + '_1_to_5']) for q in QFIELDS] for r in score_rows])
    assert np.all((scores >= 1) & (scores <= 5))
    corr = np.corrcoef(np.array([average_ranks(scores[:, i]) for i in range(4)]))
    categories = []
    for cat in CATS:
        rr = [r for r in score_rows if r['category'] == cat]
        categories.append({'category': cat, 'n': len(rr), **{q: float(np.mean([float(r[q + '_1_to_5']) for r in rr])) for q in QFIELDS}})
    condition_rows = data['tables']['quality_records']['rows']
    old = {r['case_id']: r for r in condition_rows if r['study'] == 'prompt' and r['condition'] == 'naive_old_prompt'}
    new = {r['case_id']: r for r in condition_rows if r['study'] == 'prompt' and r['condition'] == 'current_prompt_reused'}
    prompt_ids = ['VT-F01', 'VT-C01', 'VT-S03', 'VT-R01']
    assert set(old) == set(new) == set(prompt_ids)
    assert all(old[cid]['context_hash'] == new[cid]['context_hash'] for cid in prompt_ids)
    prompt = [{'case_id': cid, 'old': {q: old[cid][q] for q in QFIELDS},
               'current': {q: new[cid][q] for q in QFIELDS},
               'delta': {q: new[cid][q] - old[cid][q] for q in QFIELDS}} for cid in prompt_ids]
    return {
        'created_date': '2026-10-17', 'data_sha256': hashlib.sha256((BASE / 'data/report-data.json').read_bytes()).hexdigest(),
        'new_model_calls': 0,
        'paired_bootstrap': {'replicates': 10000, 'seed': 20261017, 'confidence': .95,
                             'case_order': case_ids, 'resampling_unit': 'matched case_id',
                             'method': 'same query-resampling indices across all modes and metrics; percentile interval',
                             'scope': 'conditional query sampling; not repeated-run variance or simultaneous coverage', 'effects': effects},
        'rank_matrix': {'mode_order': MODES, 'category_order': CATS, 'records': ranks,
                        'missing_rank': 'not found within the evaluated top-20 list; never rank zero'},
        'ai_quality': {'n': 60, 'dimensions': QFIELDS, 'spearman_matrix': corr.tolist(),
                       'category_means': categories,
                       'high_fluency_low_correctness': int(((scores[:, 3] >= 4) & (scores[:, 0] <= 2)).sum()),
                       'high_fluency_low_citation': int(((scores[:, 3] >= 4) & (scores[:, 2] <= 2)).sum()),
                       'all_dimensions_at_least_4': int((scores >= 4).all(axis=1).sum())},
        'prompt_pairs': {'n': 4, 'cases': prompt,
                         'mean_changes': {q: float(np.mean([r['delta'][q] for r in prompt])) for q in QFIELDS},
                         'current_answers_reused_from_parameter_baseline': True,
                         'scope': 'four matched dev questions, one answer per condition; exploratory'},
    }


def frame(title, subtitle, height=3.8):
    fig = plt.figure(figsize=(8.1, height), facecolor='white')
    heading(fig, title, subtitle, x=.055)
    return fig


def axis(ax, title, grid='x'):
    ax.set_title(title, loc='left', fontsize=11.6, weight='bold', pad=11, color=NAVY)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=11)
    ax.grid(axis=grid, color=GRID, linewidth=.65)
    for name in ['top', 'right']:
        ax.spines[name].set_visible(False)
    for name in ['left', 'bottom']:
        ax.spines[name].set_color(GRID)


def paired_effects(stats, save):
    fig = frame('Paired effects on evidence retrieval',
                '60 matched test questions | 10,000 query-bootstrap samples | 95% percentile intervals', 3.75)
    gs = fig.add_gridspec(1, 3, left=.215, right=.975, bottom=.19, top=.75, wspace=.32)
    labels = ['Hybrid - Dense', 'Rerank - Hybrid', 'Rerank - Dense']
    pairs = [('vector', 'hybrid'), ('hybrid', 'hybrid_rerank'), ('vector', 'hybrid_rerank')]
    colors = [BLUE, TEAL, CORAL]
    for j, (metric, title) in enumerate(zip(METRICS, ['A   Hit@5', 'B   MRR', 'C   Recall@5'])):
        ax = fig.add_subplot(gs[0, j]); axis(ax, title)
        ax.axvline(0, color=MUTED, linestyle='--', linewidth=1, zorder=1)
        for i, (old, new) in enumerate(pairs):
            e = next(x for x in stats['paired_bootstrap']['effects'] if (x['from'], x['to'], x['metric']) == (old, new, metric))
            mean = e['mean_delta']; lo, hi = e['ci95']
            if i % 2 == 0: ax.axhspan(i-.36, i+.36, color='#F5F8FA', zorder=0)
            ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt='o', color=colors[i],
                        markersize=7, capsize=3.5, elinewidth=1.7, zorder=3)
            ax.annotate(f'{mean:+.3f}', (mean, i), xytext=(0, 13), textcoords='offset points',
                        ha='center', va='bottom', color=colors[i], fontsize=10.5, weight='bold')
        ax.set_yticks(range(3), labels if j == 0 else ['']*3)
        ax.set_ylim(2.55, -.65); ax.set_xlim(-.135, .365)
        ax.set_xticks([-.1, 0, .1, .2, .3]); ax.set_xlabel('Score difference', fontsize=11)
        ax.tick_params(axis='y', length=0)
    fig.text(.055, .045, 'Zero marks no change. Intervals describe query sampling, conditional on the frozen corpus and labels.',
             color=MUTED, fontsize=9.5)
    save(fig, 'fig18_paired_effects', ['retrieval_records'],
         '60题配对查询Bootstrap；同一case_id抽样索引用于所有方法和指标。区间不解释为多次运行方差或总体显著性。',
         stats['paired_bootstrap'])


def rank_matrix(stats, save):
    records = stats['rank_matrix']['records']
    fig = frame('Where evidence is found - a question-level atlas',
                'Same 60 test questions | first relevant rank within top 20 | category-wise ordering', 5.6)
    gs = fig.add_gridspec(2, 2, left=.115, right=.975, bottom=.19, top=.79, hspace=.88, wspace=.30)
    fills = [NAVY, TEAL, '#8BB7CC', '#CEDCE6', '#E7E9EC']
    cmap = ListedColormap(fills)
    for k, cat in enumerate(CATS):
        rr = [r for r in records if r['category'] == cat]
        raw = [[r['rank'][m] for r in rr] for m in MODES]
        binned = np.array([[4 if r is None else (0 if r == 1 else 1 if r <= 5 else 2 if r <= 10 else 3) for r in row] for row in raw])
        ax = fig.add_subplot(gs[k//2, k%2])
        ax.imshow(binned, cmap=cmap, vmin=-.5, vmax=4.5, aspect='auto', interpolation='nearest')
        ax.set_title(f'{chr(65+k)}   {CAT_NAMES[k]} (n={len(rr)})', loc='left', fontsize=11.8,
                     color=NAVY, weight='bold', pad=11)
        ax.set_xticks(range(len(rr)), [r['case_id'].removeprefix('VT-') for r in rr], rotation=90, fontsize=10.4)
        ax.set_yticks(range(3), MODE_NAMES, fontsize=11)
        for i, row in enumerate(raw):
            for j, rank in enumerate(row):
                ax.text(j, i, '-' if rank is None else str(rank), ha='center', va='center',
                        fontsize=10.4, color='white' if binned[i, j] < 2 else INK)
        ax.set_xticks(np.arange(-.5, len(rr), 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=.7); ax.grid(which='major', visible=False)
        ax.tick_params(which='both', length=0)
        for s in ax.spines.values(): s.set_visible(False)
    handles = [Patch(facecolor=c, label=t) for c, t in zip(fills, ['Rank 1', 'Ranks 2-5', 'Ranks 6-10', 'Ranks 11-20', 'Not found'])]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .063), ncol=5,
               frameon=False, fontsize=10.4, handlelength=1.15, columnspacing=1.1)
    fig.text(.055, .028, 'Question IDs omit the VT- prefix. Grey cells are missing evidence, not rank zero.', color=MUTED, fontsize=9.6)
    save(fig, 'fig19_rank_matrix', ['retrieval_records'],
         '60题首个相关证据名次；按题型与题号排序，灰格为Top20内未检到，不能映射为名次0。',
         {'category_order': CATS, 'mode_order': MODES, 'rank_bins': [1, '2-5', '6-10', '11-20', 'missing'],
          'case_order': [r['case_id'] for r in records]})


def quality_association(data, stats, save):
    rr = data['tables']['baseline_ai_scores']['rows']
    fig = frame('Fluent answers still require factual and citation checks',
                '60 frozen short answers | AI-assisted ordinal ratings, 1-5 | counts and rank association', 6.1)
    gs = fig.add_gridspec(2, 2, left=.10, right=.95, bottom=.155, top=.80, hspace=.63, wspace=.42)
    pairs = [('correctness', 'completeness', 'A   Factuality and completeness'),
             ('correctness', 'citation_quality', 'B   Factuality and citations'),
             ('fluency', 'correctness', 'C   Fluency and factuality')]
    for k, (x, y, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[k//2, k%2]); axis(ax, title, grid='both')
        rho = stats['ai_quality']['spearman_matrix'][QFIELDS.index(x)][QFIELDS.index(y)]
        ax.set_title(f'{title}\nSpearman ρ = {rho:.3f}', loc='left', fontsize=10.8,
                     weight='bold', pad=11, color=NAVY, linespacing=1.5)
        counts = Counter((int(r[x+'_1_to_5']), int(r[y+'_1_to_5'])) for r in rr)
        if k == 2:
            ax.add_patch(Rectangle((3.5, .5), 2, 2, facecolor=OCHRE, alpha=.11, zorder=0))
        for (vx, vy), count in sorted(counts.items()):
            ax.scatter(vx, vy, s=count*47, color=[BLUE, TEAL, NAVY][k], alpha=.72, linewidth=.55, edgecolor='white', zorder=3)
            ax.text(vx, vy, str(count), ha='center', va='center', fontsize=10.5,
                    color='white' if count >= 4 else INK, zorder=4)
        ax.set_xlim(.2, 5.8); ax.set_ylim(.2, 5.8); ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
        ax.set_xlabel(QNAMES[QFIELDS.index(x)], fontsize=11); ax.set_ylabel(QNAMES[QFIELDS.index(y)], fontsize=11)
    ax = fig.add_subplot(gs[1, 1]); axis(ax, 'D   Rating association', grid='both')
    corr = np.array(stats['ai_quality']['spearman_matrix'])
    masked = np.ma.array(corr, mask=np.triu(np.ones_like(corr, dtype=bool), 1))
    cmap = LinearSegmentedColormap.from_list('signed_correlation', [CORAL, '#F7F9FA', TEAL]); cmap.set_bad('white')
    image = ax.imshow(masked, cmap=cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), interpolation='nearest')
    ax.set_xticks(range(4), ['Correct.', 'Complete.', 'Citation', 'Fluency'], rotation=38, ha='right', fontsize=10.4)
    ax.set_yticks(range(4), ['Correct.', 'Complete.', 'Citation', 'Fluency'], fontsize=10.4)
    ax.grid(False)
    for i in range(4):
        for j in range(i+1):
            ax.text(j, i, '1.00' if i == j else f'{corr[i,j]:.2f}', ha='center', va='center',
                    fontsize=11, color='white' if corr[i,j] > .65 else INK)
    for s in ax.spines.values():s.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.045, ticks=[-1, 0, 1]); cbar.ax.tick_params(labelsize=10)
    cbar.outline.set_visible(False)
    fig.text(.055, .055, 'Bubble area encodes count. High fluency (≥4) with low correctness (≤2): 13/60; low citation score: 22/60.',
             color=MUTED, fontsize=9.4)
    fig.text(.055, .023, 'Associations describe one AI-assisted rating set; they do not establish causation or independent human agreement.',
             color=MUTED, fontsize=9.4)
    save(fig, 'fig20_quality_association', ['baseline_ai_scores'],
         '离散评分计数气泡和Spearman秩相关；不平滑有序整数评分，不推断因果或独立人工一致性。', stats['ai_quality'])


def prompt_pairs(stats, save):
    fig = frame('Prompt improvements are not uniform across questions',
                'Four matched dev questions | identical frozen contexts | one answer per condition', 3.9)
    gs = fig.add_gridspec(1, 4, left=.14, right=.98, top=.71, bottom=.19, wspace=.34)
    rr = stats['prompt_pairs']['cases']
    for j, (field, title) in enumerate(zip(QFIELDS, QNAMES)):
        ax = fig.add_subplot(gs[0, j]); axis(ax, f'{chr(65+j)}   {title}')
        ax.set_title(f'{chr(65+j)}   {title}\nMean Δ = {stats["prompt_pairs"]["mean_changes"][field]:+.2f}',
                     loc='left', fontsize=10.8, weight='bold', pad=11, color=NAVY, linespacing=1.5)
        for i, row in enumerate(rr):
            if i % 2 == 0:ax.axhspan(i-.36, i+.36, color='#F3F7F9', zorder=0)
            old, new = row['old'][field], row['current'][field]
            ax.plot([old, new], [i, i], color=MUTED, linewidth=1.6, zorder=2)
            ax.scatter(old, i, facecolors='white', edgecolors=OCHRE, s=79, linewidths=1.6, zorder=3)
            ax.scatter(new, i, facecolors=TEAL, edgecolors='white', s=39, linewidths=.5, zorder=4)
        ax.set_xlim(.55, 5.45); ax.set_xticks(range(1, 6)); ax.set_yticks(range(4), [r['case_id'] for r in rr] if j == 0 else ['']*4)
        ax.set_ylim(3.55, -.6); ax.tick_params(axis='y', length=0); ax.set_xlabel('AI rating (1-5)', fontsize=10.6)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor='white', markeredgecolor=OCHRE,
                      markersize=7.5, label='Simplified prompt'),
               Line2D([0], [0], marker='o', color='none', markerfacecolor=TEAL, markersize=6, label='Evidence-constrained prompt')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .023), frameon=False, ncol=2, fontsize=10.5)
    save(fig, 'fig21_prompt_pairs', ['quality_records'],
         '四题新旧Prompt逐题配对；当前Prompt复用参数基线，不增加模型调用，均值提升并不意味着每题均改善。', stats['prompt_pairs'])


def build_advanced(data, save):
    stats = derive_statistics(data)
    (BASE/'data/advanced-statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    paired_effects(stats, save)
    rank_matrix(stats, save)
    quality_association(data, stats, save)
    prompt_pairs(stats, save)
    print('advanced analyses done: 4 figures; no new model calls', flush=True)


if __name__ == '__main__':
    data = json.loads((BASE/'data/report-data.json').read_text(encoding='utf-8'))
    stats = derive_statistics(data)
    (BASE/'data/advanced-statistics.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Derived statistics saved; use build_figures.py to render every figure.')
