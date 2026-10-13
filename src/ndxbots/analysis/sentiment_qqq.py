from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ndxbots.config import load_settings
from ndxbots.regime.compute import compute_regime
from ndxbots.regime.exposure import exposure_by_date


TARGETS = {
    "qqq_close": "QQQ價格水準（僅描述）",
    "qqq_ret_1d": "當日報酬",
    "qqq_ret_5d": "過去5日報酬",
    "qqq_ret_21d": "過去21日報酬",
    "qqq_forward_1d": "未來1日報酬",
    "qqq_forward_5d": "未來5日報酬",
    "qqq_forward_21d": "未來21日報酬",
}


def relationship_table(regime, qqq, settings, start="2025-01-01", end=None, baseline=None):
    """Align T-close scores with descriptive and future QQQ returns.

    Returns are calculated on the full QQQ calendar before the display window
    is cropped. Invalid supplied QQQ prices are rejected, never interpolated;
    this function does not infer absent sessions without an external calendar.
    Future observations only construct research targets; they never feed scores.
    """
    qqq = qqq[["date", "close"]].copy()
    qqq["date"] = pd.to_datetime(qqq["date"]).dt.normalize()
    if qqq.date.isna().any() or qqq.duplicated("date").any():
        raise ValueError("QQQ requires unique, valid trading dates")
    qqq = qqq.sort_values("date").set_index("date")["close"]
    if not (np.isfinite(qqq) & qqq.gt(0)).all():
        raise ValueError("QQQ close must be finite and positive")
    table = exposure_by_date(regime, settings, dates=qqq.index).set_index("date")
    table["qqq_close"] = qqq
    for horizon in (1, 5, 21):
        table[f"qqq_ret_{horizon}d"] = qqq / qqq.shift(horizon) - 1
        table[f"qqq_forward_{horizon}d"] = qqq.shift(-horizon) / qqq - 1
    table["sentiment_trade_delta"] = table["sentiment_trade"].diff()
    if baseline is not None:
        original = baseline.copy()
        original["date"] = pd.to_datetime(original["date"]).dt.normalize()
        if original.duplicated("date").any():
            raise ValueError("Baseline requires unique dates")
        original = original.set_index("date").reindex(table.index)
        for column in ("sentiment", "sentiment_trade", "sentiment_trade_smooth", "gross_target"):
            if column in original:
                table[f"baseline_{column}"] = original[column]
    table["rolling60_delta_return_corr"] = table.sentiment_trade_delta.rolling(60, min_periods=40).corr(table.qqq_ret_1d)
    start = pd.Timestamp(start)
    end = qqq.index.max() if end is None else min(pd.Timestamp(end), qqq.index.max())
    return table.loc[(table.index >= start) & (table.index <= end)].reset_index()


def correlations(table):
    """Pairwise Pearson and rank-based Spearman, with exact sample counts."""
    scores = ["sentiment", "sentiment_trade", "sentiment_trade_smooth", "sentiment_trade_delta",
              "baseline_sentiment", "baseline_sentiment_trade", "baseline_sentiment_trade_smooth"]
    rows = []
    for score in scores:
        if score not in table:
            continue
        for target, label in TARGETS.items():
            pair = table[[score, target]].replace([np.inf, -np.inf], np.nan).dropna()
            n = len(pair)
            usable = n >= 3 and pair[score].nunique() > 1 and pair[target].nunique() > 1
            rows.append({"score": score, "target": target, "target_label": label, "n": n,
                         "pearson": pair[score].corr(pair[target]) if usable else np.nan,
                         "spearman": pair[score].rank().corr(pair[target].rank()) if usable else np.nan})
    return pd.DataFrame(rows)


def _plot_setup():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Charts require pip install -e ".[research]"') from exc
    from matplotlib import font_manager
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    family = next((f for f in ("Microsoft YaHei", "Noto Sans CJK TC", "SimHei") if f in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": family, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white"})
    return plt


def render_overlay(table, output, settings=None):
    """Overlay preserved QQQ and sentiment values on separately labelled axes."""
    if table.empty:
        raise ValueError("Overlay interval contains no QQQ sessions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = settings or load_settings()
    plt = _plot_setup()
    dates = pd.to_datetime(table.date)
    normalized = table.qqq_close / table.qqq_close.iloc[0] * 100
    fig, price = plt.subplots(figsize=(16, 8))
    sentiment = price.twinx()
    handles = []
    handles += price.plot(dates, normalized, color="#2451a4", lw=2.5,
                          label="QQQ前復權價格（首日=100）")
    if "baseline_sentiment_trade_smooth" in table:
        handles += sentiment.plot(dates, table.baseline_sentiment_trade_smooth, color="#a4a4a4", lw=1.1,
                                  alpha=.9, label="修改前交易情緒（5日）")
    handles += sentiment.plot(dates, table.sentiment, color="#8bbbd9", lw=.9, alpha=.55,
                              label="新版展示情緒（含成交量）")
    handles += sentiment.plot(dates, table.sentiment_trade_smooth, color="#07847d", lw=1.9,
                              label=f"新版交易情緒（{settings.gross_smooth}日）")
    handles.append(sentiment.axhline(float(settings.dual_hot), color="#c24d3d", ls="--", lw=.9,
                                    alpha=.8, label=f"過熱參考{settings.dual_hot:g}"))
    handles.append(sentiment.axhline(float(settings.gross_ladder_start), color="#c38a2b", ls="--", lw=.9,
                                    alpha=.8, label=f"降倉起點{settings.gross_ladder_start:g}"))
    price.set_ylabel("QQQ價格指數（首日=100）", color="#2451a4", labelpad=12)
    sentiment.set_ylabel("情緒分數（0—1）", color="#07847d", labelpad=12)
    price.tick_params(axis="y", colors="#2451a4")
    sentiment.tick_params(axis="y", colors="#07847d")
    price.set_ylim(normalized.min() - 7, normalized.max() + 7)
    sentiment.set_ylim(0, 1)
    price.set_xlim(dates.min(), dates.max())
    sentiment.spines["right"].set_visible(True)
    price.set_title(f"QQQ價格與情緒疊圖  |  {dates.min():%Y-%m-%d} — {dates.max():%Y-%m-%d}",
                    loc="left", fontsize=21, fontweight="bold", pad=70)
    price.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
                 frameon=False, fontsize=10, columnspacing=2.2)
    fig.text(.075, .025, f"共{len(table)}個交易日 · 左軸QQQ、右軸情緒，各自使用不同單位與尺度。",
             color="#5b6573", fontsize=10)
    fig.tight_layout(rect=(.01, .055, .99, .97))
    fig.savefig(output / "sentiment_qqq_overlay.png", dpi=180)
    fig.savefig(output / "sentiment_qqq_overlay.svg")
    plt.close(fig)


def render_report(table, output, metadata=None, settings=None):
    if table.empty:
        raise ValueError("Requested chart interval contains no QQQ sessions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "sentiment_qqq_daily.csv", index=False, encoding="utf-8-sig")
    summary = correlations(table)
    summary.to_csv(output / "correlations.csv", index=False, encoding="utf-8-sig")
    plt = _plot_setup()
    settings = settings or load_settings()
    hot = float(settings.dual_hot)
    ladder_start = float(settings.gross_ladder_start)
    dates = pd.to_datetime(table.date)
    period = f"{dates.min():%Y-%m-%d} — {dates.max():%Y-%m-%d}"
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [2, 2.3, 1.2, 1.5]})
    axes[0].plot(dates, table.qqq_close / table.qqq_close.iloc[0] * 100, color="#2451a4", lw=2)
    axes[0].set_ylabel("QQQ前復權價格\n首日=100")
    axes[0].set_title(f"情緒與QQQ對照  |  {period}", loc="left", fontsize=18, pad=16, fontweight="bold")
    if "baseline_sentiment_trade_smooth" in table:
        axes[1].plot(dates, table.baseline_sentiment_trade_smooth, color="#a5a5a5", lw=1.2, label="修改前交易情緒（5日）")
    axes[1].plot(dates, table.sentiment, color="#91bad5", lw=.8, alpha=.7, label="新版展示情緒（含成交量）")
    axes[1].plot(dates, table.sentiment_trade_smooth, color="#087c76", lw=1.8, label="新版交易情緒（5日）")
    axes[1].axhline(hot, color="#b94737", ls="--", lw=.8, label=f"過熱參考{hot:g}")
    axes[1].axhline(ladder_start, color="#b88b3a", ls="--", lw=.8, label=f"降倉起點{ladder_start:g}")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("情緒分數")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)
    axes[2].step(dates, table.gross_target, where="post", color="#774598", lw=1.5, label="情緒層毛曝險目標")
    fallback = table.gross_data_fallback.fillna(True)
    axes[2].fill_between(dates, 0, 1.05, where=fallback.to_numpy(bool), color="#e9a24d", alpha=.25, label="資料降級")
    axes[2].set_ylim(.45, 1.05)
    axes[2].set_ylabel("情緒層目標\n未含Hurst縮放")
    axes[2].legend(loc="lower left", ncol=2, fontsize=8)
    axes[3].plot(dates, table.rolling60_delta_return_corr, color="#495566", lw=1.4)
    axes[3].axhline(0, color="#777", lw=.7)
    axes[3].set_ylim(-1, 1)
    axes[3].set_ylabel("60日滾動Pearson\nΔ情緒 vs 當日QQQ報酬")
    fig.text(.075, .015, "T日收盤情緒；未來報酬僅作研究標籤。前復權價格不等同完整股息再投資回報。", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(output / "sentiment_qqq_timeline.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, target in zip(axes.flat, ("qqq_ret_1d", "qqq_forward_1d", "qqq_forward_5d", "qqq_forward_21d")):
        score = "sentiment_trade_smooth"
        pair = table[[score, target]].dropna()
        row = summary.loc[summary.score.eq(score) & summary.target.eq(target)].iloc[0]
        ax.scatter(pair[score], pair[target] * 100, s=13, color="#087c76", alpha=.35, edgecolors="none")
        if len(pair) >= 3 and pair[score].nunique() > 1:
            fit = np.polyfit(pair[score], pair[target] * 100, 1)
            x = np.linspace(pair[score].min(), pair[score].max(), 100)
            ax.plot(x, np.polyval(fit, x), color="#b94737", lw=1.6)
        ax.axhline(0, color="#888", lw=.7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("T日交易情緒（5日平滑）")
        ax.set_ylabel(f"{TARGETS[target]}（%）")
        ax.set_title(f"{TARGETS[target]}  |  Pearson={row.pearson:.3f}  Spearman={row.spearman:.3f}  n={row.n}", fontsize=11)
    fig.suptitle(f"情緒與QQQ報酬散點圖  |  {period}", fontsize=18, fontweight="bold")
    fig.text(.06, .015, "5/21日報酬窗口互相重疊，觀測並非獨立；相關性不等同預測能力或因果。", fontsize=9, color="#555")
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(output / "sentiment_qqq_scatter.png", dpi=170)
    plt.close(fig)
    render_overlay(table, output, settings=settings)

    meta = dict(metadata or {})
    meta.update({"chart_start": str(dates.min().date()), "chart_end": str(dates.max().date()), "sessions": len(table),
                 "valid_trade_score_sessions": int(table.sentiment_trade.notna().sum()),
                 "exposure_fallback_sessions": int(table.gross_data_fallback.sum()),
                 "min_membership_coverage": float(table.membership_coverage.min()) if "membership_coverage" in table else None,
                 "latest_qqq_close": float(table.qqq_close.iloc[-1]),
                 "dual_hot": hot, "gross_ladder_start": ladder_start,
                 "return_definition": "QFQ close ratios, not a complete total-return dividend reinvestment model",
                 "sample_status": "Descriptive in-sample analysis, not independent out-of-sample validation"})
    (output / "source_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    selected = summary.loc[summary.score.isin(["sentiment_trade_smooth", "baseline_sentiment_trade_smooth"])].copy()
    selected = selected.rename(columns={"score": "情緒版本", "target_label": "QQQ對照", "n": "樣本數", "pearson": "Pearson", "spearman": "Spearman"}).drop(columns="target")
    table_html = selected.to_html(index=False, float_format=lambda x: f"{x:.3f}", border=0)
    images = "".join(f'<img alt="{name}" src="data:image/png;base64,{base64.b64encode((output / name).read_bytes()).decode()}">' for name in ("sentiment_qqq_overlay.png", "sentiment_qqq_timeline.png", "sentiment_qqq_scatter.png"))
    notes = ("當日、過去5/21日的相關主要描述同步關係；未來1/5/21日才是領先關係檢查。價格水準相關容易受趨勢影響。"
             "未來報酬缺乏完整觀察窗口的尾端日期自動排除，每项保留實際樣本數。5/21日窗口重疊，不能將每列當成獨立樣本。"
             "本圖沒有優化權重或閾值，也不是獨立樣本外驗證。FINRA若沒有實際發布日期，仍使用明確標記的月末＋21天估計；歷史修訂版本未知。"
             "成分覆蓋率以倉庫歷史成分表為準，個別成分資料缺失及供應商復權修訂仍可能影響數值。")
    document = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>情緒與QQQ相關性</title>
<style>body{{margin:0;background:#f0f3f7;color:#17283a;font-family:"Microsoft YaHei",sans-serif}}main{{max-width:1200px;margin:35px auto;padding:24px}}h1{{font-size:30px}}p{{line-height:1.8}}.card{{background:white;border-radius:12px;padding:24px;margin:22px 0;overflow:auto}}img{{width:100%;height:auto;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:10px;border-bottom:1px solid #e6eaf0;text-align:left}}a{{color:#2451a4}}</style>
<main><h1>情緒指標 × QQQ</h1><p>{period} · {len(table)} 個交易日 · 有效交易情緒 {meta['valid_trade_score_sessions']} 日 · 情緒層資料降級 {meta['exposure_fallback_sessions']} 日</p>
<div class="card"><p>時間圖分開呈現QQQ、情緒與倉位，散點圖檢查報酬關係。灰線是原版算法在同一份新行情上的結果，避免把資料修訂誤認為程式改良效果。倉位圖只表示情緒層目標，完整策略另有Hurst、選股及帳戶限制。</p>{images}</div>
<div class="card"><h2>修改前後相關係數</h2>{table_html}<p><a href="sentiment_qqq_daily.csv">下載每日對齊資料</a> · <a href="correlations.csv">全部相關係數</a> · <a href="source_metadata.json">資料來源與品質</a></p></div>
<div class="card"><h2>讀圖方式</h2><p>{html.escape(notes)}</p></div></main></html>'''
    (output / "sentiment_qqq_report.html").write_text(document, encoding="utf-8")
    return meta, summary


def main():
    parser = argparse.ArgumentParser(description="Generate causal sentiment vs QQQ research charts")
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    settings = load_settings()
    panel = pd.read_parquet(args.panel)
    regime = compute_regime(panel, settings)
    baseline = pd.read_csv(args.baseline) if args.baseline else None
    table = relationship_table(regime, panel.loc[panel.code.eq(settings.benchmark)], settings,
                               args.start, args.end, baseline)
    meta, _ = render_report(table, args.output, {"panel": str(args.panel.resolve())}, settings=settings)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
