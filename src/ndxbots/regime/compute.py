from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ndxbots.config import Settings, load_settings
from ndxbots.data.ingest import panel_path
from ndxbots.factors.price import rsi
from ndxbots.regime.external import load_external
from ndxbots.regime.leverage import load_margin_debt, refresh_margin_debt, yoy_on_trading_days
from ndxbots.regime.membership import member_mask
from ndxbots.regime.stats import classify_state, compound_return, rolling_percentile


def regime_path(settings: Settings) -> Path:
    return settings.data_root / "regime" / "daily.parquet"


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame | None:
    if col not in panel.columns:
        return None
    out = panel.pivot(index="date", columns="code", values=col).sort_index()
    out.index = pd.to_datetime(out.index).normalize()
    return out


def _breadth(close, volume, members, settings):
    member = members.reindex(index=close.index, columns=close.columns).fillna(False)
    alive = member & close.notna()
    ma = close.rolling(settings.regime_ma_window, min_periods=settings.regime_ma_window).mean()
    ma_ok = alive & ma.notna()
    above = (close > ma) & ma_ok
    raw1 = above.sum(axis=1) / ma_ok.sum(axis=1).replace(0, np.nan)
    roll_high = close.rolling(settings.regime_nhnl_window, min_periods=settings.regime_nhnl_window).max()
    roll_low = close.rolling(settings.regime_nhnl_window, min_periods=settings.regime_nhnl_window).min()
    nhnl_ok = alive & roll_high.notna() & roll_low.notna()
    nh = (close >= roll_high) & nhnl_ok
    nl = (close <= roll_low) & nhnl_ok
    raw2 = (nh.sum(axis=1) - nl.sum(axis=1)) / nhnl_ok.sum(axis=1).replace(0, np.nan)
    ret = close.pct_change(fill_method=None)
    if volume is None:
        raw3 = pd.Series(np.nan, index=close.index)
    else:
        vol = volume.reindex(index=close.index, columns=close.columns)
        up = ((ret > 0) & alive & vol.notna()).astype(float) * vol
        down = ((ret < 0) & alive & vol.notna()).astype(float) * vol
        raw3 = up.sum(axis=1) / (up.sum(axis=1) + down.sum(axis=1)).replace(0, np.nan)
    rsi14 = rsi(close, settings.regime_rsi_window)
    rsi_ok = alive & rsi14.notna()
    raw4 = rsi14.where(rsi_ok).median(axis=1, skipna=True)
    raw4 = raw4.where(rsi_ok.sum(axis=1) > 0)
    n_valid = alive.sum(axis=1)
    thin = n_valid < settings.regime_min_names
    return pd.DataFrame({"n_members": n_valid, "breadth_ma50": raw1.mask(thin), "breadth_nhnl": raw2.mask(thin), "breadth_upvol": raw3.mask(thin), "breadth_rsi": raw4.mask(thin)})


def _crowding_raw(close, amount, members, benchmark, settings):
    member = members.reindex(index=close.index, columns=close.columns).fillna(False)
    if benchmark in member.columns:
        member[benchmark] = False
    ret = close.pct_change(fill_method=None)
    eligible = member & ret.notna()
    ew = ret.where(eligible).mean(axis=1, skipna=True)
    ew = ew.where(eligible.sum(axis=1) >= settings.regime_min_names)
    ew60 = compound_return(ew, settings.regime_ew_window)
    qqq60 = close[benchmark] / close[benchmark].shift(settings.regime_ew_window) - 1 if benchmark in close.columns else pd.Series(np.nan, index=close.index)
    crowd_index = qqq60 - ew60
    if amount is None:
        crowd_trade = pd.Series(np.nan, index=close.index)
    else:
        use = amount.reindex(index=close.index, columns=close.columns)
        use = use.where(member & use.notna() & (use > 0))

        def _top_share(row):
            valid = row.dropna()
            if len(valid) < settings.regime_min_names or float(valid.sum()) <= 0:
                return np.nan
            return float(valid.nlargest(settings.regime_top_amount).sum() / valid.sum())

        crowd_trade = use.apply(_top_share, axis=1)
    return pd.DataFrame({"qqq_ret_60": qqq60, "ew_ret_60": ew60, "crowd_index_raw": crowd_index, "crowd_trade_raw": crowd_trade}, index=close.index)


def _zone(score):
    out = pd.Series("不足", index=score.index, dtype="object")
    ok = score.notna()
    out = out.mask(ok & score.ge(80), "極度貪婪")
    out = out.mask(ok & score.ge(60) & score.lt(80), "貪婪")
    out = out.mask(ok & score.ge(40) & score.lt(60), "中性")
    out = out.mask(ok & score.ge(20) & score.lt(40), "恐慌")
    out = out.mask(ok & score.lt(20), "極度恐慌")
    return out


def _overlay(close, breadth, scores, external, settings):
    index = close.index
    bench = close[settings.benchmark] if settings.benchmark in close.columns else pd.Series(np.nan, index=index)
    ext = external.reindex(index).ffill() if not external.empty else pd.DataFrame(index=index)
    window = settings.regime_window
    min_p = settings.regime_min_periods
    vix = ext["vix"] if "vix" in ext.columns else pd.Series(np.nan, index=index)
    hyg = ext["hyg"] if "hyg" in ext.columns else pd.Series(np.nan, index=index)
    lqd = ext["lqd"] if "lqd" in ext.columns else pd.Series(np.nan, index=index)
    tlt = ext["tlt"] if "tlt" in ext.columns else pd.Series(np.nan, index=index)
    breadth_level = scores[["pct_ma50", "pct_nhnl", "pct_rsi"]].mean(axis=1, skipna=False)
    level = pd.DataFrame({
        "level_breadth": breadth_level,
        "level_leverage": scores["leverage_pct"],
        "level_credit": rolling_percentile(hyg.pct_change(60) - lqd.pct_change(60), window, min_p),
        "level_momentum": rolling_percentile(bench / bench.rolling(200, min_periods=200).mean() - 1, window, min_p),
        "level_vix": 1 - rolling_percentile(vix, window, min_p),
    })
    smooth = int(getattr(settings, "sentiment_smooth", 5))
    sentiment = level.mean(axis=1, skipna=False).rolling(smooth, min_periods=smooth).mean()
    fast = pd.DataFrame({
        "fast_vix": (1 - rolling_percentile(vix, 20, 15)) * 100,
        "fast_breadth": rolling_percentile(breadth["breadth_ma50"] - breadth["breadth_ma50"].shift(5), window, min_p) * 100,
        "fast_haven": rolling_percentile(bench.pct_change(5) - tlt.pct_change(5), window, min_p) * 100,
        "fast_mom": rolling_percentile(bench / bench.rolling(20, min_periods=20).mean() - 1, window, min_p) * 100,
    })
    fast_mean = fast.mean(axis=1, skipna=False)
    z_window = int(getattr(settings, "z_window", window))
    z = (fast_mean - fast_mean.rolling(z_window, min_periods=min_p).mean()) / fast_mean.rolling(z_window, min_periods=min_p).std()
    out = pd.concat([level, fast], axis=1)
    out["sentiment"] = sentiment
    out["sentiment_100"] = sentiment * 100
    out["sentiment_zone"] = _zone(out["sentiment_100"])
    out["sentiment_fast"] = fast_mean
    out["sentiment_z"] = z
    out["sentiment_degraded"] = level.isna().any(axis=1) | fast.isna().any(axis=1)
    return out


def compute_regime(panel, settings):
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    close = _wide(panel, "close")
    if close is None or close.empty:
        raise ValueError("面板沒有 close")
    volume = _wide(panel, "volume")
    turnover = _wide(panel, "turnover")
    amount = turnover if turnover is not None else (close * volume if volume is not None else None)
    members, source = member_mask(close.index, list(close.columns), settings, settings.benchmark)
    breadth = _breadth(close, volume, members, settings)
    crowd = _crowding_raw(close, amount, members, settings.benchmark, settings)
    monthly, lev_source = load_margin_debt(settings)
    yoy = yoy_on_trading_days(monthly, close.index, settings.regime_leverage_lag_days)
    window = settings.regime_window
    min_p = settings.regime_min_periods
    scores = pd.DataFrame({
        "pct_ma50": rolling_percentile(breadth["breadth_ma50"], window, min_p),
        "pct_nhnl": rolling_percentile(breadth["breadth_nhnl"], window, min_p),
        "pct_upvol": rolling_percentile(breadth["breadth_upvol"], window, min_p),
        "pct_rsi": rolling_percentile(breadth["breadth_rsi"], window, min_p),
        "leverage_pct": rolling_percentile(yoy, window, min_p),
        "pct_crowd_index": rolling_percentile(crowd["crowd_index_raw"], window, min_p),
        "pct_crowd_trade": rolling_percentile(crowd["crowd_trade_raw"], window, min_p),
    }, index=close.index)
    scores["breadth_score"] = scores[["pct_ma50", "pct_nhnl", "pct_upvol", "pct_rsi"]].mean(axis=1, skipna=False)
    external = load_external(settings, panel)
    overlay = _overlay(close, breadth, scores, external, settings)
    scores["sentiment_legacy"] = scores["breadth_score"] * settings.regime_breadth_weight + scores["leverage_pct"] * settings.regime_leverage_weight
    missing_lev = scores["leverage_pct"].isna() & scores["breadth_score"].notna()
    scores.loc[missing_lev, "sentiment_legacy"] = scores.loc[missing_lev, "breadth_score"]
    scores["sentiment"] = overlay["sentiment"]
    scores["crowding"] = scores[["pct_crowd_index", "pct_crowd_trade", "leverage_pct"]].mean(axis=1, skipna=False)
    missing_crowd = scores["leverage_pct"].isna() & scores[["pct_crowd_index", "pct_crowd_trade"]].notna().all(axis=1)
    scores.loc[missing_crowd, "crowding"] = scores.loc[missing_crowd, ["pct_crowd_index", "pct_crowd_trade"]].mean(axis=1)
    qqq = close[settings.benchmark] if settings.benchmark in close.columns else pd.Series(np.nan, index=close.index)
    divergence = (qqq >= qqq.rolling(window, min_periods=min_p).max()) & scores["sentiment"].notna() & (scores["sentiment"] < settings.regime_hot)
    out = pd.concat([breadth, crowd, scores, overlay.drop(columns=["sentiment"])], axis=1)
    out["leverage_yoy"] = yoy
    out["qqq_close"] = qqq
    out["qqq_sentiment_divergence"] = divergence.fillna(False)
    out["membership_source"] = source
    out["leverage_source"] = lev_source or "missing"
    out["external_source"] = "panel+yahoo" if not external.empty else "missing"
    out["state"] = [classify_state(s, c, settings.regime_hot, settings.regime_cold, settings.regime_split) for s, c in zip(out["sentiment"], out["crowding"], strict=True)]
    out.index.name = "date"
    return out.reset_index()


def run_regime(refresh_leverage: bool = False):
    settings = load_settings()
    path = panel_path(settings)
    if not path.exists():
        raise SystemExit(f"還沒有行情面板: {path}\n請先運行 python -m ndxbots.data.ingest")
    if refresh_leverage:
        print(f"已更新融資盤 {refresh_margin_debt(settings)}")
    panel = pd.read_parquet(path)
    table = compute_regime(panel, settings)
    out = regime_path(settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    latest = table.dropna(subset=["sentiment"]).tail(1)
    print(f"狀態表已寫 {out}")
    print(f"行數={len(table)}  日期 {table['date'].min().date()} → {table['date'].max().date()}")
    print(f"成分來源: {table['membership_source'].iloc[-1]}")
    print(f"融資盤來源: {table['leverage_source'].iloc[-1]}")
    print(f"外部序列: {table['external_source'].iloc[-1]}")
    if latest.empty:
        print("有效情緒分數還沒出來。")
        return table
    row = latest.iloc[0]
    z = row.get("sentiment_z")
    z_txt = f"{z:+.2f}" if pd.notna(z) else "nan"
    print(f"最近有效日 {pd.Timestamp(row['date']).date()}  情緒={row['sentiment']:.3f}  區={row.get('sentiment_zone')}  Z={z_txt}  擁擠={row['crowding']:.3f}")
    print(f"狀態: {row['state']}")
    if bool(row.get("sentiment_degraded", False)):
        print("注意: VIX/HYG/LQD/TLT 有缺，當天水平或 Z 不是完整五項/四項。")
    if bool(row["qqq_sentiment_divergence"]):
        print("背離: QQQ 創 252 日新高，但情緒未進入過熱區。")
    if str(row["membership_source"]) == "static_snapshot":
        print("注意: 沒有歷史成分表，廣度/擁擠用的是當前名單。")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="計算納指100情緒、五區和 Z-score 變化")
    parser.add_argument("--refresh-leverage", action="store_true")
    args = parser.parse_args()
    run_regime(refresh_leverage=args.refresh_leverage)


if __name__ == "__main__":
    main()
