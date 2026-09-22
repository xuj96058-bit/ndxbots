from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ndxbots.config import Settings, load_settings
from ndxbots.data.calendar import save_calendar, trading_days_from_benchmark
from ndxbots.data.futu_client import FutuClient, FutuError
from ndxbots.data.universe import resolve_universe


def raw_dir(settings: Settings) -> Path:
    return settings.data_root / "raw" / "kline"


def kline_path(settings: Settings, code: str) -> Path:
    safe = code.replace(".", "_")
    return raw_dir(settings) / f"{safe}.parquet"


def universe_path(settings: Settings) -> Path:
    return settings.data_root / "meta" / "universe.csv"


def panel_path(settings: Settings) -> Path:
    return settings.data_root / "panel" / "daily.parquet"


def calendar_path(settings: Settings) -> Path:
    return settings.data_root / "meta" / "us_trading_days.parquet"


def _existing_last_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["date"])
    if df.empty:
        return None
    return pd.to_datetime(df["date"]).max()


def _merge_kline(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        out = new
    else:
        out = pd.concat([old, new], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    return out.reset_index(drop=True)


def _clip_kline(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """把本地 K 线裁到 [kline_start, kline_end] 闭区间。end 为 None 时只裁起点。"""
    if df is None or df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    start = pd.Timestamp(settings.kline_start)
    out = out[out["date"] >= start]
    if settings.kline_end:
        out = out[out["date"] <= pd.Timestamp(settings.kline_end)]
    return out.reset_index(drop=True)


def ingest_one(
    client: FutuClient,
    code: str,
    settings: Settings,
    full_refresh: bool = False,
) -> tuple[int, str]:
    path = kline_path(settings, code)
    start = settings.kline_start
    end = settings.kline_end
    old = None

    if path.exists() and not full_refresh:
        old = pd.read_parquet(path)
        last = _existing_last_date(path)
        if last is not None:
            start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            # 本地已经覆盖到（或超过）结束日，不必再请求
            if end is not None and last.normalize() >= pd.Timestamp(end):
                clipped = _clip_kline(old, settings)
                if len(clipped) != len(old):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    clipped.to_parquet(path, index=False)
                return len(clipped), "already_up_to_date"
            today = pd.Timestamp.today().normalize()
            if pd.Timestamp(start) > today:
                return len(old), "already_up_to_date"
            if end is not None and pd.Timestamp(start) > pd.Timestamp(end):
                clipped = _clip_kline(old, settings)
                if len(clipped) != len(old):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    clipped.to_parquet(path, index=False)
                return len(clipped), "already_up_to_date"

    try:
        new = client.history_kline(code, start=start, end=end)
    except FutuError as exc:
        return 0, f"error: {exc}"

    if new.empty and old is None:
        return 0, "empty"
    merged = _clip_kline(_merge_kline(old, new), settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    added = 0 if new.empty else len(new)
    return len(merged), f"ok added={added}"


def build_panel(settings: Settings, codes: list[str]) -> pd.DataFrame:
    frames = []
    for code in codes:
        path = kline_path(settings, code)
        if not path.exists():
            continue
        df = _clip_kline(pd.read_parquet(path), settings)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["date", "code"]).reset_index(drop=True)
    panel_path(settings).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path(settings), index=False)
    return panel


def run_ingest(
    full_refresh: bool = False,
    limit: int | None = None,
    codes: list[str] | None = None,
) -> None:
    settings = load_settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)

    client = FutuClient(settings)
    with client.session():
        if codes:
            universe = pd.DataFrame({"code": codes, "source": "cli"})
        else:
            universe = resolve_universe(
                client=client,
                settings=settings,
                save_to=universe_path(settings),
            )
        work = universe["code"].tolist()
        if limit:
            work = work[:limit]

        print(f"成分来源: {universe['source'].iloc[0] if 'source' in universe else 'cli'}")
        end_label = settings.kline_end or "latest"
        print(f"待下载: {len(work)} 只  区间={settings.kline_start} → {end_label}")
        print(f"数据目录: {settings.data_root}")

        ok, fail = 0, 0
        for i, code in enumerate(work, 1):
            n, msg = ingest_one(client, code, settings, full_refresh=full_refresh)
            flag = "OK" if msg.startswith("ok") or msg == "already_up_to_date" else "FAIL"
            if flag == "OK":
                ok += 1
            else:
                fail += 1
            print(f"[{i:3d}/{len(work)}] {flag} {code:12s} rows={n:<6} {msg}")
            time.sleep(settings.request_sleep_sec)

    panel = build_panel(settings, universe["code"].tolist() if codes is None else codes)
    if not panel.empty:
        bench = panel[panel["code"] == settings.benchmark]
        if bench.empty:
            bench = panel
        days = trading_days_from_benchmark(bench)
        save_calendar(days, calendar_path(settings))
        print(
            f"\n面板已写 {panel_path(settings)}\n"
            f"行数={len(panel)}  股票数={panel['code'].nunique()}  "
            f"日期 {panel['date'].min().date()} → {panel['date'].max().date()}"
        )
    print(f"完成: 成功 {ok}  失败 {fail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载纳指100 + QQQ 日K到本地")
    parser.add_argument("--full-refresh", action="store_true", help="忽略本地缓存，整段重拉")
    parser.add_argument("--limit", type=int, default=None, help="只拉前 N 只，试跑用")
    parser.add_argument(
        "--codes",
        nargs="*",
        default=None,
        help="指定代码，如 US.AAPL US.QQQ",
    )
    args = parser.parse_args()
    run_ingest(full_refresh=args.full_refresh, limit=args.limit, codes=args.codes)


if __name__ == "__main__":
    main()