from __future__ import annotations

from ndxbots.config import load_settings
from ndxbots.data.ingest import panel_path, universe_path
import pandas as pd


def main() -> None:
    settings = load_settings()
    path = panel_path(settings)
    if not path.exists():
        raise SystemExit(f"还没有面板: {path}\n请先运行 python -m ndxbots.data.ingest")

    df = pd.read_parquet(path)
    print(f"文件: {path}")
    print(f"行数: {len(df)}")
    print(f"股票: {df['code'].nunique()}")
    print(f"区间: {df['date'].min().date()} → {df['date'].max().date()}")
    print("\n每只行数（最少的 8 只，可能上市较晚或下载失败）:")
    counts = df.groupby("code").size().sort_values()
    print(counts.head(8).to_string())
    print("\n样本:")
    print(df.head(3).to_string(index=False))

    uni = universe_path(settings)
    if uni.exists():
        u = pd.read_csv(uni)
        print(f"\n成分来源: {u['source'].iloc[0] if 'source' in u.columns else '?'}")
        print(f"成分文件: {uni}  ({len(u)} 行)")


if __name__ == "__main__":
    main()
