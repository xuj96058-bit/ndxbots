from __future__ import annotations

import argparse

from ndxbots.backtest.engine import bt_dir, run_backtest
from ndxbots.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="观察池日线回测（对齐 QQQ）")
    parser.parse_args()
    settings = load_settings()
    result = run_backtest(settings)

    out = bt_dir(settings)
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / "equity.csv"
    hold_path = out / "holdings.csv"
    stats_path = out / "stats.txt"

    result["curve"].to_csv(curve_path, index=False)
    result["holdings"].to_csv(hold_path, index=False)
    text = str(result["stats"])
    stats_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n已写 {curve_path}")
    print(f"已写 {hold_path}")
    print(f"已写 {stats_path}")


if __name__ == "__main__":
    main()
