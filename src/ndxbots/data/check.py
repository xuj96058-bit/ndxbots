from __future__ import annotations

from ndxbots.config import load_settings
from ndxbots.data.futu_client import FutuClient
from ndxbots.data.universe import find_ndx_plate, fallback_codes


def main() -> None:
    settings = load_settings()
    print(f"OpenD  {settings.futu_host}:{settings.futu_port}")
    print(f"数据目录  {settings.data_root}")

    client = FutuClient(settings)
    with client.session():
        print(client.ping())
        hits = find_ndx_plate(client, settings.plate_keywords)
        if hits.empty:
            print("未搜到纳指100板块，将使用内置兜底名单。")
            print(f"兜底数量: {len(fallback_codes(settings))}")
        else:
            print("可能的纳指100板块：")
            print(hits[["code", "plate_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
