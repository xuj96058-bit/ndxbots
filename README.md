# ndxbots

納斯達克 100 本機量化平台：從富途拉日 K，算因子，生成觀察池，再和 QQQ 做日線回測。另有一層市場狀態：情緒與擁擠度不改選股，只在回測裡把已選好的倉打一個小於等於 1 的折扣。

行情和回測結果只存在你電腦的 `data/` 裡，**不要上傳到 GitHub**。

## 最近一次核對（2026-10-03）

用本機日 K 離線跑完整條管線，沒有連富途重拉。行情覆蓋 102 檔（含 `US.QQQ`），2016-01-04 → 2026-09-18。成分用 `meta/ndx_membership.csv`（point-in-time），融資盤用倉庫裡截至 2026-08 的 FINRA 表。這是當時 `config.yaml` 的一次結果，不是收益承諾。

回測區間 2018-01-02 → 2026-09-18，2,190 個交易日，期初 $20,000，單邊成本 10 bp，階梯減倉開著。

| | 策略 | QQQ |
|---|---:|---:|
| 期末 | $99,102 | $96,576 |
| 總收益 | +395.5% | +382.9% |
| 年化 | +20.17% | +19.82% |
| 超額 | +12.6 個百分點 | |
| 波動 | 32.6% | |
| Sharpe | 0.73 | |
| 信息比 | 0.08 | |
| 最大回撤 | −58.7% | |
| 日勝率 | 52.8% | |
| 日均換手 | 9.9% | |
| 日均多 / 空 | 2.90 / 1.10 | |

曆年收益（當年期初淨值到年末；2026 截至 09-18）：

| 年 | 策略 | QQQ |
|---|---:|---:|
| 2018 | −4.7% | −1.9% |
| 2019 | +0.5% | +39.2% |
| 2020 | +192.6% | +48.7% |
| 2021 | +52.1% | +27.4% |
| 2022 | −41.4% | −32.6% |
| 2023 | +2.0% | +54.9% |
| 2024 | +62.5% | +25.8% |
| 2025 | −23.3% | +20.8% |
| 2026 YTD | +56.1% | +17.7% |

2020 把曲線拉開，2022 和 2025 又把超額吐回去。全段只略跑贏 QQQ，回撤比指數深。

2026-09-18 市場狀態：潛在底部區，情緒 0.262，擁擠 0.251，當日 QQQ 制度為震盪。觀察池多 CRWD、FTNT、MRVL、PANW、MU，空 AXON、ADSK、GEHC、PDD、SHOP。當天實際持倉因留倉規則不是這 10 檔，而是多 CRWD、KLAC、ON，空 AXON。

## 你需要先有的

1. Python 3.10+
2. 已安裝並登入 [FutuOpenD](https://www.futunn.com/download/openAPI)（預設 `127.0.0.1:11111`）
3. 富途帳號已開通 **美股行情**

## 安裝（Windows）

```powershell
cd ndxbots
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Mac / Linux 把啟動換成 `source .venv/bin/activate`。

改連線或選股參數，編輯專案根目錄的 `config.yaml`，不必改程式碼。

## 每天怎麽跑

按順序執行。前一步沒成功，後面會找不到檔案。

### 1. 測連線

```powershell
python -m ndxbots.data.check
```

### 2. 拉日 K

```powershell
python -m ndxbots.data.ingest --codes US.AAPL US.QQQ
python -m ndxbots.data.inspect_panel
python -m ndxbots.data.ingest
```

### 3. 算因子

```powershell
python -m ndxbots.factors.compute
```

### 4. 情緒與擁擠度

```powershell
python -m ndxbots.regime
python -m ndxbots.regime --refresh-leverage
```

`--refresh-leverage` 會從 FINRA 官方月表更新融資盤，寫到 `data/meta/finra_margin_debt.csv`。不刷新時用倉庫裡的 `meta/finra_margin_debt.csv`（截至 2026-08）。

輸出在 `data/regime/daily.parquet`。這是狀態變數：

- 情緒 > 0.8：過熱，不代表馬上跌
- 情緒 < 0.2：過冷，不代表馬上漲
- 情緒和擁擠都 > 0.8：雙重極端，收緊風控、不追高
- QQQ 創 252 日新高但情緒未過熱：內部背離

成分要 point-in-time，把 `meta/ndx_membership.example.csv` 抄成 `meta/ndx_membership.csv`，欄位是 `code,start,end`。沒有這份表時，用當前名單並標成 `static_snapshot`，歷史廣度會有成分偏差。倉庫裡已有一份 `meta/ndx_membership.csv`。

### 5. 生成觀察池

```powershell
python -m ndxbots.strategy.build
```

預設規則（均可在 `config.yaml` 的 `strategy` 段改）：

- 用 `my_ma50_gap`、`my_struct_gap` 橫截排名打分
- MA200 斜率五段決定多空方向；新開多仍須在 MA200 上方
- **觀察池多空合併一共 Top 10**（不是每側 10）
- 全場真正持倉最多 4 檔
- RSI 閘只卡新開：多 ≤75、空 ≥25；布林帶寬截面分位低於 20% 不做
- QQQ 市場狀態：震盪可新開、持倉上限仍為 4；大漲空頭更嚴但不禁止（合併池最多 3 空，新開空須低於 MA200 至少 4%）；跌勢新開多須高於 MA200 至少 4%

### 6. 回測（對齊 QQQ）

```powershell
python -m ndxbots.backtest
```

T 日收盤定池，權重作用在 T+1 收益上，單邊成本預設 10 bp。

`regime.use_gross_ladder` 打開時，回測會讀 `data/regime/daily.parquet`，用去掉上漲成交量的 5 日情緒做階梯減倉，情緒和擁擠的 5 日均值都 ≥ 0.80 時收到 0.60。沒有狀態檔就退回等權。選股不變。關掉這個開關即回復原版。

## 不要提交到 GitHub

- `.venv/`、`.env`、`data/`、`__pycache__/`
