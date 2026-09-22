# ndxbots

納斯達克 100 本機量化平台：從富途拉日 K，算因子，生成觀察池，再和 QQQ 做日線回測。

行情和回測結果只存在你電腦的 `data/` 裡，**不要上傳到 GitHub**。

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

## 每天怎麼跑

按順序執行。前一步沒成功，後面會找不到檔案。

### 1. 測連線

```powershell
python -m ndxbots.data.check
```

能印出 OpenD 位址和納指 100 板塊就說明通了。失敗時先看 OpenD 是否啟動、連接埠是否 11111、美股行情權限。

### 2. 拉日 K（第一次先試 2 檔）

```powershell
python -m ndxbots.data.ingest --codes US.AAPL US.QQQ
python -m ndxbots.data.inspect_panel
```

全市場成分 + QQQ：

```powershell
python -m ndxbots.data.ingest
```

第一次全量大約 10–20 分鐘，保持 OpenD 開著。之後每天收盤後再跑同一條，只補缺的日期。

| 參數 | 作用 |
|---|---|
| `--codes US.AAPL US.QQQ` | 只拉指定代碼 |
| `--limit 5` | 只拉前 5 檔，試限頻 |
| `--full-refresh` | 忽略本機快取，整段重拉 |

價格為 **前復權**。成分優先用富途「納斯達克100 / NDX」板塊；搜不到就用內建兜底名單。

### 3. 算因子

```powershell
python -m ndxbots.factors.compute
```

會寫出 `data/factors/daily.parquet`，包含動量、波動、RSI，以及 `custom.py` 裡的 `my_*` 因子。

自己加因子：編輯 `src/ndxbots/factors/custom.py`，再重新 compute。

可選，批次看哪些因子更能預測未來 21 日是否跑贏 QQQ：

```powershell
python -m ndxbots.factors.mine --horizon 21
```

### 4. 生成觀察池

```powershell
python -m ndxbots.strategy.build
```

預設規則（均可在 `config.yaml` 的 `strategy` 段改）：

- 用 `my_ma50_gap`、`my_struct_gap` 橫截排名打分
- 收盤必須在日線 MA200 上方
- 離 21 日高點回撤大約在 1%–12% 之間（避免追頂或結構壞掉）
- ATR 太小不做
- 觀察池 Top 10，真正持倉最多 4 檔

執行層只許盯池內股票。

### 5. 回測（對齊 QQQ）

```powershell
python -m ndxbots.backtest
```

T 日收盤定池，權重作用在 T+1 收益上（`next_close`），單邊成本預設 10 bp，初始資金 20000。摘要會列印到螢幕，並寫入 `data/backtest/stats.txt`。

## 目錄

```text
config.yaml                 連線、區間、選股、回測參數
src/ndxbots/data/           拉行情、檢查、面板
src/ndxbots/factors/        算因子、挖因子
src/ndxbots/strategy/       打分、觀察池
src/ndxbots/backtest/       日線回測
ndxbots_patch/              歷史補丁副本（原始碼已合併進 src/，一般不用再拷）

data/                       本機生成，已加入 .gitignore
  raw/kline/                每檔股票一個 parquet
  panel/daily.parquet       對齊後的長表
  meta/universe.csv         成分來源
  meta/us_trading_days.parquet
  factors/daily.parquet
  mining/factor_ranking.csv
  strategy/scores.parquet
  strategy/observation_pool.csv
  backtest/equity.csv
  backtest/holdings.csv
  backtest/stats.txt
```

面板欄位大致為：`date, code, open, high, low, close, volume, turnover, pe_ratio, ...`。代碼帶 `US.` 前綴。

## 常用設定

`config.yaml` 裡這幾項最常改：

```yaml
data:
  kline_start: "2016-01-01"   # 下載起點
  # kline_end: "2026-09-01"   # 不寫 = 拉到最新

strategy:
  factors: [my_ma50_gap, my_struct_gap]
  top_n: 10
  max_hold: 4
  require_above_ma200: true

backtest:
  start: "2018-01-01"         # 回測起點，更早的資料留給 MA200 預熱
  cost_bps: 10
  initial_cash: 20000
```

臨時覆寫（可選）：環境變數 `FUTU_HOST`、`FUTU_PORT`、`KLINE_START`、`KLINE_END`、`DATA_DIR`。

## 不要提交到 GitHub

- `.venv/` 虛擬環境
- `.env` 金鑰
- `data/` 行情和回測結果
- `__pycache__/`、`.idea/`

`.gitignore` 已經擋住這些。用網頁拖檔案時，請不要選擇它們。
