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

### 4. 生成觀察池

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

### 5. 回測（對齊 QQQ）

```powershell
python -m ndxbots.backtest
```

T 日收盤定池，權重作用在 T+1 收益上，單邊成本預設 10 bp。

## 不要提交到 GitHub

- `.venv/`、`.env`、`data/`、`__pycache__/`
