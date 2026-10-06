# ndxbots

納斯達克 100 本機量化平台。從富途拉日 K，算因子，生成觀察池，再和 QQQ 做日線回測。

行情和回測結果只存在你電腦的 `data/` 裡，**不要上傳到 GitHub**。

## 有什麽功能

- **行情**：連富途 OpenD 拉納指 100 和 QQQ 日 K，落到 `data/raw/kline/`，再拼成 `data/panel/daily.parquet` 和交易日曆。板塊拉不到成分時，用內置名單兜底。
- **因子**：日 K 算收益、波動、RSI、距高點距離，以及 `custom.py` 裡的 `my_*`（MA50 缺口、結構缺口、MA200、ATR、MACD、布林帶）。另有因子挖掘（`factors.mine`）。
- **觀察池**：用 `my_ma50_gap`、`my_struct_gap` 橫截排名打分。MA200 斜率分強、弱、走平，決定多空方向。多空合併一共 Top 10，不是每側 10。全場最多持 4 檔，舊倉有最少持有天數和留倉名次。
- **閘門**：RSI 只卡新開；布林帶寬太窄不做；QQQ 近 21 日漲跌分大漲、跌勢、震盪，只收紧新開條件，不改打分。MACD 閘默認關。
- **市場狀態**：廣度、情緒、擁擠度、FINRA 融資盤。成分讀 `meta/ndx_membership.csv`，按 `code,start,end` 做 point-in-time；沒有這份表就用當前名單，並標成 `static_snapshot`。狀態不改選股。
- **倉位折扣**：水平五區只在極度貪婪把毛曝光收到 0.60；Z-score 上穿 +1.5 減 0.20、下穿 -1.5 加 0.20，地板 0.60。雙熱仍收到地板。不改觀察池。外部 VIX/HYG/LQD/TLT 由 `regime` 向 Yahoo 讀，落到 `data/meta/`。見 `docs/sentiment_overlay.md`。
- **回測**：T 日收盤定池，權重作用在 T+1 收益。多空搶同一組 4 個名額，每檔等權，空頭為負。對齊 QQQ，單邊成本默認 10 bp。輸出淨值、持倉和摘要。

參數都在 `config.yaml`，不必改程式碼。

## 這版增加的

- 市場狀態層：情緒五區、Z-score 變化倉位、擁擠度、融資槓桿、成分 point-in-time。
- 毛曝光只縮已選好的倉，不改觀察池。
- QQQ 制度閘：大漲、跌勢、震盪分別限制新開，不禁止對應方向。
- 多空合併池：一共 10 檔觀察、4 檔持倉，名額由多空抢。
- MA200 斜率分段、空間過濾、最少持有天數和留倉名次。

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

## 每天怎麽跑

按順序執行。前一步沒成功，後面會找不到檔案。

```powershell
python -m ndxbots.data.check
python -m ndxbots.data.ingest
python -m ndxbots.factors.compute
python -m ndxbots.regime
python -m ndxbots.strategy.build
python -m ndxbots.backtest
```

`--refresh-leverage` 會從 FINRA 官方月表更新融資盤。不刷新時用倉庫裡的 `meta/finra_margin_debt.csv`。

## 不要提交到 GitHub

- `.venv/`、`.env`、`data/`、`__pycache__/`
