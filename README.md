# ndxbots

納斯達克 100 本機量化平台。從富途拉日 K，算因子，生成觀察池，再和 QQQ 做日線回測。

行情和回測結果只存在你電腦的 `data/` 裡，**不要上傳到 GitHub**。

## 有什麽功能

- **行情**：連富途 OpenD 拉納指 100 和 QQQ 日 K，落到 `data/raw/kline/`，再拼成 `data/panel/daily.parquet` 和交易日曆。板塊拉不到成分時，用內置名單兜底。
- **因子**：日 K 算收益、波動、RSI、距高點距離，以及 `custom.py` 裡的 `my_*`（MA50 缺口、結構缺口、MA200、ATR、MACD、布林帶）。另有因子挖掘（`factors.mine`）。
- **觀察池與持倉**：用 `my_ma50_gap`、`my_struct_gap` 橫截排名打分。MA200 斜率分強、弱、走平，決定多空方向。多空合併一共 Top 10，不是每側 10。觀察池與持倉候選都按 `side_score` 和合併 `combo_rank` 排序，同分次序一致。全場最多持 4 檔，池外舊倉仍可能續持。
- **閘門**：RSI 只卡新開；布林帶寬太窄不做；QQQ 近 21 日漲跌分大漲、跌勢、震盪，只收紧新開條件，不改打分。MACD 閘默認關。
- **市場狀態**：廣度、情緒、擁擠度、FINRA 融資盤。廣度成分讀 `meta/ndx_membership.csv`，按 `code,start,end` 篩選；沒有這份表就用當前名單，並標成 `static_snapshot`。這份成分表沒有把選股價格資料自動補成完整歷史納指成分股。狀態不改個股評分。
- **倉位折扣**：`use_gross_ladder` 打開時，用 5 日情緒做階梯減倉；情緒和擁擠的 5 日均值都過熱時，毛曝光收到地板。沒有狀態檔就退回等權。
- **回測**：T 日收盤後產生固定申請數量，T+1 交易日收盤成交，之後才計算新倉收益。用實際模擬數量、成交價及現金計算淨值；單邊成本默認 10 bp，按實際成交金額收取。普通調倉目標每檔等權、空頭為負；非調倉日保持數量，退出空位留現金。新增買入所需現金不足時按比例部分成交。

參數都在 `config.yaml`，不必改程式碼。

## 統一排名＋分差版本

- 普通換股在星期二、星期五收盤後評估，假日不額外補調倉。成交在下一個 QQQ 交易日收盤，通常為星期三、星期一。
- 新倉只從合併 Top 10 選取，先填真正空位，再處理換股。舊倉持有未滿 10 個交易日，或同方向合併排名仍在前 8，優先保留。
- 可替換舊倉須有同方向的新候選，且新股評分至少高 0.05 才換股。0.05 是評分差，不是預期收益 5%。若只是失去新倉資格但方向仍有效，可以續持。
- 每天檢查方向：缺少當日資料、方向失效或改變時退出，優先於最少持有天數。既有情緒降倉仍每天檢查；增倉等待普通調倉日。
- `observation_pool.csv` 是觀察名單；`exec_pool.csv` 是完整模擬持倉的下一次成交目標，包含池外續持及零目標退出，兩者用途不同。
- 這版沒有加入組合回撤階梯、額外反彈倉、波動率調權或新的空頭限額。

相關參數：

```yaml
strategy:
  rebalance_weekdays: [1, 4]  # Python 星期編號：星期二、星期五
  replacement_score_gap: 0.05
  max_hold: 4
  min_hold_days: 10
  keep_rank: 8
backtest:
  exec: next_close
  cost_bps: 10
```

此處的持倉與數量是研究模型，不會向券商下單。前復權價格和零碎調整單位不能直接當成券商整股數量；尚未建模借券可得性、借券費、保證金、融資利息、稅及閒置現金利息。現金限制不等於完整空頭保證金限制，價格變動也可能令實際毛曝險高於目標。歷史數據必須另行核對覆蓋範圍與發布日期。

`ndxbots_patch/` 是舊版存檔，請勿用它覆蓋本版本。更新現有本機項目時保留自己的 `data/`、`.env` 和虛擬環境，使用主目錄的程式與設定重新生成觀察池及回測。

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

策略建置會讀取已有的行情面板，用與回測相同的成交邏輯生成 `exec_pool.csv`。其中 `model_source=backtest`，數量與淨值來自本地模擬，不代表券商實際持倉；用於真實帳戶時必須與該帳戶的持倉、現金及已成交訂單重新對帳。

驗證規則：

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

## 不要提交到 GitHub

- `.venv/`、`.env`、`data/`、`__pycache__/`
