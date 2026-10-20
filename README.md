# ndxbots

> **提示：這個是初步具備盈利能力的模型，完整版的請 email 聯系，xuj96058@gmail.com**

納斯達克 100 本機研究平台：從富途 OpenD 讀取行情、計算日線因子與觀察池，並用已確認的第一組日內成交規則回測。此版本已取代舊成交邏輯。

行情、帳戶模擬及研究結果存放在本機 `data/`；**不要上傳行情、回測結果、`.env` 或虛擬環境到 GitHub**。`docs/research/` 僅保存經用戶確認公開的派生示例圖。研究流程使用行情介面；獨立的 `order.py` 提供手動下單。新增的 `python -m ndxbots.paper` 按既有策略自動執行指定的富途 **US SIMULATE 模擬帳戶**，預設預覽，沒有自動實盤入口。

## 已確認的策略與成交規則

日線選股規則保持原設定：走勢品質權重 **0.40**、趨勢組 **0.60**、多空合併 **Top 8**、全場最多持 **2 檔**。`allow_short: true`；日線 MACD 閘仍為關閉。普通換股在星期二、星期五收盤後評估；每天檢查方向退出及既有情緒減倉。

T 日收盤後先確定方向、目標倉位與買入名義預算，再在下一個 QQQ 交易日 T+1 執行。T+1 必須完成**實際可供資金支持的全部買入預算**，分批比例按預算計算；減倉與清倉按既有持倉數量執行。持倉名額、可用現金、其他待買預算與成本仍受帳本限制，不能把未取得資金的原始申請金額當作已完成買入。

以下時間均為 **America/New_York／美東市場時間**，會隨夏令時間轉換：

| 動作 | 第一組固定規則 |
| --- | --- |
| 多頭買入、開盤相對 T 收盤跳高不超過 1% | 10:00 執行全部可用名義預算。 |
| 多頭買入、開盤跳高超過 1% | 10:00 先執行預算的 50%；剩餘 50% 在預先存在的限價 `T+1 開盤價 − 0.2 × T 日 ATR20` 等待回踩。 |
| 空頭新增或加倉 | 對稱使用不利跳空與反彈限價；不利跳空是相對 T 收盤下跌超過 1%，限價為 `T+1 開盤價 + 0.2 × T 日 ATR20`。 |
| 限價尚未完成 | 正常日 15:55、半日市 12:55 執行剩餘預算，確保 T+1 完成。 |
| 普通減倉、換股退出 | T+1 09:45 執行。 |
| 方向失效、方向改變、目標毛曝險降低、情緒減倉或缺少當日資料造成的緊急退出／減倉 | T+1 09:30 執行，優先釋放資金與名額。 |

ATR20 是截至 T 日的最近 20 根日線真實波幅的**簡單平均**，不是 Wilder ATR，也不是 T+1 尚未完成的日線。10:00 前不啟動剩餘限價；不能用那之前已經出現的最低價宣稱限價成交。限價成交要求價格穿過限價超過 1 bp，僅觸及不算成交。五分鐘棒只能在已存在的訂單與已取得的資金下模擬觸價；同棒釋放的資金不能追溯買在該棒先前的低價。

這些日內時間、1%、50% 與 0.2 ATR 是本次確認後固定的成交規則，沒有提供優化參數開關。`backtest.exec` 只接受 `next_intraday`。回測預設起點為 `2021-01-01`；初始資金 20,000、每側成本 10 bp。日期、成本與既有選股參數仍由 `config.yaml` 設定。

## 日線評分與留倉

`strategy.factors` 為 `my_ma50_gap`、`my_struct_gap`。各因子先在當日股票截面做百分位排名，再取平均形成趨勢分數 `T`。品質因子 `my_efficiency_21` 是 21 個交易日的淨收盤變動，除以這 21 次收盤變動絕對值之和，需要 22 個完整收盤觀測；資料不足為 NaN，完整且完全平坦的路徑為 0。

品質分數多頭為 `Q_long = rank(E)`、空頭為 `Q_short = rank(-E)`。預設：

```text
long_score  = 0.6 * T       + 0.4 * Q_long
short_score = 0.6 * (1 - T) + 0.4 * Q_short
```

多頭的兩個趨勢因子各占 30%，走勢品質占 40%。空頭分數獨立計算。不要把品質因子另加進 `strategy.factors`，否則會改變趨勢組定義。

MA200 方向與斜率、RSI、布林帶寬及 QQQ 制度閘沿用 `config.yaml`。普通調倉先填空位，再按同方向評分差至少 0.05 換股；0.05 是評分差。選股狀態未滿 10 個交易日或同方向合併排名仍在前 8，優先保留；方向失效退出優先於保留期。選股狀態的留倉計數包含回測前暖身，實際成交持倉由數量帳本另行確認。

市場狀態使用廣度、情緒、擁擠度與 FINRA 融資盤，並按既有毛曝光階梯調整目標。`meta/ndx_membership.csv` 只決定廣度計算的成分日期；缺檔時標記 `static_snapshot`。它不會自動把價格資料補成完整歷史納指成分股。

情緒計算會分開輸出 `sentiment`（四項廣度的展示分數）和 `sentiment_trade`（原有排除上漲成交量的三項廣度交易分數）。倉位控制使用後者的 5 日平滑值；`sentiment_trade_smooth`、`crowding_smooth`、`gross_reason`、資料品質、情緒層 `regime_gross_target` 與 Hurst 縮放後 `gross_target` 均進入回測／執行目標審計。報告以實際訊號日為準，缺值不再顯示較早有效日冒充當日狀態。

分位使用並列值中間排名，完整不變序列為 0.5；RSI 在只有上漲、只有下跌、完全平坦時分別為 100、0、50。每项廣度及擁擠成分各自要求至少 `min_names` 個有效貢獻者，且有歷史成分表時至少覆蓋當日應有成分的 `min_coverage`（預設 80%）。輸出每项有效樣本数、覆蓋率和缺失原因；有成分表不代表價格已完整覆蓋歷史成分。

FINRA YoY 按十二個日曆月比較。CSV 可加入 `available_on` 實際發布日期；沒有時使用 `leverage_lag_days` 估計，來源標記為 `estimated_lag`。21 日估計不是經核實的歷史發布日曆，月表亦不是歷史修訂版本存檔。數據年齡從參考月末起計，超過 `leverage_max_age_days`（預設 90）停止沿用；缺少槓桿分時使用原有廣度降級公式並明確標記。

`regime.missing_policy: floor` 是預設缺值政策：市場狀態檔／交易日／有效分數／平滑暖身不足時，情緒層毛曝險使用 `gross_floor`（預設 0.6），不再靜默回到 1.0。設定 `error` 可讓回測評估區間內的缺值直接失敗。舊快取仍能讀取，但品質標記為未核實；更新後請重新執行 `python -m ndxbots.regime`。這些修正可能改變既有交易與研究結果，舊研究收益不能當作新版本結果。

## 情緒與 QQQ 關係圖

安裝可選繪圖依賴後，可對本機完整日線面板生成對照圖、散點圖、每日資料和 Pearson／Spearman 表：

```powershell
python -m pip install -e ".[research]"
python -m ndxbots.analysis.sentiment_qqq --panel data/panel/daily.parquet --start 2025-01-01 --output data/research/sentiment_qqq
```

報告包含雙軸疊圖、分面對照圖、報酬散點圖和每日對齊資料。疊圖左軸是首日設為 100 的 QQQ 前復權價格，右軸是 0–1 情緒分數，兩軸各自使用不同尺度。

先用面板完整暖身期計算指標和報酬，最後才裁切展示日期。未來 1／5／21 個 QQQ 交易日報酬只作研究標籤，不進入情緒或倉位訊號；沒有完整未來窗口的尾端日期排除，相關表保留各自樣本数。價格水準相關僅描述同期走勢，5／21 日窗口互相重疊，不能當作獨立觀測或因果、預測能力的證明。使用前復權收盤比值，沒有另行建模完整股息再投資總報酬。可用 `--baseline` 傳入同一行情上計算的舊版分數 CSV，比較改動前後；不要混用不同供應商修訂或復權基準。此報告沒有選參或權重優化，VXN 尚未加入正式倉位規則。

Hurst 的 `hard + scale` 使用 `my_hurst_126`：最近 126 根完整收盤價取 log，對 lag 2–20 的價差標準差做 log–log 回歸，再按本策略指定公式取 `H = 2 × slope`。只使用截至訊號日的資料。`strategy.hurst_gate_mode` 決定硬閘位置：

- **`legacy_regime`（目前 config.yaml 選用）**：重建 Grok 舊版的行情狀態硬閘，在形成觀察池與排名之前套用 `(~new_long) | (H > 0.67)`。`new_long` 沿用既有 MA200／斜率行情旗標，並不代表帳戶第一次買入。延續原多頭行情的股票可能在 H 不足時仍新入選或新買入；此模式不再加入實際帳戶的新開多 Hurst 檢查，以保留所選舊版策略的行為。
- **`strict_entry`**：保留修復版的真正新開多硬閘。原觀察池及排名形成後，新增持股候選必須 `H > 0.67`；帳戶實際新開多再檢查同日 H，避免回測前暖身的選股狀態繞過限制。既有持倉不會單因 H 降低而被剔除；被阻擋的空轉多仍可平掉原空倉。

兩種模式都沿用原持倉、換股及方向規則，且 Hurst 硬閘不作用於空頭。模式切換後需重新建置策略評分表；不要混用另一模式的舊評分快取。正式成交始終採下述原有分鐘規則，模式切換不改成交時間、限價、現金預留或完成度檢查。

最終選股確定後，以基準（預設 `US.QQQ`）當日 H 縮放既有毛曝險目標：`H > 0.75` 乘 1.3、`H < 0.55` 乘 0.5，其餘及當日缺值乘 1.0，縮放後上限為 1.5。訊號仍由原有調倉／風險減倉規則產生，並在 T+1 執行。現金及成本限制仍有效，1.3 倍目標不代表會借款完成 130% 的純多倉。更新後應先重算日線因子；沒有 `my_hurst_126` 欄的舊因子表沿用未加 Hurst 硬閘時的行為。有欄但個股缺值時，`legacy_regime` 只在上述行情旗標為真時阻擋，`strict_entry` 則阻擋實際新開多。

`observation_pool.csv` 是觀察名單；`exec_pool.csv` 是本機模擬帳戶的下一次成交目標，包含池外續持、退出數量與買入預算。參考數量、名義預算和實際券商整股數量用途不同；這些檔案不是券商訂單。

## 分鐘行情與快取

日線行情使用前復權價格。新增的 `ndxbots.data.intraday` 只建立 `OpenQuoteContext`，請求 `K_5M`、`AuType.NONE`、`Session.RTH`，不建立交易 context。Futu 依帳號權限提供歷史行情與額度，缺失資料會明確報錯。

```text
data/intraday/kline/US_AAPL.parquet
data/intraday/kline/US_AAPL.json
data/intraday/calendar.parquet
data/intraday/calendar.json
```

每股快取保留未復權 OHLC、成交量、成交額。`time_key` 是供應商的棒**結束時間**，`bar_end` 帶美東時區；正常日五分鐘網格為 09:35 至 16:00，共 78 根，半日市為 09:35 至 13:00，共 42 根。半日市由 OpenD 交易日曆的 `MORNING/HALF` 明確決定；資料缺半天不會被當成半日市。[Futu 日曆文件](https://openapi.futunn.com/futu-api-doc/en/quote/request-trading-days.html) 說明日曆排除週末與假日，臨時停市仍須另行核對。

回測把當天分鐘價格按 `日線前復權開盤價 / 第一根五分鐘原始開盤價` 換成日線單位，不用 T+1 收盤決定早盤價格尺度。缺首棒、首棒零成交量、缺網格、缺日曆或缺所需股日都直接失敗。分鐘快取不會由日線價格補造，也不會在回測時自動連線抓資料。

預設分鐘下載計劃直接從日線因子呼叫純 `build_score_table`，取歷史 `in_hold` 股日及其後兩個 QQQ 交易日，覆蓋進場、調整與退出緩衝；無需先執行依賴分鐘資料的策略建置。下載窗口內無關的 IPO 或暖身初期日期先被裁掉，再檢查所需五分鐘網格。完整快取可重用；分頁失敗不會把部分資料覆蓋到既有股價檔，重新執行會以新連線重播缺失日期。

`data.intraday_root: null` 表示使用 `data.root/intraday/kline`。可設定這個欄位或 `INTRADAY_DIR` 指定另一個本機快取目錄；日曆位於該目錄上一層。

## 安裝與執行

需要 Python 3.10+、已登入的 [FutuOpenD](https://www.futunn.com/download/openAPI)，以及相應的美股歷史行情權限。OpenD 預設地址為 `127.0.0.1:11111`。

```powershell
cd ndxbots
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .
```

更新已有項目時保留本機 `data/`、`.env` 與虛擬環境，重新安裝後按此順序重建：

```powershell
python -m ndxbots.data.check
python -m ndxbots.data.ingest
python -m ndxbots.factors.compute
python -m ndxbots.regime
python -m ndxbots.data.intraday
python -m ndxbots.strategy.build
python -m ndxbots.backtest
```

指定分鐘下載日期或股票，可使用：

```powershell
python -m ndxbots.data.intraday --start 2021-01-01 --end 2026-09-18
python -m ndxbots.data.intraday --codes US.AAPL US.MSFT --start 2021-01-01 --end 2026-09-18
```

`--codes` 篩選日線信號計劃中的股票。首次下載可能受歷史代碼額度、分頁數量及 OpenD 速率限制影響。最近尚未結束的交易日不具完整網格，應將結束日期設為最近一個已完成交易日。

`--refresh-leverage` 會從 FINRA 官方月表更新融資盤；未刷新時使用倉庫的 `meta/finra_margin_debt.csv`。品質權重大於 0 時，缺少 `my_efficiency_21` 的舊因子檔會明確要求重新執行 `factors.compute`。

策略建置和回測共用本機成交帳本。回測輸出曲線、成交、目標及完成度／資金限制審計至 `data/backtest/`；策略目標的 `model_source=backtest` 表示來源是模擬帳戶。

```powershell
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

Mac/Linux 使用 `source .venv/bin/activate`。

## 手動下單：order.py

安裝後，在倉庫根目錄執行 `python order.py`，或在任何目錄使用 `python -m ndxbots.order`。支援 `accounts`（帳戶清單）、`place`（下單）、`orders`（訂單查詢）及 `cancel`（撤單）。預設 `--env SIMULATE`；`place` 和 `cancel` 未加 `--submit` 時只作離線預覽，不連線、不解鎖、不送出。

連線沿用 `config.yaml` 的 `futu.host`／`futu.port`，可用 `FUTU_HOST`／`FUTU_PORT` 覆蓋。券商預設為 `--security-firm FUTUSECURITIES`；其他券商需指定已安裝 SDK 支援的 `SecurityFirm` 名稱。帳戶和訂單查詢需要本機已登入的 OpenD；實際送單還需要相應帳戶及美股交易權限。先從 `accounts` 取得正確環境的 `acc_id`，再明確指定帳戶，不會自動使用清單的第一個帳戶。[Futu 帳戶清單文件](https://openapi.futunn.com/futu-api-doc/trade/get-acc-list.html)

以下 `123` 是示例帳戶編號，請替換成自己的模擬帳戶；價格只用於示範命令格式：

```powershell
python order.py accounts
python order.py place --acc-id 123 --code US.AAPL --side BUY --qty 1 --price 100
python order.py place --acc-id 123 --code US.AAPL --side BUY --qty 1 --price 100 --submit
python order.py orders --acc-id 123
python order.py cancel --acc-id 123 --order-id YOUR_ORDER_ID
python order.py cancel --acc-id 123 --order-id YOUR_ORDER_ID --submit
```

`--qty` 為正整數的實際股數；只支援美股整股、正常交易時段（RTH）及當日有效（DAY）訂單，不接受期權代碼。`--order-type LIMIT` 是預設，必須提供正數 `--price`：低於 1 美元最多四位小數，1 美元或以上最多兩位，不會自動取整。市價單使用 `--order-type MARKET` 並省略 `--price`。可加 `--remark`，上限為 64 個 UTF-8 位元組。`BUY`／`SELL` 是券商買賣方向：`BUY` 可能回補空倉，`SELL` 可能平多或依帳戶權限建立空倉，不等同策略的多空訊號；本入口不提供 `BUY_BACK`／`SELL_SHORT`。[Futu 下單文件](https://openapi.futunn.com/futu-api-doc/en/trade/place-order.html)

實盤先列出帳戶，核對後用該帳戶預覽；送出時必須同時指定 `--env REAL --submit --confirm-real`。以下 `456` 需替換為實盤帳戶：

```powershell
python order.py accounts --env REAL
python order.py place --env REAL --acc-id 456 --code US.AAPL --side BUY --qty 1 --price 100
python order.py place --env REAL --acc-id 456 --code US.AAPL --side BUY --qty 1 --price 100 --submit --confirm-real
python order.py orders --env REAL --acc-id 456
```

實盤需要先解鎖交易。GUI 版 OpenD 請在介面手動解鎖；程式沒有提供密碼時沿用 OpenD 的解鎖狀態。命令列版 OpenD 可透過 `FUTU_TRADE_PASSWORD` 環境變數（或本機 `.env`）解鎖，或在實盤 `place`／`cancel` 加 `--unlock`，於未設定該變數時安全輸入密碼。密碼不接受命令列參數；使用 GUI 版時不要設定此變數或加 `--unlock`。實盤撤單同樣需要 `--env REAL --submit --confirm-real`。

送單或撤單後應以 `orders` 查核狀態，可用 `--order-id` 查指定訂單；查詢涵蓋未完成訂單及最近 24 小時完成／取消的訂單，更早記錄請在券商介面查閱。收到訂單編號不表示已成交，撤單申請也不保證成功。若連線逾時或結果不明，程式不會自動重送；先查詢訂單及券商介面，再決定是否重試，以免重複下單。

`order.py` 不讀取 `exec_pool.csv`，也不自動執行策略或 T+1 日內排程。研究帳戶的前復權參考數量不能直接當券商股數。下節的 `ndxbots.paper` 是獨立的模擬盤自動執行入口；手動下單命令的行為保持不變。

## 模擬盤自動執行：ndxbots.paper

`ndxbots.paper` 共用既有日線評分、選股及目標倉位規則，從指定模擬帳戶的實際持倉、資金及訂單建立 T+1 計劃。它不把回測的 `exec_pool.csv` 當券商指令，也不重跑歷史回測。只接受美股整股、RTH 與 DAY 訂單；目前沒有 REAL 選項。

需要已登入的 OpenD、相應行情權限及一個專供此策略使用的 US SIMULATE 帳戶。先列出帳戶並核對 ID；以下 `123` 是範例，必須換成你的模擬 `acc_id`：

```powershell
python -m ndxbots.paper accounts
python -m ndxbots.paper prepare --acc-id 123 --refresh
python -m ndxbots.paper status --acc-id 123
python -m ndxbots.paper tick --acc-id 123
```

`prepare` 讀取券商資料並保存本機計劃，不送單；`tick` 預設只對帳／預覽，不保存新的執行進度或送出訂單。建立新計劃的窗口是**最近正式交易日收盤後至少 20 分鐘，至下一個正式交易日 09:30 前**，全部按 `America/New_York`。週末、假日及半日市以 OpenD 官方交易日曆為準。首次在盤中啟動而沒有計劃時，worker 會等待收盤後的下一個準備窗口，不補做當天的早盤交易。

`--refresh` 在每個新訊號日完整更新設定區間的前復權日線，然後重算因子及市場狀態，避免把拆股前後不同尺度的快取拼接。它會使用歷史行情額度；預期股票缺少 T 日資料、下載失敗或更新逾時時不建立計劃。請預留下載時間。未加 `--refresh` 時使用已有日線／因子，仍檢查日期與完整性。前一份計劃或訂單未解決時，不覆蓋帳本建立下一份。

持續預覽及實際模擬執行分別使用：

```powershell
# 持續讀取並預覽；可建立本機計劃，但不送單或撤單
python -m ndxbots.paper run --acc-id 123 --refresh --poll-seconds 10

# 核對帳戶與計劃後，送出／撤銷模擬訂單
python -m ndxbots.paper run --acc-id 123 --submit --refresh --poll-seconds 10
```

輪詢最短 5 秒，建議 10 秒。worker 按前述 09:30／09:45／10:00 及收盤前五分鐘規則處理執行；首次錯過應開始時間超過兩分鐘會停止該計劃的自動執行，不把遲到交易標為準時完成。已存在的限價以券商實際成交回報為準；回測的五分鐘棒穿價假設不會生成券商成交。

執行狀態保存於 `data/paper/{acc_id}.sqlite3`，送單意圖先持久保存，再交給券商。重啟先核對訂單累計成交、持倉與資金；送單／撤單結果不明時不自動重送。截止前先撤剩餘限價，確認最終成交及撤單結果後才計算補單。整股尾差會保留；行情過期、公司行動價格不一致、資金／借券額度不足或拒單可能導致未完成，不能保證券商在期限內成交。

```powershell
python -m ndxbots.paper status --acc-id 123
python -m ndxbots.paper stop --acc-id 123
python -m ndxbots.paper resume --acc-id 123
```

`stop` 寫入持久停止旗標；正在以 `--submit` 運行的 worker 會在後續對帳中嘗試撤銷自己可確認的未完成訂單，保留現有持倉。停止命令本身不連線券商，也不代表掛單已撤。`resume` 只清除停止旗標，不會清空不明訂單、帳本或重新啟動已停掉的服務。不要刪除狀態庫來繞過對帳，也不要在兩台機器同時運行同一帳戶。

目前即使上一份計劃已完成，既有帳本仍會核對建立計劃時的策略設定指紋。修改 `config.yaml` 的策略參數會阻止後續執行及新計劃；需要先停止 worker、核對訂單與持倉，再另行進行受控狀態遷移。本版本沒有自動遷移命令，不能用刪除帳本解決。

驗證與部署：

```powershell
python -m unittest discover -s tests -p "test_paper*.py" -v
```

測試使用假的 SDK／券商及可控時鐘，覆蓋計劃、時間窗口、部分成交、撤單競態、重啟與重複送單防護；**離線測試通過不代表已通過 OpenD 模擬盤整合**。首次連線仍需在已核對帳戶觀察行情、送單、成交、撤單與重啟。伺服器安裝、OpenD 登入、systemd 範本及備份步驟見 [模擬盤部署文件](docs/paper-deployment.md)；範本不代表已部署或啟動伺服器。

## 研究快照與限制

目前預設策略對應用戶選定的「重建舊硬閘＋QQQ 縮放」。本機缺資料研究從 **2018-01-02 至 2026-09-18**，以 2016 年起日線暖身、本金 20,000、單邊成本 10 bp，期末為 **1,147,806.29**，累計 **5,639.03%**、252 日年化 **59.3628%**、Sharpe **1.3884**、最大回撤 **37.6848%**。該研究有 1,026 筆原分鐘成交及 635 筆缺資料後的 T+1 收盤替代成交；645 筆無資金加倉未成交，10 筆加倉等到收盤賣出釋放資金才成交。原完成度旗標保留，不能稱作全部訂單都符合原盤中期限。

**收盤替代只用於本機研究，不屬於本倉庫的正式執行層。** `python -m ndxbots.backtest` 和策略帳本仍要求完整的原生五分鐘資料，缺資料報錯；没有自動收盤替代，也不會合成分鐘棒。上述研究終值不能直接當作完整分鐘成交的回測結果。此次更新保留正式回測原來的 `2021-01-01` 預設起點，研究起點 2018 年由本機研究腳本指定。

逐筆核對該舊版研究，316 次實際新開多中有 49 次訊號日 H 不大於 0.67。這是 `legacy_regime` 的已知行為；選用 `strict_entry` 會改變交易及收益。GitHub 上不含本機行情或研究成交檔，重現研究仍需同一份日線、分鐘快取與独立的研究替代腳本。

已確認的第一組來自 **2,368 個買賣成交組合**的歷史比較。研究期間為 2021 年起至 **2026-09-18**，初始資金 20,000、每側成本 10 bp；該次凍結資料的期末淨值為 **521,128.80**，淨年化 **77.3462%**，最大回撤約 **44.09%**。這是已檢視全期資料後選出的歷史結果；2025 年及 2026 年部分年度也已被看過，不能再宣稱是獨立樣本外驗證。

重現需要同一日線價格、因子、成分／市場狀態來源、五分鐘行情及交易日曆，並固定上述回測日期、成本與選股設定。新下載的行情修訂、復權單位及來源差異可能改變結果。原研究凍結資料保存在本機，不隨程式上傳；只拿到 GitHub 程式並不能取得或證明這次研究快照。

五分鐘 OHLC 觸及限價不保證真實成交，也沒有逐筆先後、委託排隊、買賣價差、成交量容量或開盤競價撮合模型。研究允許零碎前復權單位；期限單是保證當日完成的模擬成交假設。尚未建模借券可得性、借券費、融資與現金利息、完整保證金、稅或個人券商限制。現金限制不等於完整空頭保證金限制，價格波動可能令實際毛曝險超過目標。

歷史選股資料不是已修復的逐時點歷史納指成分全集，存在存活者與成分選擇偏差。分鐘及日線來源的 OHLC／復權差異也需要保留審計。程式未加入新的止損或組合回撤限制；歷史年化及回撤不是未來收益承諾。`order.py` 提供明確指定股數的手動訂單；`ndxbots.paper` 的股數換算、借券額度查詢及對帳只適用於此處明確限定的模擬盤流程，沒有把策略接入自動實盤。

不要提交：`data/`、`.env`、`.venv/`、`__pycache__/` 或本機行情、回測及研究結果；經用戶確認公開且放在 `docs/research/` 的派生示例圖除外。

