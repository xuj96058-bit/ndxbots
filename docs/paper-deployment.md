# Ubuntu 伺服器：ndxbots 模擬盤自動執行

這份文件部署 `python -m ndxbots.paper`，只連接明確指定的 **US SIMULATE 帳戶**。自動入口沒有 REAL 選項。`prepare` 建立計劃；`tick` 和 `run` 預設不送單，只有加上 `--submit` 才會送出或撤銷模擬訂單。

本倉庫提供可檢視的安裝步驟及 systemd 範本。提供範本不代表已購買伺服器、安裝 OpenD、完成登入或通過券商整合測試。仍需取得伺服器 SSH 存取、OpenD 行情權限及選定模擬帳戶。

## 目錄與程序

建議先使用 Ubuntu 24.04 LTS、x86_64。2 vCPU／4 GB RAM 可作低併發起點；這是估算，沒有伺服器負載基準。歷史資料批次計算較多時再評估 8 GB。這個執行服務不需要 GPU，也不在每次輪詢執行歷史回測。

| 項目 | 位置／用途 |
| --- | --- |
| 程式與虛擬環境 | `/opt/ndxbots`、`/opt/ndxbots/.venv` |
| 行情、因子、策略與帳戶狀態 | `/opt/ndxbots/data/` |
| 指定帳戶的持久狀態 | `data/paper/{acc_id}.sqlite3` |
| 指定帳戶的停止旗標 | `data/paper/{acc_id}.STOP` |
| OpenD 發行包 | `/opt/futu-opend/` |
| 執行使用者及其 home | `ndxbots`、`/var/lib/ndxbots` |
| 服務環境設定 | `/etc/ndxbots/paper.env` |
| 程式服務 | `ndxbots-paper.service` |
| 本機券商連線閘道 | `futu-opend.service`，`127.0.0.1:11111` |

兩個服務使用同一個非 root 使用者。這也讓 OpenD 首次互動登入與後續服務啟動使用相同的 home 及檔案權限。

## 1. 安裝程式

以有 sudo 權限的 SSH 使用者執行。若已建立 `ndxbots` 使用者，略過 `useradd`。以下 clone 的分支必須先發布到 GitHub；部署前確認它是已檢視及測試的版本。

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip git ca-certificates tzdata
sudo useradd --system --create-home --home-dir /var/lib/ndxbots --shell /bin/bash ndxbots
sudo install -d -o ndxbots -g ndxbots -m 0750 /opt/ndxbots /opt/futu-opend
sudo install -d -o root -g ndxbots -m 0750 /etc/ndxbots
sudo -u ndxbots -H git clone --branch codex/paper-auto-execution https://github.com/xuj96058-bit/ndxbots.git /opt/ndxbots
sudo -u ndxbots -H python3 -m venv /opt/ndxbots/.venv
sudo -u ndxbots -H /opt/ndxbots/.venv/bin/python -m pip install -e /opt/ndxbots
```

記錄部署的 commit，將它與測試結果一起保存：

```bash
sudo -u ndxbots -H git -C /opt/ndxbots rev-parse HEAD
sudo -u ndxbots -H bash
cd /opt/ndxbots
.venv/bin/python -m unittest discover -s tests -v
exit
```

這些是離線測試。即使全部通過，也不能代替 OpenD 登入、實際行情、帳戶查詢及模擬訂單生命週期驗證。

## 2. 搬移既有資料

保留本機 `config.yaml` 的已確認策略設定，將已有行情及研究所需檔案搬到伺服器的 `data/`。如果曾運行自動模擬執行，先停止舊機的 worker，確認沒有第二個程序繼續送單，再搬移同一帳戶的 `data/paper/` 狀態與停止旗標。不要在兩台機器同時運行同一帳戶。

應讓搬入的檔案屬於 `ndxbots` 使用者。`.env` 中若有其他本機設定，僅透過私人 SSH 通道搬移必要項目；不要放進 Git。這個 SIMULATE 服務不需要交易解鎖密碼。

首次更新的日線與因子可能需要時間及行情額度。`--refresh` 會在建立新計劃時完整重抓設定區間的日線，再更新因子與市場狀態，避免拆股後混合不同前復權尺度；每個訊號日的成功更新會保存標記。它不是歷史分鐘下載或歷史回測。完整日線更新會使用歷史行情額度；任何預期股票缺少 T 日資料或下載失敗，都會拒絕建立計劃並報出原因。請在開盤前預留更新時間，檢查資料日期及計劃，而不是到 09:30 才首次安裝。

資料擷取最多等待 30 分鐘，因子與市場狀態更新各最多 10 分鐘；任一步驟失敗或逾時都不會標記本次更新成功。券商讀取及送單分別在有時間上限的隔離程序內執行；送單／撤單逾時代表結果不明，需要用已保存意圖對帳，不能直接重送。

## 3. 安裝及登入 OpenD

從 [Futu 官方命令列 OpenD 文件](https://openapi.futunn.com/futu-api-doc/en/opend/opend-cmd.html) 的 Ubuntu 下載入口取得適合 x86_64 的發行包。解壓整個包至 `/opt/futu-opend/`，確保 `FutuOpenD`、`FutuOpenD.xml`、`Appdata.dat` 位於該目錄；保留包內其他檔案。不要只複製可執行檔。

確認目錄可由 `ndxbots` 使用者讀寫後，以該使用者互動啟動：

```bash
sudo -u ndxbots -H bash
cd /opt/futu-opend
./FutuOpenD -cfg_file=/opt/futu-opend/FutuOpenD.xml -api_ip=127.0.0.1 -api_port=11111 -console=1 -no_monitor=1
```

由使用者在 OpenD 的互動提示中輸入帳戶、密碼及完成驗證，選擇記住登入。上述服務範本針對支援記住登入的 OpenD 10.10 或以上版本；後續使用 `-login_by_remember=1`。若登入失效，停止服務後以相同 Linux 使用者重新互動登入。

OpenD 只監聽 `127.0.0.1:11111`；不要在雲端防火牆開放 11111。保留 XML 的 telnet／WebSocket 遠端管理為未啟用，無需公開交易連接埠。密碼不要寫進 systemd unit 或 shell 命令列。

首次確認登入成功後，以 Ctrl+C 結束互動程序，再輸入 `exit` 回到管理使用者，然後啟動下述服務。不要同時啟動第二個使用相同埠的 OpenD。

## 4. 安裝設定及服務範本

```bash
sudo install -o root -g ndxbots -m 0640 /opt/ndxbots/deploy/paper.env.example /etc/ndxbots/paper.env
sudoedit /etc/ndxbots/paper.env
```

先填入 `NDXBOTS_OPEND_LOGIN_ACCOUNT`，它是 OpenD 登入 ID，與交易 `acc_id` 不同。建議使用數字 Futu ID。若使用電話號碼登入，按官方文件在 `futu-opend.service` 加入正確 `-area_code=...`，不要把電話登入格式當成交易帳戶 ID。

`NDXBOTS_PAPER_ACCOUNT_ID` 必須稍後填入 `accounts` 回傳、由你核對的 US SIMULATE 帳戶 ID。範例故意沒有預填可交易帳戶。這兩個設定值都不是密碼。

```bash
sudo install -m 0644 /opt/ndxbots/deploy/futu-opend.service /etc/systemd/system/futu-opend.service
sudo install -m 0644 /opt/ndxbots/deploy/ndxbots-paper.service /etc/systemd/system/ndxbots-paper.service
sudo systemd-analyze verify /etc/systemd/system/futu-opend.service /etc/systemd/system/ndxbots-paper.service
sudo systemctl daemon-reload
sudo systemctl start futu-opend.service
sudo journalctl -u futu-opend.service -n 60 --no-pager
```

`After=` 只規定啟動順序，不代表 OpenD 已登入。繼續前先確認帳戶查詢成功：

```bash
sudo -u ndxbots -H bash
cd /opt/ndxbots
.venv/bin/python -m ndxbots.paper accounts
exit
```

將核對過的模擬 `acc_id` 寫入 `/etc/ndxbots/paper.env`。若缺行情權限、帳戶未啟用或沒有適用模擬帳戶，先在 OpenD／富途介面處理；不要換用另一個環境的帳戶。

## 5. 先預覽，再驗證模擬執行

以相同使用者載入已檢視的服務環境檔，保證手動驗證與常駐服務使用相同帳戶：

```bash
sudo -u ndxbots -H bash
cd /opt/ndxbots
set -a
. /etc/ndxbots/paper.env
set +a
.venv/bin/python -m ndxbots.paper prepare --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID" --refresh
.venv/bin/python -m ndxbots.paper status --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID"
.venv/bin/python -m ndxbots.paper tick --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID"
```

這三步不送單。核對訊號日、下一個交易日、持倉、可用資金、退出與新增方向、整股數量及預算。舊研究帳本的前復權數量不能直接當券商股數；現有帳戶的訂單及持倉也必須與計劃一致。

`prepare` 只接受最近正式交易日收盤後至少 20 分鐘、至下一個正式交易日 09:30 前的準備窗口。首次在盤中啟動且沒有計劃，worker 會等到收盤後再準備下一日；不補做當天的早盤排程。已準備計劃首次錯過應執行時間超過兩分鐘也會拒絕補做。以上均按 `America/New_York` 及官方假日／半日市日曆。

指定帳戶確認後，在計劃執行日開盤前啟動，並於美股交易時段觀察首次模擬執行：

```bash
.venv/bin/python -m ndxbots.paper run --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID" --submit --refresh --poll-seconds 10
```

驗證項目：

- 券商介面及程式都指向同一 US SIMULATE 帳戶。
- 新增訂單、部分／全部成交、撤單後剩餘數量與持倉能互相核對；不能只看到訂單 ID 就標示完成。
- 有符合條件的計劃時，09:30 緊急退出、09:45 普通退出、10:00 新增的順序正確；不利跳空分批的剩餘限價及收盤前五分鐘處理沒有重複訂單。
- 程序停止並重啟後，已有訂單得到對帳，沒有再送同一意圖。尚未觀察到的部分成交／半日市等場景，要記作離線測試覆蓋，不能標為已通過券商整合。

Futu 模擬盤支援限價、市價及 DAY 訂單；不支援成交明細查詢／成交推送及訂單費用查詢。應用訂單累計成交量、均價、持倉及資金對帳。參考 [官方模擬交易限制](https://openapi.futunn.com/futu-api-doc/en/qa/trade.html)、[下單回應](https://openapi.futunn.com/futu-api-doc/en/trade/place-order.html) 及 [撤單說明](https://openapi.futunn.com/futu-api-doc/en/trade/modify-order.html)。

首次觀察結束後，Ctrl+C 停止前景 worker，再 `exit` 回到管理使用者。停止程序不保證券商掛單已取消，先檢查其狀態。完成上述檢查後，才啟用常駐服務：

```bash
sudo systemctl enable --now futu-opend.service
sudo systemctl enable --now ndxbots-paper.service
sudo systemctl status ndxbots-paper.service --no-pager
sudo journalctl -u ndxbots-paper.service -n 100 --no-pager
```

自動 worker 會在收盤後至少二十分鐘／下次開盤前建立下一份計劃；輪詢本身不代表每十秒重新選股或抓取全量歷史資料。所有執行時間以美東市場時區及交易日曆為準，不以伺服器或香港時鐘硬編碼。

## 6. 停止、查看與恢復

以下命令在 `/opt/ndxbots`、`ndxbots` 使用者及已載入環境檔的 shell 執行：

```bash
.venv/bin/python -m ndxbots.paper status --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID"
.venv/bin/python -m ndxbots.paper stop --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID"
```

`stop` 持久保存該帳戶的停止旗標。它不是「所有券商掛單已撤銷」的證明；檢查程式狀態及券商介面，對已有訂單逐一確認。要停掉常駐程序，由管理使用者執行：

```bash
sudo systemctl stop ndxbots-paper.service
```

準備恢復時，先對帳及處理不明訂單，再以服務使用者清除停止旗標：

```bash
.venv/bin/python -m ndxbots.paper resume --acc-id "$NDXBOTS_PAPER_ACCOUNT_ID"
```

若服務已停止，由管理使用者再次 `sudo systemctl start ndxbots-paper.service`。不要為了繞過不明訂單刪除 SQLite 狀態庫或重建空白帳戶狀態。

## 7. 更新與備份

停止 worker、檢查券商現有訂單，並在沒有程序寫入時備份整個 `data/`、策略設定和目前部署 commit。帳戶狀態與停止旗標一起保留。OpenD 的登入狀態及環境檔屬私人伺服器資料，不要加入程式倉庫。

切換到另一個已檢視的 commit 後，重新安裝依賴、執行離線測試，再做 `status`／不含 `--submit` 的 `tick` 查核，最後重啟 worker。已不明的送單、公司行動或持倉差異需要先核對，不能靠升級或重啟直接清除。

策略設定變更另外處理：既有帳本會一直核對原計劃的策略設定指紋，即使上一份計劃已完成也一樣。修改 `config.yaml` 的策略參數後，執行及建立下一份計劃都會停止。先停止 worker、對帳及備份，再另行安排人工受控遷移；本版本沒有自動遷移命令，不要刪除帳本或修改指紋繞過檢查。STOP 撤單流程仍可在策略設定不一致時處理自己已確認的訂單。

需要核對 Linux 服務行為時，參考 [systemd.service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)、[systemd.exec](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html) 及 [systemd.unit](https://www.freedesktop.org/software/systemd/man/latest/systemd.unit.html)。
