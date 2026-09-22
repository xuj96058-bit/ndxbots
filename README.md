# ndxbots

纳斯达克 100 本地量化平台：从富途拉日 K，算因子，生成观察池，再和 QQQ 做日线回测。

行情和回测结果只存在你电脑的 `data/` 里，**不要上传到 GitHub**。

## 你需要先有的

1. Python 3.10+
2. 已安装并登录 [FutuOpenD](https://www.futunn.com/download/openAPI)（默认 `127.0.0.1:11111`）
3. 富途账号已开通 **美股行情**

## 安装（Windows）

```powershell
cd ndxbots
python -m venv .venv
.\.venv\Scripts\activate
pip install -e .
```

Mac / Linux 把激活换成 `source .venv/bin/activate`。

改连接或选股参数，编辑项目根目录的 `config.yaml`，不必改代码。

## 每天怎么跑

按顺序执行。前一步没成功，后面会找不到文件。

### 1. 测连接

```powershell
python -m ndxbots.data.check
```

能打出 OpenD 地址和纳指 100 板块就说明通了。失败时先看 OpenD 是否启动、端口是否 11111、美股行情权限。

### 2. 拉日 K（第一次先试 2 只）

```powershell
python -m ndxbots.data.ingest --codes US.AAPL US.QQQ
python -m ndxbots.data.inspect_panel
```

全市场成分 + QQQ：

```powershell
python -m ndxbots.data.ingest
```

第一次全量大约 10–20 分钟，保持 OpenD 开着。之后每天收盘后再跑同一条，只补缺的日期。

| 参数 | 作用 |
|---|---|
| `--codes US.AAPL US.QQQ` | 只拉指定代码 |
| `--limit 5` | 只拉前 5 只，试限频 |
| `--full-refresh` | 忽略本地缓存，整段重拉 |

价格为 **前复权**。成分优先用富途「纳斯达克100 / NDX」板块；搜不到就用内置兜底名单。

### 3. 算因子

```powershell
python -m ndxbots.factors.compute
```

会写出 `data/factors/daily.parquet`，包含动量、波动、RSI，以及 `custom.py` 里的 `my_*` 因子。

自己加因子：编辑 `src/ndxbots/factors/custom.py`，再重新 compute。

可选，批量看哪些因子更能预测未来 21 日是否跑赢 QQQ：

```powershell
python -m ndxbots.factors.mine --horizon 21
```

### 4. 生成观察池

```powershell
python -m ndxbots.strategy.build
```

默认规则（均可在 `config.yaml` 的 `strategy` 段改）：

- 用 `my_ma50_gap`、`my_struct_gap` 截面排名打分
- 收盘必须在日线 MA200 上方
- 离 21 日高点回撤大约在 1%–12% 之间（避免追顶或结构坏掉）
- ATR 太小不做
- 观察池 Top 10，真正持仓最多 4 只

执行层只许盯池内股票。

### 5. 回测（对齐 QQQ）

```powershell
python -m ndxbots.backtest
```

T 日收盘定池，权重作用在 T+1 收益上（`next_close`），单边成本默认 10 bp，初始资金 20000。摘要会打印到屏幕，并写入 `data/backtest/stats.txt`。

## 目录

```text
config.yaml                 连接、区间、选股、回测参数
src/ndxbots/data/           拉行情、检查、面板
src/ndxbots/factors/        算因子、挖因子
src/ndxbots/strategy/       打分、观察池
src/ndxbots/backtest/       日线回测
ndxbots_patch/              历史补丁副本（源码已合并进 src/，一般不用再拷）

data/                       本机生成，已加入 .gitignore
  raw/kline/                每只股票一个 parquet
  panel/daily.parquet       对齐后的长表
  meta/universe.csv         成分来源
  meta/us_trading_days.parquet
  factors/daily.parquet
  mining/factor_ranking.csv
  strategy/scores.parquet
  strategy/observation_pool.csv
  backtest/equity.csv
  backtest/holdings.csv
  backtest/stats.txt
```

面板字段大致为：`date, code, open, high, low, close, volume, turnover, pe_ratio, ...`。代码带 `US.` 前缀。

## 常用配置

`config.yaml` 里这几项最常改：

```yaml
data:
  kline_start: "2016-01-01"   # 下载起点
  # kline_end: "2026-09-01"   # 不写 = 拉到最新

strategy:
  factors: [my_ma50_gap, my_struct_gap]
  top_n: 10
  max_hold: 4
  require_above_ma200: true

backtest:
  start: "2018-01-01"         # 回测起点，更早的数据留给 MA200 预热
  cost_bps: 10
  initial_cash: 20000
```

临时覆盖（可选）：环境变量 `FUTU_HOST`、`FUTU_PORT`、`KLINE_START`、`KLINE_END`、`DATA_DIR`。

## 不要提交到 GitHub

- `.venv/` 虚拟环境
- `.env` 密钥
- `data/` 行情和回测结果
- `__pycache__/`、`.idea/`

`.gitignore` 已经挡住这些。用网页拖文件时，请不要选择它们。
