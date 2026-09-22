# ndxbots — 第一阶段：数据层

纳斯达克100本地量化平台。本阶段只做一件事：把日 K 从富途拉到本地，变成可复用的面板表。

## 你需要先有的

1. Python 3.10+
2. 已安装并登录 **FutuOpenD**（默认 `127.0.0.1:11111`）
3. 富途账号已开通 **美股行情**

## 安装

```bash
cd ndxbots
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate

pip install -e .
cp .env.example .env
```

## 三步走

### 1. 测连接

```bash
python -m ndxbots.data.check
```

成功会看到类似：

```text
OK  AAPL 2026-09-18 close=...
```

若这里失败，先不要下载。检查 OpenD 是否启动、端口是否 11111、美股行情权限。

### 2. 先试拉 2 只（强烈建议）

```bash
python -m ndxbots.data.ingest --codes US.AAPL US.QQQ
```

完成后看：

- `data/raw/kline/US_AAPL.parquet`
- `data/raw/kline/US_QQQ.parquet`
- `data/panel/daily.parquet`

### 3. 拉全市场成分 + QQQ

```bash
python -m ndxbots.data.ingest
```

程序会先在富途板块里搜「纳斯达克100 / NDX」。搜到就用官方成分；搜不到就用内置兜底名单。第一次全量大约要 10–20 分钟，请保持 OpenD 开着。

之后每天收盘后再跑同一条命令，只补缺的日期。

整段重拉：

```bash
python -m ndxbots.data.ingest --full-refresh
```

只跑前 5 只试限频：

```bash
python -m ndxbots.data.ingest --limit 5
```

## 目录约定

```text
data/
  raw/kline/          每只股票一个 parquet
  panel/daily.parquet 全部对齐后的长表
  meta/universe.csv   成分来源
  meta/us_trading_days.parquet
```

面板字段：`date, code, open, high, low, close, volume, turnover, pe_ratio, ...`  
价格为 **前复权**，用来算收益和动量。

## 本阶段过关标准

- `check` 能连上 OpenD
- AAPL 和 QQQ 有 2016 至今的日 K
- `daily.parquet` 能用 pandas 读出来，日期连续、代码带 `US.` 前缀

下一阶段才会写因子。现在不要加策略。
