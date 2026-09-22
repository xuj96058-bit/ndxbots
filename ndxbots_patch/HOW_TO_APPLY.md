# 第一版融合补丁怎么接到现有 ndxbots

把本目录里的文件拷到你的项目根（`C:\Users\xuj96\Downloads\ndxbots`），覆盖同名文件。

新增的文件直接放进去即可：

```text
config.yaml                          覆盖（多了 strategy / backtest 两段）
src/ndxbots/config.py                覆盖
src/ndxbots/factors/custom.py        覆盖
src/ndxbots/factors/compute.py       覆盖
src/ndxbots/factors/__init__.py      覆盖
src/ndxbots/strategy/__init__.py     新建
src/ndxbots/strategy/__main__.py     新建
src/ndxbots/strategy/scores.py       新建
src/ndxbots/strategy/build.py        新建
src/ndxbots/backtest/__init__.py     新建
src/ndxbots/backtest/__main__.py     新建
src/ndxbots/backtest/engine.py       新建
src/ndxbots/backtest/run.py          新建
```

不要动 `data/`、`src/ndxbots/data/`、`factors/price.py`、`factors/mine.py`。

然后：

```powershell
.\.venv\Scripts\activate
python -m ndxbots.factors.compute
python -m ndxbots.factors.mine --horizon 21
python -m ndxbots.strategy.build
python -m ndxbots.backtest
```

产出：

```text
data/factors/daily.parquet           含 my_* 因子
data/strategy/scores.parquet         每天每档分数
data/strategy/observation_pool.csv   最新观察池（明天盯这些）
data/backtest/equity.csv             组合 vs QQQ
data/backtest/holdings.csv           历史持仓
data/backtest/stats.txt              摘要
```
