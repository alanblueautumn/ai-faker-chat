# 数据重建与使用命令

本项目的 `data/` 目录不要提交到 git。它包含聊天记录衍生数据、向量索引和用户画像，可以随时从原始聊天 txt 重新生成。

## 1. 原始文件

默认原始聊天记录文件：

```bash
杭州牛马二群_chat.txt
```

如果你的原始文件名不同，把下面命令里的文件名替换成自己的 txt 路径。

## 2. 扫描聊天记录

```bash
uv run python scripts/scan_chat.py 杭州牛马二群_chat.txt --top-users 50
```

用途：

- 查看解析成功率
- 查看用户排行
- 查看异常行样例

## 3. 生成基础数据

把原始 txt 转成结构化消息：

```bash
uv run python scripts/parse_chat.py 杭州牛马二群_chat.txt --out data/messages.jsonl
```

生成用户统计：

```bash
uv run python scripts/user_stats.py data/messages.jsonl --out data/reports/user_stats.json
```

生成上下文窗口：

```bash
uv run python scripts/build_windows.py data/messages.jsonl --out data/windows.jsonl
```

这三步会生成：

```text
data/messages.jsonl
data/reports/user_stats.json
data/windows.jsonl
```

## 4. 生成用户相似度报告

默认会清洗低信息和复读内容：

```bash
uv run python scripts/user_similarity.py data/messages.jsonl --out data/reports/user_similarity.json
```

查询某个用户最相似的前 10 个用户：

```bash
uv run python scripts/top_similar_users.py --user "👴🍼👶" --top-k 10
```

如果要关闭清洗做对比：

```bash
uv run python scripts/user_similarity.py data/messages.jsonl --out data/reports/user_similarity.raw.json --no-clean
```

## 5. 为某个用户生成 FAISS 检索索引

当前设计里 `data/vector/` 一次保存一个或一批用户的索引。生成新用户会覆盖同目录下旧索引。

为 `👴🍼👶` 生成索引：

```bash
uv run python scripts/build_index.py data/windows.jsonl --user "👴🍼👶" --out-dir data/vector
```

为 `唐小弟` 生成索引：

```bash
uv run python scripts/build_index.py data/windows.jsonl --user "唐小弟" --out-dir data/vector
```

默认会清洗低信息和复读目标消息。关闭清洗：

```bash
uv run python scripts/build_index.py data/windows.jsonl --user "👴🍼👶" --out-dir data/vector --no-clean
```

生成结果：

```text
data/vector/windows.faiss
data/vector/window_ids.json
```

## 6. 配置 DeepSeek

在项目根目录创建 `.env`：

```env
DEEPSEEK_API_KEY=你的 DeepSeek 官方 key
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

`.env` 已在 `.gitignore` 中，不要提交。

## 7. 生成用户风格档案

```bash
uv run python scripts/generate_profile.py --user "👴🍼👶"
```

减少样本数量：

```bash
uv run python scripts/generate_profile.py --user "👴🍼👶" --limit 100
```

生成结果在：

```text
data/profiles/
```

## 8. 生成群风格档案

```bash
uv run python scripts/generate_group_profile.py
```

生成结果：

```text
data/profiles/group.md
```

## 9. 单次模拟回复

默认不使用群风格：

```bash
uv run python scripts/chat_as.py --user "👴🍼👶" --context "我:这个手机能不能买"
```

使用群风格：

```bash
uv run python scripts/chat_as.py --user "👴🍼👶" --group-profile data/profiles/group.md --context "我:这个手机能不能买"
```

查看完整调试过程：

```bash
uv run python scripts/chat_as.py --user "👴🍼👶" --debug --context "我:这个手机能不能买"
```

## 10. 交互式聊天

默认不使用群风格：

```bash
uv run python scripts/talk_as.py --user "👴🍼👶"
```

使用群风格：

```bash
uv run python scripts/talk_as.py --user "👴🍼👶" --group-profile data/profiles/group.md
```

交互命令：

```text
/exit   退出
/reset  清空当前上下文
```

## 11. AI 版经典语录

```bash
uv run python scripts/user_quotes_ai.py --user "👴🍼👶"
```

更高精度：

```bash
uv run python scripts/user_quotes_ai.py --user "👴🍼👶" --max-candidates 600 --before 5 --after 3
```

默认输出最经典 5 句，并展示出自聊天行号。

## 12. Git 注意事项

不要提交这些文件：

```text
data/
*_chat.txt
.env
```

如果误提交了 `data/`，GitHub 可能因为大文件拒绝 push。应从 git 历史中移除 `data/`，不要用普通删除来处理历史里的大文件。
