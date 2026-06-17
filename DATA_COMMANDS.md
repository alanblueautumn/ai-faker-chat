# 数据重建与使用命令

本项目的 `data/` 目录不要提交到 git。它包含聊天记录衍生数据、向量索引和用户画像，可以随时从原始聊天 txt 重新生成。

## 1. 原始文件

默认原始聊天记录文件是 `.md`：

```bash
杭州牛马二群_chat.txt
```

如果你的原始文件名不同，把下面命令里的文件名替换成自己的 `.md` 聊天记录路径。

## 2. 扫描聊天记录

```bash
uv run python scripts/scan_chat.py 杭州牛马二群_chat.txt --top-users 50
```

用途：

- 查看解析成功率
- 查看用户排行
- 查看异常行样例

## 3. 生成基础数据

把原始聊天记录转成结构化消息：

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

`messages.jsonl` 会保留 Markdown 原始发言时间：

```json
{"id":1,"line_no":1,"time":"16:33:28","user":"安达桜","content":"跟新出的15号电极柱比呢"}
```

`windows.jsonl` 的上下文文本也会带上时间，方便向量检索和 AI prompt 保留对话节奏。

## 4. 生成用户相似度报告

默认会清洗低信息和复读内容，并使用 `BAAI/bge-large-zh-v1.5`：

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

默认会清洗低信息和复读目标消息，并使用 `BAAI/bge-large-zh-v1.5`。关闭清洗：

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

默认使用 `deepseek-v4-pro`。

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

默认使用 `deepseek-v4-pro`。

```bash
uv run python scripts/generate_group_profile.py
```

生成结果：

```text
data/profiles/group.md
```

## 9. 生成高精度用户画像

默认使用 `deepseek-v4-pro`，会综合清洗后代表发言、统计特征和上下文窗口，输出简洁版画像。

```bash
uv run python scripts/generate_deep_profile.py --user "👴🍼👶"
```

更高精度：

```bash
uv run python scripts/generate_deep_profile.py --user "👴🍼👶" --sample-limit 800 --window-limit 120
```

输出结构：

```text
简要概括
说话习惯
典型表达
不要这样生成
模仿指令
```

生成结果在：

```text
data/profiles/*.deep.md
```

## 10. 单次模拟回复

默认使用 `deepseek-v4-flash`。

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

## 11. 交互式聊天

默认使用 `deepseek-v4-flash`。

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

## 12. AI 版经典语录

默认使用 `deepseek-v4-pro`。

```bash
uv run python scripts/user_quotes_ai.py --user "👴🍼👶"
```

更高精度：

```bash
uv run python scripts/user_quotes_ai.py --user "👴🍼👶" --max-candidates 600 --before 5 --after 3
```

默认输出最经典 5 句，并展示出自聊天行号。

## 13. 群友关系图谱

先从基础 JSON 数据构建关系图谱：

```bash
uv run python scripts/build_group_graph.py data/messages.jsonl --out data/reports/group_graph.json
```

再渲染成 HTML：

```bash
uv run python scripts/render_group_graph.py --graph data/reports/group_graph.json --out data/reports/group_graph.html
```

生成 AI 解释层：

```bash
uv run python scripts/enrich_group_graph_ai.py --graph data/reports/group_graph.json --messages data/messages.jsonl --out data/reports/group_graph_ai.json
```

带 AI 解释渲染 HTML：

```bash
uv run python scripts/render_group_graph.py --graph data/reports/group_graph.json --ai data/reports/group_graph_ai.json --out data/reports/group_graph.html
```

如果图太密，可以提高最小连接强度：

```bash
uv run python scripts/build_group_graph.py data/messages.jsonl --out data/reports/group_graph.json --min-weight 50
```

HTML 默认展示权重最高的 300 条边，可以调整：

```bash
uv run python scripts/render_group_graph.py --graph data/reports/group_graph.json --out data/reports/group_graph.html --max-edges 500
```

图谱说明：

```text
LIKELY_REPLY / INTERACTS_WITH 都是基于消息顺序推断，不代表明确回复。
边权重综合了近邻接话、窗口共现和语气规则统计。
```

## 14. 群聊常见话题

先从上下文窗口聚类出基础话题：

```bash
uv run python scripts/build_group_topics.py --windows data/windows.jsonl --out data/reports/group_topics_base.json
```

生成 AI 话题解释：

```bash
uv run python scripts/enrich_group_topics_ai.py --topics data/reports/group_topics_base.json --out data/reports/group_topics_ai.json
```

渲染 HTML：

```bash
uv run python scripts/render_group_topics.py --topics data/reports/group_topics_base.json --ai data/reports/group_topics_ai.json --out data/reports/group_topics.html
```

调整话题数量：

```bash
uv run python scripts/build_group_topics.py --windows data/windows.jsonl --out data/reports/group_topics_base.json --topic-count 16
```

## 15. Git 注意事项

不要提交这些文件：

```text
data/
*_chat.txt
.env
```

如果误提交了 `data/`，GitHub 可能因为大文件拒绝 push。应从 git 历史中移除 `data/`，不要用普通删除来处理历史里的大文件。
