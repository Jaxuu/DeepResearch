---
name: data-visualization
description: 当用户要求"对比数据""看趋势""生成图表""画个图""可视化"或需要把数值呈现成图时使用。通过编写规范的 Chart.js JSON 调用绘图 API，生成高质量真实图表图片。禁止用 Markdown 表格或 ASCII 字符画代替真实图表。
---

# 专业数据可视化图表绘制

## 依赖的底层工具 (Required Tools)
- `generate_chart_image`: (必须从 MCP Server 获取) 用于将 JSON 转换为图表图片链接。

## 标准作业流程 (SOP)
1. **数据校验**：确保你已经通过网页搜索或本地文档收集到了确切的数值。绝对禁止使用虚构或估算的数据。
2. **选择图表类型**：
   - 趋势变化 -> 使用 `line` (折线图)
   - 对比大小 -> 使用 `bar` (柱状图)
   - 占比分析 -> 使用 `pie` 或 `doughnut` (饼图/环形图)
3. **编写图表配置**：构思符合 Chart.js 格式的 JSON。JSON 必须包含 `type` 和 `data` 节点。
4. **执行生成**：调用 `generate_chart_image` 工具，将原始数据和图表生成目标传给它。
5. **最终交付**：将工具返回的 Markdown 图片链接（如 `![alt](url)`）原封不动地输出到最终报告中。

## 绝对红线 (What NOT to do)
- 🚫 **绝对禁止**使用 Markdown 表格、ASCII 字符（如 `|---|`、`***` 或空格）来手绘简陋的图表。
- 🚫 **绝对禁止**擅自修改或翻译生成的图片 URL 链接。