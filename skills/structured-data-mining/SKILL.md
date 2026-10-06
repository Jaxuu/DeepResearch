---
name: structured-data-mining
description: 当用户要求从 Excel (.xlsx)、CSV 或 JSON 等表格结构化文件中计算数据、查找特定行、求和或做统计分析时使用。
---

# 结构化数据深度挖掘

## 依赖的底层工具 (Required Tools)
- `inspect_structured_data`: 用于在写代码前“看一眼”表格的结构。
- `execute_python_code`: 用于真正执行计算。

## 标准作业流程 (SOP)
1. **盲写代码是大忌**：在面对 Excel/CSV 文件时，必须第一步先调用 `inspect_structured_data` 探查文件的 Sheet 名称和真实的列名。
   - ⚠️ **极其重要**：`inspect_structured_data` 只会返回前 3 行的预览数据！**绝对禁止**你自己对这 3 行预览数据进行求和或计算来作为最终结果。你必须使用查到的列名，编写 Python 代码去读取完整的文件进行计算！
2. **编写精准脚本**：拿到真实的列名后，调用 `execute_python_code`，使用 `pandas` 编写脚本。确保打印 (`print`) 出最终答案。
3. **容错处理**：如果 Python 报错“KeyError (找不到列)”，重新查阅 `inspect_structured_data` 的输出，检查是否带有空格或特殊字符。