---
name: python-code-interpreter
description: 当遇到需要解析复杂本地文件（如Excel、Word、PPTX）、执行外部现成的 Python 脚本附件，或涉及复杂财务数据、数学公式、概率统计、大量单位换算时使用。必须通过执行Python代码获取精确结果，严禁手动心算或猜测代码运行结果。
---

# Python 代码沙盒与脚本执行

## 依赖的底层工具 (Required Tools)
- `execute_python_code`: **(必须)** 这是本技能的绝对核心！只要你被分配了本技能，你**必须**在 required_tools 中包含 `execute_python_code`，否则系统将崩溃！
- `execute_local_python_script`: (可选) 用于直接在系统终端中运行附件中的 `.py` 脚本。
- `read_local_text_file`: (可选) 配合使用，用于在写代码前预览纯文本文件的结构。

## 标准作业流程 (SOP)
1. **意图判别与选路**：
   - **场景 A（执行现成脚本）**：如果用户直接提供了一个 `.py` 文件，必须**直接调用 `execute_local_python_script`** 获取输出。
   - **场景 B（需要自主写代码）**：如果是文件解析、数学计算或逻辑推演，自主编写代码并调用 `execute_python_code`。
2. **编写执行脚本 (针对场景 B)**：
   - 处理 Excel -> 必须使用 `openpyxl` 或 `pandas`
   - 处理 Word -> 必须使用 `python-docx`
   - 处理 PPT -> 必须使用 `python-pptx`。PPT 文字极度分散，必须递归 `shape.shapes`、表格 `table.rows` 及备注页。
   - **[语义扩展覆写原则 (CRITICAL)]**：当主管要求你搜索某个**生物分类、物品类别或宽泛概念**（例如 `crustaceans` 甲壳动物）时，**即使主管在任务里强硬要求了 "exact match (精确字符串匹配)"，你也绝对不能只用单一字符串写 `if 'crustaceans' in text:`！** 你必须在 Python 代码中，利用你的生物学常识，定义一个包含该概念所有常见子类、具体名称的列表（例如 `['crustacean', 'crab', 'crayfish', 'lobster', 'shrimp', 'isopod', 'prawn', 'yeti crab']`）。只要查到列表中任意一个词，即算作命中！
   - 处理大规模结构化计算、复杂财务数据、概率统计 -> 使用 `pandas` 或 `numpy`
   - 处理基础复杂公式、乘方开根号 -> 使用内置 `math`
3. **防御性编程与异常捕获 (CRITICAL)**：
   - 你的代码必须在最外层包裹 `try-except` 块，并将详尽的错误信息打印出来：`print(f"DEBUG_ERROR: {str(e)}")`。
4. **暴露输出结果 (CRITICAL)**：
   - 必须使用 `print()` 语句。打印时必须带有清晰完整的上下文描述（例如 `print(f"The number of slides mentioning crustaceans is {count}")`）。**绝对禁止**只打印一个孤立的数字，否则提取器会当做噪音忽略！
5. **自我纠错 (Self-Correction)**：遇到报错信息，必须重新调整代码并再次调用工具重试。
6. **最终交付**：将沙箱或终端返回的确切数值转化为自然语言写入最终报告中。

## 绝对红线 (What NOT to do)
- 🚫 **绝对禁止机械字面匹配**：在寻找包含某类事物的数量时，必须用 Python 代码实现常识同义词的扩展，无视上级不合理的 "exact match" 限制！
- 🚫 **绝对禁止手动心算与猜测**：所有结果必须由代码 `print` 得出。
- 🚫 **绝对禁止在 `research_topic` 中要求只返回孤立数字**：委派代码执行任务时，必须要求下属打印带上下文的完整句子。