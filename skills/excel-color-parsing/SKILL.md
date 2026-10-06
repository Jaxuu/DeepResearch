---
name: excel-color-parsing
description: 当任务依赖 Excel 单元格的填充颜色（如“绿色格子代表 X”、“蓝色格子是障碍物”）时使用。只负责正确读取颜色，不负责后续算法。
---

# Excel 单元格颜色解析

## 依赖的底层工具 (Required Tools)
- `execute_python_code`: **(核心)** 用 `openpyxl` 读取单元格填充色。
- ⚠️ **禁止**使用 `inspect_structured_data`——它只读文本，丢失颜色信息。

## 关键要点

### 1. 网格范围必须动态确定
```python
ws = wb.active
max_r, max_c = ws.max_row, ws.max_column   # 禁止硬编码 30x30
```
### 2. 颜色读取必须处理 theme+tint
Excel 有两种颜色存储方式，只读 rgb 会漏掉主题色填充的格子：
```python
THEME_RGB = {0:"FFFFFF",1:"000000",2:"EEECE1",3:"1F497D",4:"4F81BD",
             5:"C0504D",6:"9BBB59",7:"8064A2",8:"4BACC6",9:"F79646"}

def _apply_tint(hex_rgb, tint):
    if not tint: return hex_rgb
    r,g,b = (int(hex_rgb[i:i+2],16) for i in (0,2,4))
    if tint < 0: r,g,b = (int(c*(1+tint)) for c in (r,g,b))
    else:        r,g,b = (int(c*(1-tint)+255*tint) for c in (r,g,b))
    return f"{r:02X}{g:02X}{b:02X}"

def get_cell_hex(cell):
    fill = cell.fill
    if fill is None or fill.patternType is None: return None
    fg = fill.fgColor
    if fg is None: return None
    if fg.type == "rgb" and fg.rgb and fg.rgb not in ("00000000","FFFFFFFF"):
        return fg.rgb[-6:].upper()
    if fg.type == "theme" and fg.theme in THEME_RGB:
        return _apply_tint(THEME_RGB[fg.theme], fg.tint or 0.0)
    return None
```
### 3. 颜色阈值不要拍脑袋
题目说"绿色"或"蓝色"时，不要写死 R<50 and G<50 and B>150 这类阈值。
- 先 dump 所有非默认底色格子的 hex，去重后 print 出来看。
- 再根据聚类结果确定目标色系的阈值。
- 宽松判据示例：蓝色系可用 B > R and B > G and B > 100。
### 4. 输出
用 print() 输出明确的颜色类别和坐标，例如：
```python
print(f"Found {len(green_cells)} green cells at: {sorted(green_cells)}")
print(f"Found {len(blue_cells)} blue obstacle cells.")
```


## 绝对红线
- 🚫 禁止用 inspect_structured_data 处理颜色相关任务。
- 🚫 禁止硬编码行列范围。
- 🚫 禁止在 fgColor.type == "theme" 时直接跳过（会丢失整片格子）。