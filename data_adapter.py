# ==========================================
# 数据适配器模块 - 处理真实数据导入和模板生成
# 功能：将用户上传的 CSV/Excel 文件转换为标准课程数据格式
# ==========================================
import pandas as pd
import uuid
import re

def parse_chinese_time_constraint(text):
    """
    支持两类输入：
      1) 明确时间段：周三08:00-10:00不可用 / 周三 08:00-10:00 不可用
      2) 语义时间段：周三上午不可用 / 周三下午不可用 / 周三晚上不可用

    输出：标准时间槽列表，例如 ["Wed_08:00", "Wed_14:00"]
    系统标准时间槽：08:00, 10:00, 14:00, 16:00, 19:00
    """
    if not text or str(text) == "nan" or "无" in str(text):
        return []

    s = str(text).strip()

    week_map = {
        "周一": "Mon", "周二": "Tue", "周三": "Wed",
        "周四": "Thu", "周五": "Fri", "周六": "Sat", "周日": "Sun"
    }

    # 语义时间段 -> 对应“开始时间槽”
    # 你说“下午不可用有两个不可用区间”，这里按 14:00 和 16:00 两节处理
    semantic_to_slots = {
        "上午": ["08:00", "10:00"],
        "下午": ["14:00", "16:00"],
        "晚上": ["19:00"],
        "晚间": ["19:00"],
    }

    # 标准时间槽
    std_slots = ["08:00", "10:00", "14:00", "16:00", "19:00"]

    blocked = set()

    # 先按常见分隔符切片，支持 “； ; 、 , ， 换行”
    parts = re.split(r"[；;、,\n\r]+", s)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # 找到对应星期（一个 part 里可能只写了一个星期）
        hit_days = [cn for cn in week_map.keys() if cn in part]
        if not hit_days:
            continue

        for cn_day in hit_days:
            en_day = week_map[cn_day]

            # (A) 语义：上午/下午/晚上不可用
            for key, slots in semantic_to_slots.items():
                if key in part and ("不可用" in part or "不排" in part or "禁排" in part):
                    for t in slots:
                        blocked.add(f"{en_day}_{t}")

            # (B) 明确时间：直接抓出所有出现的标准时间点
            # 例如 “周三08:00-10:00不可用” 会包含 08:00 和 10:00
            for t in std_slots:
                if t in part:
                    blocked.add(f"{en_day}_{t}")

    # 返回稳定顺序（按周/按时间排序）
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    time_order = {t: i for i, t in enumerate(std_slots)}

    def sort_key(x):
        d, t = x.split("_", 1)
        return (day_order.index(d) if d in day_order else 999,
                time_order.get(t, 999))

    return sorted(blocked, key=sort_key)

def load_real_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 1. 映射字典更新：适配新数据集的列名
        col_map = {
            "name": "课程名称",
            "teacher": "教师",
            "size": "学生人数限制",      #  可选
            "type": "教室类型",
            "weeks": "周次",            # 单双周
            "major": "学生专业限制",     # 专业班级限制
            "link": "连课",             # 变更
            "block": "教师时间限制"      # 变更 支持多种表述
        }
        
        # 周次值的映射
        week_val_map = {
            "每周": "all",
            "单周": "odd",
            "双周": "even",
            "all": "all", "odd": "odd", "even": "even" # 兼容旧格式
        }

        standard_courses = []
        
        for index, row in df.iterrows():
            # 获取基础信息
            c_name = row.get(col_map["name"], f"课程_{index}")
            c_teacher = row.get(col_map["teacher"], "待定")
            c_size = row.get(col_map["size"], 40)
            
            # 教室类型处理
            raw_type = str(row.get(col_map["type"], "multimedia"))
            
            # 先排除"非"字，再判断"实验"或"机房"
            if "非" in raw_type:
                c_type = "multimedia"
            elif "实验" in raw_type or "机房" in raw_type:
                c_type = "lab"
            else:
                c_type = "multimedia"
            
            # 周次处理 (中文转英文代码)
            raw_weeks = str(row.get(col_map["weeks"], "每周")).strip()
            c_weeks = week_val_map.get(raw_weeks, "all")
            
            # 适用班级处理
            raw_majors = row.get(col_map["major"], "")
            if pd.notna(raw_majors) and str(raw_majors).strip():
                # 支持中文逗号和英文逗号
                c_majors = [m.strip() for m in str(raw_majors).replace("，", ",").split(",")]
            else:
                c_majors = ["全校通选"]
                
            # 教师禁排时间处理
            raw_block = row.get(col_map["block"], "")
            c_blocks = parse_chinese_time_constraint(raw_block)
            
            # 连堂课处理
            # 逻辑：如果"连课"为1，我们将这行数据生成两个课程对象，并赋予相同的 link_id
            is_linked = row.get(col_map["link"], 0)
            
            try:
                is_linked = int(is_linked)
            except:
                is_linked = 0
                
            if is_linked == 1:
                # 生成一个唯一的 link_id
                unique_link_id = f"L_{index}_{uuid.uuid4().hex[:4]}"
                
                # 创建第一节课
                standard_courses.append({
                    "id": f"real_{index}_1",
                    "name": c_name, # 或者 c_name + "_1"
                    "teacher": c_teacher,
                    "size": int(c_size),
                    "type": c_type,
                    "weeks": c_weeks,
                    "class_groups": c_majors,
                    "link_id": unique_link_id,
                    "blocked_hours": c_blocks
                })
                
                # 创建第二节课 (连堂的后半部分)
                standard_courses.append({
                    "id": f"real_{index}_2",
                    "name": c_name, # 或者 c_name + "_2" 可选区分名字
                    "teacher": c_teacher,
                    "size": int(c_size),
                    "type": c_type,
                    "weeks": c_weeks,
                    "class_groups": c_majors,
                    "link_id": unique_link_id,
                    "blocked_hours": c_blocks
                })
            else:
                # 普通单节课
                standard_courses.append({
                    "id": f"real_{index}",
                    "name": c_name,
                    "teacher": c_teacher,
                    "size": int(c_size),
                    "type": c_type,
                    "weeks": c_weeks,
                    "class_groups": c_majors,
                    "link_id": None,
                    "blocked_hours": c_blocks
                })
            
        return standard_courses, None
        
    except Exception as e:
        import traceback
        return [], f"数据解析失败: {str(e)}\n{traceback.format_exc()}"

def generate_empty_template():
    # 更新模板以匹配新格式
    df = pd.DataFrame(columns=[
        "课程名称", "教师", "学生人数限制", "教室类型", 
        "周次", "学生专业限制", "连课", "教师时间限制"
    ])
    df.loc[0] = [
        "示例课程", "王老师", 60, "非实验室", 
        "每周", "自动化,电气", 1, "周三08:00-10:00不可用"
    ]
    return df.to_csv(index=False).encode('utf-8')