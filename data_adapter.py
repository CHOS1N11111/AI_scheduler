# ==========================================
# 数据适配器模块 - 处理真实数据导入和模板生成
# 功能：将用户上传的 CSV/Excel 文件转换为标准课程数据格式
# ==========================================
import pandas as pd
import uuid

def parse_chinese_time_constraint(text):
    """
    解析中文时间限制，例如 "周三08:00-10:00不可用" -> ["Wed_08:00"]
    假设系统标准时间槽为: 08:00, 10:00, 14:00, 16:00, 19:00
    """
    if not text or str(text) == "nan" or "无" in str(text):
        return []

    # 星期映射
    week_map = {
        "周一": "Mon", "周二": "Tue", "周三": "Wed", 
        "周四": "Thu", "周五": "Fri", "周六": "Sat", "周日": "Sun"
    }
    
    # 简单解析逻辑 (根据你的数据特征定制)
    # 示例格式: "周三08:00-10:00不可用"
    blocked_slots = []
    
    for cn_day, en_day in week_map.items():
        if cn_day in text:
            # 提取时间部分 (简单匹配)
            if "08:00" in text: blocked_slots.append(f"{en_day}_08:00")
            if "10:00" in text: blocked_slots.append(f"{en_day}_10:00")
            if "14:00" in text: blocked_slots.append(f"{en_day}_14:00")
            if "16:00" in text: blocked_slots.append(f"{en_day}_16:00")
            if "19:00" in text: blocked_slots.append(f"{en_day}_19:00")
            
    return blocked_slots

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
            "size": "学生人数限制",      # 变更
            "type": "教室类型",
            "weeks": "周次",
            "major": "学生专业限制",     # 变更
            "link": "连课",             # 变更
            "block": "教师时间限制"      # 变更
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
            
            # 【修复】先排除"非"字，再判断"实验"或"机房"
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