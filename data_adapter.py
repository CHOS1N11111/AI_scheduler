import pandas as pd
import random

def load_real_data(uploaded_file):
    """
    读取用户上传的 CSV/Excel，并转换为标准格式
    假设上传的文件有以下列：课程名称, 教师, 人数, 教室类型, 周次要求
    """
    try:
        # 尝试读取 CSV 或 Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 标准化列名 (防止用户上传的文件列名不一样)
        # 这里做一个简单的映射，你们根据实际下载的表格调整
        required_columns = {
            "课程名称": "name",
            "教师": "teacher",
            "人数": "size",
            "教室类型": "type", # 期望值为 "multimedia" 或 "lab"
            "周次": "weeks",    # 期望值为 "all", "odd", "even"
            "适用专业": "major_limit",    # 对应列名，如 "计科,自动化"
            "连堂标记": "link_id",        # 如 "A1", "A1" 代表这两门连堂
            "教师禁排": "blocked_hours"   # 如 "Mon_08:00;Tue_10:00"
        }
        
        # 简单清洗数据
        standard_courses = []
        for index, row in df.iterrows():
            # 如果表格里没有某些列，给予默认值
            c_name = row.get("课程名称", f"未知课程_{index}")
            c_teacher = row.get("教师", "待定教师")
            c_size = row.get("人数", 40)
            c_type = row.get("教室类型", "multimedia") # 默认为多媒体
            c_weeks = row.get("周次", "all")
            
            # 类型映射 (根据你们学校的实际写法修改)
            if "机房" in str(c_type) or "实验" in str(c_type):
                c_type = "lab"
            else:
                c_type = "multimedia"
            
            # 解析专业 (假设用逗号分隔)
            raw_majors = row.get("适用专业", "全校通选")
            c_majors = [m.strip() for m in str(raw_majors).split(",")] if raw_majors and str(raw_majors) != "nan" else ["全校通选"]
            
            # 解析禁排 (假设分号分隔)
            raw_blocks = row.get("教师禁排", "")
            c_blocks = [b.strip() for b in str(raw_blocks).split(";") if b.strip()] if raw_blocks and str(raw_blocks) != "nan" else []
            
            # 获取连堂标记
            c_link_id = row.get("连堂标记", None)
            if c_link_id and str(c_link_id) != "nan":
                c_link_id = str(c_link_id).strip()
            else:
                c_link_id = None
                
            standard_courses.append({
                "id": f"real_{index}",
                "name": c_name,
                "teacher": c_teacher,
                "size": int(c_size),
                "type": c_type,
                "weeks": c_weeks,
                "class_groups": c_majors,      # 适用班级列表
                "link_id": c_link_id,          # 连堂标记
                "blocked_hours": c_blocks      # 该课程教师的禁排时间
            })
            
        return standard_courses, None
        
    except Exception as e:
        return [], str(e)

def generate_empty_template():
    """生成一个空模板供下载"""
    df = pd.DataFrame(columns=["课程名称", "教师", "人数", "教室类型", "周次", "适用专业", "连堂标记", "教师禁排"])
    df.loc[0] = ["示例:大数据分析", "王教授", 60, "机房", "all", "计科,自动化", "L1", "Mon_08:00;Tue_10:00"]
    return df.to_csv(index=False).encode('utf-8')