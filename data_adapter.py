# ==========================================
# 数据适配器模块 - 处理真实数据导入和模板生成
# 功能：将用户上传的 CSV/Excel 文件转换为标准课程数据格式
# ==========================================
import pandas as pd
import random


def load_real_data(uploaded_file):
    """
    读取用户上传的 CSV/Excel 文件，并转换为排课系统的标准格式
    
    参数说明：
        uploaded_file: Streamlit 上传的文件对象
    
    返回值：
        (standard_courses, error_msg) 元组
        - standard_courses: 标准化后的课程列表，每个课程包含以下字段：
            - id: 课程唯一标识
            - name: 课程名称
            - teacher: 授课教师
            - size: 学生人数
            - type: 教室类型 ("multimedia" 或 "lab")
            - weeks: 周次模式 ("all" 全周、"odd" 单周、"even" 双周)
            - class_groups: 适用班级列表 (字符串列表)
            - link_id: 连堂标记 (相同 ID 的课程需要连续排课)
            - blocked_hours: 教师禁排时间列表 (如 ["Mon_08:00", "Tue_10:00"])
        - error_msg: 错误信息，成功时为 None
    
    假设上传的文件有以下列（用户可灵活命名）：
        - 课程名称: 课程的名称
        - 教师: 授课教师的名字
        - 人数: 班级人数
        - 教室类型: "multimedia" 或 "机房"/"实验"
        - 周次: "all" / "odd" / "even"
        - 适用专业: 多个班级用逗号分隔 (如 "计科2301,电气2302")
        - 连堂标记: 相同值代表这几门课需要连着上
        - 教师禁排: 教师无法教课的时间，用分号分隔 (如 "Mon_08:00;Tue_10:00")
    """
    try:
        # 第一步：尝试读取文件（支持 CSV 和 Excel）
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # 第二步：标准化列名（防止用户上传的文件列名不一致）
        # 创建列名到标准字段的映射字典
        required_columns = {
            "课程名称": "name",
            "教师": "teacher",
            "人数": "size",
            "教室类型": "type", # 期望值为 "multimedia" 或 "lab"
            "周次": "weeks",    # 期望值为 "all", "odd", "even"
            "适用专业": "major_limit",    # 对应列名，如 "计科,自动化"
            "连堂标记": "link_id",        # 如 "A1", "A1" 代表这两门连堂课
            "教师禁排": "blocked_hours"   # 如 "Mon_08:00;Tue_10:00"
        }
        
        # 第三步：逐行处理并清洗数据
        standard_courses = []
        for index, row in df.iterrows():
            # 如果表格里没有某些列，给予默认值
            c_name = row.get("课程名称", f"未知课程_{index}")
            c_teacher = row.get("教师", "待定教师")
            c_size = row.get("人数", 40)
            c_type = row.get("教室类型", "multimedia") # 默认为多媒体
            c_weeks = row.get("周次", "all")
            
            # 教室类型映射：识别"机房"或"实验"关键词，标准化为 "lab"
            if "机房" in str(c_type) or "实验" in str(c_type):
                c_type = "lab"
            else:
                c_type = "multimedia"
            
            # 解析适用专业：多个班级用逗号分隔
            raw_majors = row.get("适用专业", "全校通选")
            c_majors = [m.strip() for m in str(raw_majors).split(",")] if raw_majors and str(raw_majors) != "nan" else ["全校通选"]
            
            # 解析禁排时间：多个时间用分号分隔 (如 "Mon_08:00;Tue_10:00")
            raw_blocks = row.get("教师禁排", "")
            c_blocks = [b.strip() for b in str(raw_blocks).split(";") if b.strip()] if raw_blocks and str(raw_blocks) != "nan" else []
            
            # 获取连堂标记：相同标记的课程需要连续排课
            c_link_id = row.get("连堂标记", None)
            if c_link_id and str(c_link_id) != "nan":
                c_link_id = str(c_link_id).strip()
            else:
                c_link_id = None
                
            # 组装标准格式的课程对象
            standard_courses.append({
                "id": f"real_{index}",  # 添加 "real_" 前缀以区分真实数据
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
        # 异常捕获：文件格式错误或其他问题
        return [], str(e)

def generate_empty_template():
    """
    生成一个空的 CSV 模板供用户下载，用于填写真实课程数据
    
    返回值：
        CSV 文件的字节串，可直接用于下载
    """
    # 创建包含所有所需列的空 DataFrame
    df = pd.DataFrame(columns=["课程名称", "教师", "人数", "教室类型", "周次", "适用专业", "连堂标记", "教师禁排"])
    # 添加一行示例数据帮助用户理解填写格式
    df.loc[0] = ["示例:大数据分析", "王教授", 60, "机房", "all", "计科,自动化", "L1", "Mon_08:00;Tue_10:00"]
    # 转换为 CSV 字符串并编码为 UTF-8
    return df.to_csv(index=False).encode('utf-8')