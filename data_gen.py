# ==========================================
# 模拟数据生成模块
# 功能：根据参数动态生成虚拟的课程数据用于测试和演示
# ==========================================
import random

def generate_mock_data(
    num_courses=30, 
    prob_lab=0.3,      # 实验课比例 (0.0 - 1.0)
    prob_odd_even=0.2  # 单双周课程比例 (0.0 - 1.0)
):
    """
    根据参数动态生成模拟数据
    :param num_courses: 生成多少门课
    :param prob_lab: 多少比例是机房实验课 (Lab)
    :param prob_odd_even: 多少比例是单/双周课程
    """
    # 1. 基础时间片
    times = [f"{d}_{t}" for d in ["Mon", "Tue", "Wed", "Thu", "Fri"] 
             for t in ["08:00", "10:00", "14:00", "16:00", "19:00"]]

    # 2. 教师库
    teachers = ["王教授", "李老师", "张助教", "赵教授", "刘老师", "陈博士", "杨讲师"]
    
    # 3. 专业/班级池 (新增)
    all_majors = ["计科2301", "计科2302", "电气2301", "电气2302", "自动化2301", "AI2301"]
    
    # 4. 课程模板库
    lecture_names = ["AI导论", "离散数学", "高等数学", "通信原理", "操作系统", "马克思原理", "英语写作"]
    lab_names = ["Python编程", "数据库实验", "数字电路实验", "计算机网络实验", "Web开发基础"]

    courses = []
    
    # 模拟教师忙碌时间 (新增)
    # key: teacher_name, val: list of blocked times ["Mon_08:00", ...]
    teacher_blocks = {}
    
    # 第二步：逐个生成课程
    for i in range(num_courses):
        # --- A. 决定课程类型 (Lab 实验课 vs Multimedia 理论课) ---
        if random.random() < prob_lab:
            # 生成实验课：需要专门的机房
            c_type = "lab"
            base_name = random.choice(lab_names)
        else:
            # 生成理论课：可以在多媒体教室上课
            c_type = "multimedia"
            base_name = random.choice(lecture_names)
            
        # --- B. 决定周次模式 (All 全周 vs Odd 单周 vs Even 双周) ---
        ## 单双周课程可以在同一时间和地点与其他课程复用资源
        week_mode = "all"
        if random.random() < prob_odd_even:
            week_mode = random.choice(["odd", "even"])
            
        # --- C. 决定课程的其他属性 ---
        # 实验课通常人数较少，理论课人数较多
        size = random.randint(20, 60) if c_type == "lab" else random.randint(40, 120)
        
        # 拼接课程名字（加上序号以区分）
        c_name = f"{base_name}_{i+1}"
        
        # --- D. 随机分配该课程适用的班级（行政班级不冲突约束） ---
        # 随机决定该课程有多少个班级需要参加（0.6概率1班，0.2概率2班，0.2概率3班）
        r = random.random()
        if r < 0.6:
            num_classes = 1
        elif r < 0.8:
            num_classes = 2
        else:
            num_classes = 3
        # 防护：确保随机数不超过可用专业数
        assigned_majors = random.sample(all_majors, min(num_classes, len(all_majors)))
        
        # --- E. 连堂课逻辑（需要连续排课的课程对） ---
        # 5% 概率该课程是连堂课（需要和另一门课连续排课）
        link_id = None
        if random.random() < 0.05:
            link_id = f"Link_{i}"  # 相同 link_id 的课程必须连在一起
        
        # --- F. 该课程教师的禁排时间（教师不可用时间段） ---
        blocked_hours = []
        # 30% 的课有禁排时间
        if random.random() < 0.3:  
            num_blocked = random.randint(1, 3)  # 随机选择1-3个禁排时间
            blocked_hours = random.sample(times, num_blocked)
        
        # 组装课程对象
        courses.append({
            "id": f"sim_{i}",  # 添加 "sim_" 前缀表示这是模拟数据
            "name": c_name,
            "teacher": random.choice(teachers),
            "size": size,
            "type": c_type,
            "weeks": week_mode,
            "class_groups": assigned_majors,      # 该课程涉及的班级列表
            "link_id": link_id,                   # 连堂标记
            "blocked_hours": blocked_hours        # 教师禁排时间列表
        })
    
    # 第三步：生成教师的全局禁排时间约束
    # 某些教师在特定时间段可能不可用（如定期会议、其他教学任务等）
    for teacher in teachers:
        # 30% 的教师有某些禁排时间
        if random.random() < 0.3:
            num_blocked = random.randint(1, 4)
            teacher_blocks[teacher] = random.sample(times, num_blocked)
        else:
            teacher_blocks[teacher] = []

    # 返回标准格式
    return {
        "metadata": {"times": times},
        # 注意：这里只返回课程，教室数据由前端 Sidebar 配置决定
        "courses": courses,
        "teacher_blocks": teacher_blocks  # 新增: 教师忙碌时间约束
    }