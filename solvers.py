# ==========================================
# 排课求解器模块 - 实现三种排课算法
# 
# 提供三个求解器类：
#   1. GreedySolver (贪心算法) - 快速但可能次优
#   2. GASolver (遗传算法) - 平衡速度和质量
#   3. CPSATSolver (OR-Tools精确建模) - 最优但可能较慢
# ==========================================
import random
import numpy as np
import copy
from ortools.sat.python import cp_model
import collections
# ==========================================
# 辅助函数：判断两个周次模式是否冲突
# ==========================================
def weeks_conflict(w1, w2):
    if w1 == "all" or w2 == "all": return True
    return w1 == w2  # odd vs odd = True, odd vs even = False

# ==========================================
# 1. 贪心算法求解器 (Greedy Baseline)
# ==========================================
# 策略：按课程人数从大到小排列，依次为每门课程分配教室和时间
# 优点：速度快，适合快速演示
# 缺点：结果可能不是全局最优
# ==========================================
class GreedySolver:
    def __init__(self, data):
        """
        初始化贪心求解器
        
        参数：
            data: 包含元数据、教室列表和课程列表的字典
                - metadata: {"times": [...]}  所有可用时间片
                - rooms: 教室列表
                - courses: 课程列表
        """
        self.data = data
        
    def solve(self):
        """
        执行贪心排课算法
        
        返回值：
            (schedule, msg) 元组
            - schedule: 排课结果列表，每个元素包含 course、time、room 信息
            - msg: 算法执行的简要说明信息
        """
        schedule = []
        # 修正1: 占用表的值改为列表，存储该时空下所有已占用的周次
        # key=(time, room), val=[week_pattern_1, week_pattern_2, ...]
        occupied = {} 
        # key=(time, teacher), val=[week_pattern_1, ...]
        teacher_busy = {}
        # [新增] 专业/班级占用表：记录各班级各时间的课程占用
        # key=(time, major), val=[week_pattern_1, ...]
        major_busy = {}
        # [新增] 追踪已分配的课程（用于连堂检查）
        # key=link_id, val=[course_obj1, course_obj2, ...]
        linked_courses = collections.defaultdict(list)

        # 策略：按人数从大到小排序，先排难排的（人数多、要求严的课程）
        sorted_courses = sorted(self.data["courses"], key=lambda x: x["size"], reverse=True)

        # [新增] 先分组连堂课程（为连堂检查做准备）
        for course in sorted_courses:
            link_id = course.get("link_id")
            if link_id:
                linked_courses[link_id].append(course)

        # 贪心主循环：为每门课程寻找合适的时间和教室
        for course in sorted_courses:
            assigned = False
            # 遍历所有时间和教室的组合
            for t in self.data["metadata"]["times"]:
                if assigned: break
                for r in self.data["rooms"]:
                    # --- 硬约束检查 ---
                    # 1. 教室类型不对
                    if course["type"] != r["type"]: continue
                    # 2. 教室容量不足
                    if course["size"] > r["cap"]: continue
                    
                    # [新增] 3. 教师禁排时间检查（需求8：某些时间教师不可用）
                    blocked_hours = course.get("blocked_hours", [])
                    if t in blocked_hours:
                        continue  # 该时间教师不可用，跳过
                    
                    # 4. 教室时间冲突检查 (考虑单双周复用)
                    # 修正2: 检查教室占用 (遍历列表)
                    is_room_conflict = False
                    if (t, r["id"]) in occupied:
                        for existing_week in occupied[(t, r["id"])]:
                            # 检查是否与已有课程的周次冲突
                            if weeks_conflict(existing_week, course["weeks"]):
                                is_room_conflict = True
                                break
                    if is_room_conflict: continue
                    
                    # 5. 教师时间冲突检查 (考虑单双周复用)
                    # 修正3: 检查老师占用 (遍历列表)
                    is_teacher_conflict = False
                    if (t, course["teacher"]) in teacher_busy:
                        for existing_week in teacher_busy[(t, course["teacher"])]:
                            # 检查是否与已有课程的周次冲突
                            if weeks_conflict(existing_week, course["weeks"]):
                                is_teacher_conflict = True
                                break
                    if is_teacher_conflict: continue
                    
                    # [新增] 6. 检查专业/班级冲突（需求6&7：同班级同时间只能一门课）
                    majors = course.get("class_groups", [])
                    if not majors:
                        majors = ["全校通选"]
                    is_major_conflict = False
                    for major in majors:
                        if (t, major) in major_busy:
                            for existing_week in major_busy[(t, major)]:
                                if weeks_conflict(existing_week, course["weeks"]):
                                    is_major_conflict = True
                                    break
                        if is_major_conflict: break
                    if is_major_conflict: continue
                    
                    # [新增] 7. 连堂约束检查（需求9：相同link_id的课程必须连续排课）
                    link_id = course.get("link_id")
                    skip_linked = False
                    if link_id:
                        # 检查连堂伙伴是否已分配，以及是否能在t+1连续
                        partner_courses = linked_courses[link_id]
                        other_courses = [c for c in partner_courses if c != course]
                        
                        # 查找伙伴课程是否已在schedule中
                        partner_assignment = None
                        for sched_item in schedule:
                            if sched_item["course"] in other_courses:
                                partner_assignment = sched_item
                                break
                        
                        if partner_assignment:
                            # 伙伴已分配，检查是否能连续排课
                            partner_time = partner_assignment["time"]
                            partner_room = partner_assignment["room"]
                            
                            # 计算时间差和是否同天
                            all_times = self.data["metadata"]["times"]
                            try:
                                t_idx = all_times.index(t)
                                partner_t_idx = all_times.index(partner_time)
                                
                                # 检查是否相邻且同天（每天5节课）
                                if abs(t_idx - partner_t_idx) == 1 and (t_idx // 5) == (partner_t_idx // 5) and r["id"] == partner_room:
                                    # 可以连堂
                                    pass
                                else:
                                    skip_linked = True
                            except ValueError:
                                skip_linked = True
                    
                    if skip_linked: continue
                    
                    # --- 分配成功 ---
                    schedule.append({
                        "course": course, "time": t, "room": r["id"]
                    })
                    
                    # 修正4: 更新占用表 (Append 模式)
                    if (t, r["id"]) not in occupied: occupied[(t, r["id"])] = []
                    occupied[(t, r["id"])].append(course["weeks"])
                    
                    if (t, course["teacher"]) not in teacher_busy: teacher_busy[(t, course["teacher"])] = []
                    teacher_busy[(t, course["teacher"])].append(course["weeks"])
                    
                    # [新增] 更新专业占用表
                    for major in majors:
                        if (t, major) not in major_busy:
                            major_busy[(t, major)] = []
                        major_busy[(t, major)].append(course["weeks"])
                    
                    assigned = True
                    break
        
        return schedule, "Greedy Completed (May have unscheduled courses)"

# ==========================================
# 2. 遗传算法求解器 (Genetic Algorithm)
# ==========================================
# 策略：通过种群进化（选择、交叉、变异）找到较优排课方案
# 优点：能找到相对较好的解，比贪心更优
# 缺点：不保证最优，需要调参
# ==========================================
class GASolver:
    def __init__(self, data, weights):
        """
        初始化遗传算法求解器
        
        参数：
            data: 包含元数据、教室和课程的字典
            weights: 约束权重字典，包含：
                - "hard": 硬约束违反的惩罚权重（应该很大）
                - "soft": 软约束违反的惩罚权重（较小）
        """
        self.data = data
        self.weights = weights
        self.times = data["metadata"]["times"]
        self.rooms = data["rooms"]
        self.courses = data["courses"]
        self.room_map = {r["id"]: r for r in self.rooms}  # 快速查询教室信息

    def create_individual(self):
        """
        创建一个个体（一个随机排课方案）
        
        每个个体是一个排课列表，每个元素包含课程、时间、教室的随机分配
        返回值：排课方案列表
        """
        return [{
            "course": c,
            "time": random.choice(self.times),  # 随机选择时间
            "room": random.choice(self.rooms)["id"]  # 随机选择教室
        } for c in self.courses]

    def calculate_fitness(self, individual):
        """
        计算个体的适应度（方案的优劣程度）
        
        参数：
            individual: 一个排课方案
        
        返回值：
            (fitness, cost) 元组
            - fitness: 适应度值，范围在 [0, 1]，值越大越好
            - cost: 总惩罚成本，值越小越好
        """
        hard_pen = 0  # 硬约束违反次数
        soft_pen = 0  # 软约束违反次数
        
        # 记录占用用于冲突检测
        r_occ = {}      # key: (time, room) -> list of weeks
        t_occ = {}      # key=(time, teacher) -> list of weeks
        major_occ = {}  # [新增] key=(time, major) -> list of weeks

        # 逐个检查每门课程的分配是否违反约束
        for gene in individual:
            c = gene["course"]
            t = gene["time"]
            r_id = gene["room"]
            
            # --- 硬约束检查 ---
            
            # [硬] 1. 教室类型必须匹配
            if c["type"] != self.room_map[r_id]["type"]: 
                hard_pen += 1
            
            # [软] 2. 教室容量（越接近越好，但容量足够就不算错）
            if c["size"] > self.room_map[r_id]["cap"]: 
                soft_pen += 1

            # [硬] 3. 教室时间冲突 (考虑单双周)
            key_r = (t, r_id)
            if key_r not in r_occ: 
                r_occ[key_r] = []
            for existing_week in r_occ[key_r]:
                if weeks_conflict(existing_week, c["weeks"]):
                    hard_pen += 1
            r_occ[key_r].append(c["weeks"])

            # [硬] 4. 老师时间冲突
            key_t = (t, c["teacher"])
            if key_t not in t_occ: 
                t_occ[key_t] = []
            for existing_week in t_occ[key_t]:
                if weeks_conflict(existing_week, c["weeks"]):
                    hard_pen += 1
            t_occ[key_t].append(c["weeks"])
            
            # [新增硬] 5. 专业/班级冲突（需求6&7）
            majors = c.get("class_groups", [])
            if not majors:
                majors = ["全校通选"]
            for major in majors:
                key_maj = (t, major)
                if key_maj not in major_occ: 
                    major_occ[key_maj] = []
                for existing_week in major_occ[key_maj]:
                    if weeks_conflict(existing_week, c["weeks"]):
                        hard_pen += 1
                major_occ[key_maj].append(c["weeks"])
            
            # [新增硬] 6. 教师禁排时间检查（需求8）
            blocked_hours = c.get("blocked_hours", [])
            if t in blocked_hours:
                hard_pen += 1
            
            # [新增硬] 7. 连堂约束检查（需求9）
            link_id = c.get("link_id")
            if link_id:
                # 找到相同link_id的其他课程
                for other_gene in individual:
                    other_c = other_gene["course"]
                    if other_c.get("link_id") == link_id and other_c != c:
                        # 检查是否能连堂（相邻时间且同教室）
                        all_times = self.times
                        try:
                            t_idx = all_times.index(t)
                            other_t_idx = all_times.index(other_gene["time"])
                            
                            # 检查条件：相邻、同天、同教室
                            is_adjacent = abs(t_idx - other_t_idx) == 1
                            is_same_day = (t_idx // 5) == (other_t_idx // 5)  # 每天5节课
                            is_same_room = r_id == other_gene["room"]
                            
                            if not (is_adjacent and is_same_day and is_same_room):
                                hard_pen += 1
                        except ValueError:
                            hard_pen += 1

        # 综合计算适应度：成本越低越好
        cost = hard_pen * self.weights["hard"] + soft_pen * self.weights["soft"]
        # 适应度函数：倒数函数，成本越小适应度越大
        return 1.0 / (1.0 + cost), cost

    def run(self, pop_size=50, gens=100, callback=None):
        """
        执行遗传算法的主循环
        
        参数：
            pop_size: 种群大小
            gens: 进化代数
            callback: 回调函数，用于每代执行（如更新进度条）
        
        返回值：
            (best_individual, best_cost_history) 元组
            - best_individual: 最优排课方案
            - best_cost_history: 每代最优成本的列表
        """
        # 第一步：初始化第一代种群（随机生成）
        pop = [self.create_individual() for _ in range(pop_size)]
        best_hist = []
        
        # 第二步：进化主循环
        for g in range(gens):
            # 计算整个种群的适应度并排序
            scored = [(self.calculate_fitness(ind), ind) for ind in pop]
            scored.sort(key=lambda x: x[0][0], reverse=True)  # 按适应度降序排列
            
            best_fitness, best_cost = scored[0][0][0], scored[0][0][1]
            best_ind = scored[0][1]
            best_hist.append(best_cost)
            
            # 回调函数（用于进度更新）
            if callback: 
                callback(g, best_cost)
            
            # 如果找到完美解，提前终止
            if best_cost == 0: 
                break

            # 第三步：生成下一代 (简化版：精英保留 + 随机变异)
            # 精英策略：保留前20%的优秀个体
            new_pop = [x[1] for x in scored[:int(pop_size*0.2)]]
            
            # 随机生成新个体以保持种群大小
            while len(new_pop) < pop_size:
                # 从前50%的优秀个体中选择父亲进行变异
                parent = random.choice(scored[:int(pop_size*0.5)])[1]
                child = copy.deepcopy(parent)
                
                # 变异：30% 概率随机改变一个基因（课程的时间和教室）
                if random.random() < 0.3:
                    gene = random.choice(child)
                    gene["time"] = random.choice(self.times)
                    gene["room"] = random.choice(self.rooms)["id"]
                
                new_pop.append(child)
            
            pop = new_pop
            
        return best_ind, best_hist

# ==========================================
# 3. Google OR-Tools CP-SAT 约束规划求解器
# ==========================================
# 策略：将排课问题建模为约束满足问题(CSP)，使用整数线性规划求解
# 优点：能找到最优解或接近最优解，可处理复杂约束
# 缺点：数据量大时求解较慢，但结果最优
# ==========================================
class CPSATSolver:
    def __init__(self, data):
        """
        初始化 CP-SAT 求解器
        
        参数：
            data: 包含元数据、教室和课程的字典
        """
        self.data = data
        self.times = data["metadata"]["times"]
        self.rooms = data["rooms"]
        self.courses = data["courses"]
        
        # 预计算：每门课允许使用的教室列表 (改进点2：预过滤)
        # 这就是稀疏建模的核心：只记录可行的组合，减少变量数量
        # key: course_index, val: 可用教室下标列表
        self.possible_rooms = {} 
        for c_idx, c in enumerate(self.courses):
            valid = []
            for r_idx, r in enumerate(self.rooms):
                # 检查类型匹配
                if c["type"] != r["type"]: 
                    continue
                # 检查容量匹配
                if c["size"] > r["cap"]: 
                    continue
                valid.append(r_idx)
            self.possible_rooms[c_idx] = valid

    def solve(self, time_limit=60):
        """
        执行 CP-SAT 排课求解
        
        参数：
            time_limit: 求解的时间限制（秒）
        
        返回值：
            (schedule, msg) 元组
            - schedule: 排课结果列表
            - msg: 求解状态和说明信息
        """
        model = cp_model.CpModel()
        
        # --- 第一步：定义决策变量 (稀疏矩阵) ---
        # x[(c, t, r)] = 1 表示课程 c 在时间 t 在教室 r 上课
        x = {}
        
        # y[c] = 1 表示课程 c 被成功安排了 (改进点1：允许不排课)
        is_scheduled = {} 

        for c_idx in range(len(self.courses)):
            # 创建是否排课的标记变量
            is_scheduled[c_idx] = model.NewBoolVar(f'scheduled_{c_idx}')
            
            # [新增] 获取该课程教师的禁排时间
            teacher_blocks = self.courses[c_idx].get("blocked_hours", [])
            
            # 只为合法的教室创建变量 (Sparse Modeling)
            for r_idx in self.possible_rooms[c_idx]:
                for t_idx in range(len(self.times)):
                    t_str = self.times[t_idx]
                    
                    # [新增] 检查是否被禁排（需求8：教师时间约束）
                    if t_str in teacher_blocks:
                        continue  # 跳过创建变量，该时间点无法排课
                    
                    # 创建该课程、时间、教室的排课变量
                    x[(c_idx, t_idx, r_idx)] = model.NewBoolVar(f'x_{c_idx}_{t_idx}_{r_idx}')
            
            # [新增] 防守性约束：如果某课程完全没有变量（所有时间都被禁排），无法排课
            has_any_var = False
            for r_idx in self.possible_rooms[c_idx]:
                for t_idx in range(len(self.times)):
                    if (c_idx, t_idx, r_idx) in x:
                        has_any_var = True
                        break
                if has_any_var:
                    break
            if not has_any_var and teacher_blocks:
                # 所有时间都被禁排，该课程无法排课
                model.Add(is_scheduled[c_idx] == 0)

        # --- 第二步：添加硬约束 ---
        
        # (A) 课程安排约束：
        # 如果 y[c]=1，则必须且只能排一次；如果 y[c]=0，则一次都不能排
        for c_idx in range(len(self.courses)):
            # 收集这门课所有可能的 (t, r) 组合变量
            possible_assignments = []
            for r_idx in self.possible_rooms[c_idx]:
                for t_idx in range(len(self.times)):
                    var = x.get((c_idx, t_idx, r_idx))
                    if var is not None:
                        possible_assignments.append(var)
            
            if possible_assignments:
                # 该课程被排课当且仅当恰好选一个 (t,r) 组合
                model.Add(sum(possible_assignments) == is_scheduled[c_idx])
            else:
                # 如果这门课连一个能用的教室都找不到（比如容量都太小），那它肯定排不了
                model.Add(is_scheduled[c_idx] == 0)

        # (B) 教室时间冲突约束：处理单双周复用逻辑
        # 逻辑：
        # 1. sum(all_week_vars) + sum(odd_week_vars) <= 1
        # 2. sum(all_week_vars) + sum(even_week_vars) <= 1
        for t_idx in range(len(self.times)):
            for r_idx in range(len(self.rooms)):
                # 分类收集变量：按周次模式分类
                vars_all = []
                vars_odd = []
                vars_even = []
                
                for c_idx in range(len(self.courses)):
                    if r_idx in self.possible_rooms[c_idx]:
                        # 获取该课程的变量，使用 .get() 以防键不存在
                        var = x.get((c_idx, t_idx, r_idx))
                        if var is None:
                            continue
                        week_type = self.courses[c_idx]["weeks"]
                        
                        if week_type == "all":
                            vars_all.append(var)
                        elif week_type == "odd":
                            vars_odd.append(var)
                        elif week_type == "even":
                            vars_even.append(var)
                
                # 如果有变量，添加约束
                if vars_all or vars_odd or vars_even:
                    # 绝对硬约束：同一时间同一地点，总课程数不能超过2 
                    # (防止 odd+odd 这种情况)
                    model.Add(sum(vars_all + vars_odd + vars_even) <= 2)
                    
                    # 核心互斥逻辑
                    model.Add(sum(vars_all) + sum(vars_odd) <= 1)
                    model.Add(sum(vars_all) + sum(vars_even) <= 1)

        # (B+) [新增] 班级/专业不冲突约束（需求6&7）
        # 同一专业的学生在同一时间只能上一门课
        major_to_courses = collections.defaultdict(list)
        for c_idx, c in enumerate(self.courses):
            # class_groups 是适用班级的列表
            majors = c.get("class_groups", [])
            if not majors:  # 如果没有指定班级，使用默认值
                majors = ["全校通选"]
            for major in majors:
                major_to_courses[major].append(c_idx)
        
        for major, c_indices in major_to_courses.items():
            for t_idx in range(len(self.times)):
                # 收集该专业、该时间片下所有可能的课程变量
                major_vars = []
                for c_idx in c_indices:
                    for r_idx in self.possible_rooms[c_idx]:
                        if (c_idx, t_idx, r_idx) in x:
                            major_vars.append(x[(c_idx, t_idx, r_idx)])
                
                # 该专业在该时刻至多有一门课（考虑所有周次）
                if major_vars:
                    model.Add(sum(major_vars) <= 1)

        # (C) 老师时间冲突约束：处理单双周复用逻辑 (同上)
        # 允许同一个老师在同一时间片，分别在单周和双周上课
        teacher_map = collections.defaultdict(list)
        for c_idx, c in enumerate(self.courses):
            teacher_map[c["teacher"]].append(c_idx)
            
        for teacher, c_indices in teacher_map.items():
            for t_idx in range(len(self.times)):
                t_vars_all = []
                t_vars_odd = []
                t_vars_even = []
                
                for c_idx in c_indices:
                    for r_idx in self.possible_rooms[c_idx]:
                        var = x.get((c_idx, t_idx, r_idx))
                        if var is None:
                            continue
                        week_type = self.courses[c_idx]["weeks"]
                        
                        if week_type == "all": 
                            t_vars_all.append(var)
                        elif week_type == "odd": 
                            t_vars_odd.append(var)
                        elif week_type == "even": 
                            t_vars_even.append(var)
                
                if t_vars_all or t_vars_odd or t_vars_even:
                    model.Add(sum(t_vars_all) + sum(t_vars_odd) <= 1)
                    model.Add(sum(t_vars_all) + sum(t_vars_even) <= 1)
        
        # (C+) [新增] 连堂约束（需求9）
        # 相同 link_id 的课程必须连续排课（同教室，且时间为 t 和 t+1）
        linked_groups = collections.defaultdict(list)
        for c_idx, c in enumerate(self.courses):
            link_id = c.get("link_id")
            if link_id:
                linked_groups[link_id].append(c_idx)
        
        for link_id, c_indices in linked_groups.items():
            if len(c_indices) == 2:
                c1_idx, c2_idx = c_indices[0], c_indices[1]
                
                # 要么都排，要么都不排
                model.Add(is_scheduled[c1_idx] == is_scheduled[c2_idx])
                
                for t_idx in range(len(self.times) - 1):
                    day_idx = t_idx // 5
                    day_idx_next = (t_idx + 1) // 5
                    if day_idx != day_idx_next:
                        continue 
                    
                    for r_idx in self.possible_rooms[c1_idx]:
                        if r_idx not in self.possible_rooms[c2_idx]:
                            continue
                        
                        var_c1_t = x.get((c1_idx, t_idx, r_idx))
                        var_c2_t1 = x.get((c2_idx, t_idx + 1, r_idx))
                        
                        # 【修复点】：必须使用 is not None 来判断变量是否存在
                        if var_c1_t is not None and var_c2_t1 is not None:
                            # 逻辑：如果 c1 在 (t, r) 则 c2 必须在 (t+1, r)
                            model.Add(var_c2_t1 == 1).OnlyEnforceIf(var_c1_t)
                            model.Add(var_c1_t == 1).OnlyEnforceIf(var_c2_t1)

            elif len(c_indices) > 2:
                # 多于两门课的连堂
                for i in range(len(c_indices) - 1):
                    model.Add(is_scheduled[c_indices[i]] == is_scheduled[c_indices[i + 1]])
                
                for i in range(len(c_indices) - 1):
                    c_curr_idx = c_indices[i]
                    c_next_idx = c_indices[i + 1]
                    
                    for t_idx in range(len(self.times) - 1):
                        day_idx = t_idx // 5
                        day_idx_next = (t_idx + 1) // 5
                        if day_idx != day_idx_next:
                            continue
                        
                        for r_idx in self.possible_rooms[c_curr_idx]:
                            if r_idx not in self.possible_rooms[c_next_idx]:
                                continue
                            
                            var_curr = x.get((c_curr_idx, t_idx, r_idx))
                            var_next = x.get((c_next_idx, t_idx + 1, r_idx))
                            
                            # 【修复点】：同样使用 is not None
                            if var_curr is not None and var_next is not None:
                                model.Add(var_next == 1).OnlyEnforceIf(var_curr)
                                model.Add(var_curr == 1).OnlyEnforceIf(var_next)

        # (D) 单双周互斥 (高级约束)
        # 注：大部分情况下上面的约束已经处理，这里不再赘述

        # --- 第三步：定义目标函数 ---
        # 目标：最大化排课数量 - 软约束惩罚
        BIG_W = 10000  # 大权重，确保排课数优先

        # --- 专业-时间占用指示符 y_major_t ---
        # y_major_time[(major, t_idx)] == 1 表示该专业在该时间段有课程
        y_major_time = {}
        majors_list = list(major_to_courses.keys())
        for major in majors_list:
            for t_idx in range(len(self.times)):
                # 收集该专业在该时间的所有变量
                maj_vars = []
                for c_idx in major_to_courses[major]:
                    for r_idx in self.possible_rooms[c_idx]:
                        v = x.get((c_idx, t_idx, r_idx))
                        if v is not None:
                            maj_vars.append(v)
                if maj_vars:
                    y = model.NewBoolVar(f'y_{major}_{t_idx}')
                    y_major_time[(major, t_idx)] = y
                    # 关联 y 与 maj_vars: sum(maj_vars) >= y 且 sum(maj_vars) <= len* y
                    model.Add(sum(maj_vars) >= y)
                    model.Add(sum(maj_vars) <= len(maj_vars) * y)

        # --- 紧凑性惩罚 ---
        # 对于每个专业和每一天，对非相邻的占用时段进行惩罚。
        compact_pen_vars = []
        compact_pen_weight = 5
        slots_per_day = 5  # 假设每天5个时间段
        for major in majors_list:
            for day in range((len(self.times) // slots_per_day)):
                # 考虑同一天内间隔 > 1 的时间对
                base = day * slots_per_day
                for i in range(slots_per_day):
                    t_i = base + i
                    yi = y_major_time.get((major, t_i))
                    if yi is None: 
                        continue
                    for j in range(i+1, slots_per_day):
                        t_j = base + j
                        yj = y_major_time.get((major, t_j))
                        if yj is None: 
                            continue
                        gap = j - i
                        if gap > 1:
                            b = model.NewBoolVar(f'gap_{major}_{t_i}_{t_j}')
                            compact_pen_vars.append((b, gap))
                            # b == yi AND yj 线性化
                            model.Add(b <= yi)
                            model.Add(b <= yj)
                            model.Add(b >= yi + yj - 1)

        # --- 分散惩罚（最小化每日课时方差的近似）---
        # 对于每个专业，计算 daily_hours[d] = sum y_major_time[major, t in day]
        spread_pen_vars = []
        spread_pen_weight = 2
        for major in majors_list:
            # 构建每日课时变量
            daily_hours = []
            for day in range((len(self.times) // slots_per_day)):
                dh = model.NewIntVar(0, slots_per_day, f'dh_{major}_{day}')
                daily_hours.append(dh)
                # 每天各时段的 y 值之和等于该天的课时数
                day_vars = []
                base = day * slots_per_day
                for i in range(slots_per_day):
                    y = y_major_time.get((major, base + i))
                    if y is not None:
                        day_vars.append(y)
                if day_vars:
                    model.Add(sum(day_vars) == dh)
                else:
                    model.Add(dh == 0)

            # 对各天之间的绝对差进行惩罚
            for d1 in range(len(daily_hours)):
                for d2 in range(d1+1, len(daily_hours)):
                    diff = model.NewIntVar(0, slots_per_day, f'diff_{major}_{d1}_{d2}')
                    spread_pen_vars.append(diff)
                    # diff >= dh1 - dh2 且 diff >= dh2 - dh1
                    model.Add(daily_hours[d1] - daily_hours[d2] <= diff)
                    model.Add(daily_hours[d2] - daily_hours[d1] <= diff)

        # --- 组合惩罚线性表达式 ---
        compact_term = sum([b * gap for (b, gap) in compact_pen_vars]) if compact_pen_vars else 0
        spread_term = sum(spread_pen_vars) if spread_pen_vars else 0

        # 目标：最大化 BIG_W * 排课数量 - (软约束惩罚)
        scheduled_sum = sum(is_scheduled.values())
        penalty_expr = compact_pen_weight * compact_term + spread_pen_weight * spread_term
        model.Maximize(BIG_W * scheduled_sum - penalty_expr)

        # --- 第四步：调用求解器求解 ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model) #
        
        # --- 第五步：处理求解结果 ---
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            res = []
            unscheduled_count = 0
            
            # 逐个检查每门课程是否被成功排课
            for c_idx in range(len(self.courses)):
                if solver.Value(is_scheduled[c_idx]) == 1:
                    # 只有被安排了才能找时间地点
                    assigned = False
                    for r_idx in self.possible_rooms[c_idx]:
                        for t_idx in range(len(self.times)):
                            var = x.get((c_idx, t_idx, r_idx))
                            if var is not None and solver.Value(var) == 1:
                                res.append({
                                    "course": self.courses[c_idx],
                                    "time": self.times[t_idx],
                                    "room": self.rooms[r_idx]["id"]
                                })
                                assigned = True
                                break
                        if assigned: 
                            break
                else:
                    unscheduled_count += 1
            
            status_text = f"求解完成 ({solver.StatusName(status)})。未排入: {unscheduled_count}"
            return res, status_text
        else:
            return [], "求解失败：无可行解"
        model = cp_model.CpModel()
        