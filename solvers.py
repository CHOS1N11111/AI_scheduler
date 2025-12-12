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
    return w1 == w2  # 单周 vs 单周 = True，单周 vs 双周 = False

# ==========================================
# 1. 贪心算法（贪心基线）
# ==========================================
class GreedySolver:
    def __init__(self, data):
        self.data = data
        
    def solve(self):
        schedule = []
        # 修正1: 占用表的值改为列表，存储该时空下所有已占用的周次
        # key=(time, room)，值=[周次模式1, 周次模式2, ...]
        occupied = {} 
        # key=(time, teacher)，值=[周次模式1, ...]
        teacher_busy = {}
        # [新增] 专业/班级占用表
        # key=(time, major)，值=[周次模式1, ...]
        major_busy = {}
        # [新增] 追踪已分配的课程（用于连堂检查）
        # key=link_id，值=[课程对象1, 课程对象2, ...]
        linked_courses = collections.defaultdict(list)

        # 策略：按人数从大到小排序，先排难排的
        sorted_courses = sorted(self.data["courses"], key=lambda x: x["size"], reverse=True)

        # [新增] 先分组连堂课程
        for course in sorted_courses:
            link_id = course.get("link_id")
            if link_id:
                linked_courses[link_id].append(course)

        for course in sorted_courses:
            assigned = False
            # 遍历所有时间和教室
            for t in self.data["metadata"]["times"]:
                if assigned: break
                for r in self.data["rooms"]:
                    # --- 硬约束检查 ---
                    # 1. 教室类型不对
                    if course["type"] != r["type"]: continue
                    # 2. 教室容量不足
                    if course["size"] > r["cap"]: continue
                    
                    # [新增] 3. 教师禁排时间检查（需求8）
                    blocked_hours = course.get("blocked_hours", [])
                    if t in blocked_hours:
                        continue
                    
                    # 4. 资源冲突（时间+空间+周次）
                    # 修正2: 检查教室占用（遍历列表）
                    is_room_conflict = False
                    if (t, r["id"]) in occupied:
                        for existing_week in occupied[(t, r["id"])]:
                            if weeks_conflict(existing_week, course["weeks"]):
                                is_room_conflict = True
                                break
                    if is_room_conflict: continue
                    
                    # 修正3: 检查老师占用（遍历列表）
                    is_teacher_conflict = False
                    if (t, course["teacher"]) in teacher_busy:
                        for existing_week in teacher_busy[(t, course["teacher"])]:
                            if weeks_conflict(existing_week, course["weeks"]):
                                is_teacher_conflict = True
                                break
                    if is_teacher_conflict: continue
                    
                    # [新增] 5. 检查专业/班级冲突（需求6&7）
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
                    
                    # [新增] 6. 连堂约束检查（需求9）
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
                    
                    # 修正4: 更新占用表（追加模式）
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
        
        return schedule, "贪心算法完成（可能有未排课程）"

# ==========================================
# 2. 遗传算法（进化算法）
# ==========================================
class GASolver:
    def __init__(self, data, weights):
        self.data = data
        self.weights = weights
        self.times = data["metadata"]["times"]
        self.rooms = data["rooms"]
        self.courses = data["courses"]
        self.room_map = {r["id"]: r for r in self.rooms}

    def create_individual(self):
        return [{
            "course": c,
            "time": random.choice(self.times),
            "room": random.choice(self.rooms)["id"]
        } for c in self.courses]

    def calculate_fitness(self, individual):
        hard_pen = 0
        soft_pen = 0
        
        # 记录占用用于冲突检测
        # key=(time, room) -> 周次列表
        r_occ = {}
        # key=(time, teacher)：对应的周次列表
        t_occ = {}
        # [新增] key=(time, major)：对应的周次列表
        major_occ = {}

        for gene in individual:
            c = gene["course"]
            t = gene["time"]
            r_id = gene["room"]
            
            # [硬] 教室类型
            if c["type"] != self.room_map[r_id]["type"]: hard_pen += 1
            # [软] 教室容量
            if c["size"] > self.room_map[r_id]["cap"]: soft_pen += 1

            # [硬] 教室时间冲突（考虑单双周）
            key_r = (t, r_id)
            if key_r not in r_occ: r_occ[key_r] = []
            for existing_week in r_occ[key_r]:
                if weeks_conflict(existing_week, c["weeks"]):
                    hard_pen += 1
            r_occ[key_r].append(c["weeks"])

            # [硬] 老师时间冲突
            key_t = (t, c["teacher"])
            if key_t not in t_occ: t_occ[key_t] = []
            for existing_week in t_occ[key_t]:
                if weeks_conflict(existing_week, c["weeks"]):
                    hard_pen += 1
            t_occ[key_t].append(c["weeks"])
            
            # [新增硬] 专业/班级冲突（需求6&7）
            majors = c.get("class_groups", [])
            if not majors:
                majors = ["全校通选"]
            for major in majors:
                key_maj = (t, major)
                if key_maj not in major_occ: major_occ[key_maj] = []
                for existing_week in major_occ[key_maj]:
                    if weeks_conflict(existing_week, c["weeks"]):
                        hard_pen += 1
                major_occ[key_maj].append(c["weeks"])
            
            # [新增硬] 教师禁排时间检查（需求8）
            blocked_hours = c.get("blocked_hours", [])
            if t in blocked_hours:
                hard_pen += 1
            
            # [新增硬] 连堂约束检查（需求9）
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

        cost = hard_pen * self.weights["hard"] + soft_pen * self.weights["soft"]
        return 1.0 / (1.0 + cost), cost

    def run(self, pop_size=50, gens=100, callback=None):
        pop = [self.create_individual() for _ in range(pop_size)]
        best_hist = []
        
        for g in range(gens):
            scored = [(self.calculate_fitness(ind), ind) for ind in pop]
            scored.sort(key=lambda x: x[0][0], reverse=True)
            
            best_fitness, best_cost = scored[0][0][0], scored[0][0][1]
            best_ind = scored[0][1]
            best_hist.append(best_cost)
            
            if callback: callback(g, best_cost)
            if best_cost == 0: break # 完美解

            # 进化下一代（简化版：精英 + 随机变异）
            new_pop = [x[1] for x in scored[:int(pop_size*0.2)]] # 保留前20%
            while len(new_pop) < pop_size:
                parent = random.choice(scored[:int(pop_size*0.5)])[1]
                child = copy.deepcopy(parent)
                # 变异
                if random.random() < 0.3:
                    gene = random.choice(child)
                    gene["time"] = random.choice(self.times)
                    gene["room"] = random.choice(self.rooms)["id"]
                new_pop.append(child)
            pop = new_pop
            
        return best_ind, best_hist

# ==========================================
# 3. Google OR-Tools（CP-SAT）- 数学建模
# ==========================================
class CPSATSolver:
    def __init__(self, data):
        self.data = data
        self.times = data["metadata"]["times"]
        self.rooms = data["rooms"]
        self.courses = data["courses"]
        
        # 预计算：每门课允许使用的教室列表（改进点2：预过滤）
        # 这就是稀疏建模的核心：只记录可行的组合
        self.possible_rooms = {} 
        for c_idx, c in enumerate(self.courses):
            valid = []
            for r_idx, r in enumerate(self.rooms):
                # 检查类型匹配
                if c["type"] != r["type"]: continue
                # 检查容量匹配
                if c["size"] > r["cap"]: continue
                valid.append(r_idx)
            self.possible_rooms[c_idx] = valid

    def solve(self, time_limit=30):
        model = cp_model.CpModel()
        
        # --- 1. 定义变量（稀疏矩阵）---
        # x[(c, t, r)] = 1 表示课程 c 在时间 t 在教室 r 上课
        x = {}
        
        # y[c] = 1 表示课程 c 被成功安排了（改进点1：允许不排课）
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

        # --- 2. 硬约束（Hard Constraints）---
        
        # （A）课程安排约束：
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
                model.Add(sum(possible_assignments) == is_scheduled[c_idx])
            else:
                # 如果这门课连一个能用的教室都找不到（比如容量都太小），那它肯定排不了
                model.Add(is_scheduled[c_idx] == 0)

        # （B）教室冲突：处理单双周复用逻辑
        # 逻辑：
        # 1. sum(all_week_vars) + sum(odd_week_vars) <= 1
        # 2. sum(all_week_vars) + sum(even_week_vars) <= 1
        for t_idx in range(len(self.times)):
            for r_idx in range(len(self.rooms)):
                # 分类收集变量
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
                    # 绝对硬约束：同一时间同一地点，总课程数不能超过2（防止 odd+odd 这种情况）
                    # 其实下面的逻辑已经隐含了，但为了求解器加速可以加上
                    model.Add(sum(vars_all + vars_odd + vars_even) <= 2)
                    
                    # 核心互斥逻辑
                    model.Add(sum(vars_all) + sum(vars_odd) <= 1)
                    model.Add(sum(vars_all) + sum(vars_even) <= 1)

        # （B+）[新增] 班级/专业不冲突约束（需求6&7）
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

        # （C）老师冲突：处理单双周复用逻辑（同上）
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
                        
                        if week_type == "all": t_vars_all.append(var)
                        elif week_type == "odd": t_vars_odd.append(var)
                        elif week_type == "even": t_vars_even.append(var)
                
                if t_vars_all or t_vars_odd or t_vars_even:
                    model.Add(sum(t_vars_all) + sum(t_vars_odd) <= 1)

                    model.Add(sum(t_vars_all) + sum(t_vars_even) <= 1)
        
        # （C+）[新增] 连堂约束（需求9）
        # 相同 link_id 的课程必须连续排课（同教室，且时间为 t 和 t+1）
        linked_groups = collections.defaultdict(list)
        for c_idx, c in enumerate(self.courses):
            link_id = c.get("link_id")
            if link_id:  # 只有有 link_id 的课才参与连堂
                linked_groups[link_id].append(c_idx)
        
        for link_id, c_indices in linked_groups.items():
            # 连堂通常是两门课一对
            if len(c_indices) == 2:
                c1_idx, c2_idx = c_indices[0], c_indices[1]
                
                # 先添加"要么都排，要么都不排"的约束
                model.Add(is_scheduled[c1_idx] == is_scheduled[c2_idx])
                
                # 遍历所有时间和教室组合
                for t_idx in range(len(self.times) - 1):  # -1 是因为需要 t+1
                    # 检查是否跨天（例如同一天内的相邻时间，简化假设每天5节课）
                    # times 格式: ["Mon_08:00", "Mon_10:00", "Mon_14:00", "Mon_16:00", "Mon_19:00", "Tue_08:00", ...]
                    # 避免周一最后一节 -> 周二第一节
                    day_idx = t_idx // 5
                    day_idx_next = (t_idx + 1) // 5
                    if day_idx != day_idx_next:
                        continue  # 不同天，不能连堂
                    
                    for r_idx in self.possible_rooms[c1_idx]:
                        if r_idx not in self.possible_rooms[c2_idx]:
                            continue  # 两门课都要能用这个教室
                        
                        var_c1_t = x.get((c1_idx, t_idx, r_idx))
                        var_c2_t1 = x.get((c2_idx, t_idx + 1, r_idx))
                        
                        if var_c1_t and var_c2_t1:
                            # 逻辑：如果 c1 在 (t, r) 则 c2 必须在 (t+1, r)
                            model.Add(var_c2_t1 == 1).OnlyEnforceIf(var_c1_t)
                            # 反向：如果 c2 在 (t+1, r) 则 c1 必须在 (t, r)
                            model.Add(var_c1_t == 1).OnlyEnforceIf(var_c2_t1)
            elif len(c_indices) > 2:
                # 多于两门课的连堂，按顺序连接
                # 首先所有课程要么都排要么都不排
                for i in range(len(c_indices) - 1):
                    model.Add(is_scheduled[c_indices[i]] == is_scheduled[c_indices[i + 1]])
                
                # 然后施加连续性约束
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
                            
                            if var_curr and var_next:
                                model.Add(var_next == 1).OnlyEnforceIf(var_curr)
                                model.Add(var_curr == 1).OnlyEnforceIf(var_next)

        # （D）单双周互斥（高级约束）
        # 逻辑：如果两门课周次冲突，且都在同一时间，则不能同时发生
        # 这是一个稍微复杂的约束，为了性能，我们简化处理：
        # 如果两门课周次不兼容，则它们在（同一时间）的变量和不能大于1（不管在哪个教室）
        # 但上面的 (C) 和 (B) 已经处理了大部分冲突。
        # 这里重点是：同一个老师，能不能同时带两个班（一个单周，一个双周）？通常是可以的。
        # 同一个教室，能不能同时排两个班（一个单周，一个双周）？是可以的。

        # --- 3. 目标函数（改进点1：最大化排课数量）---
        # --- 3. 目标函数（改进点1：最大化排课数量 + 软约束惩罚）---
        # 软约束包括：午休保护、紧凑排课（减少同日空堂）、分散压力（减少每日课时方差）
        # 权重设定：确保排课数量优先，然后尽量减少惩罚
        BIG_W = 10000

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
        slots_per_day = 5  # 假设每天5个时间段（根据元数据）
        for major in majors_list:
            for day in range((len(self.times) // slots_per_day)):
                # 考虑同一天内间隔 > 1 的时间对
                base = day * slots_per_day
                for i in range(slots_per_day):
                    t_i = base + i
                    yi = y_major_time.get((major, t_i))
                    if yi is None: continue
                    for j in range(i+1, slots_per_day):
                        t_j = base + j
                        yj = y_major_time.get((major, t_j))
                        if yj is None: continue
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
        # lunch_pen = lunch_pen_weight * sum(lunch_pen_vars)
        # compact_pen = compact_pen_weight * sum(b * gap)
        # spread_pen = spread_pen_weight * sum(diff)
        compact_term = sum([b * gap for (b, gap) in compact_pen_vars]) if compact_pen_vars else 0
        spread_term = sum(spread_pen_vars) if spread_pen_vars else 0

        # 目标：最大化 BIG_W * 排课数量 - (lunch_term*权重 + compact_term*权重 + spread_term*权重)
        scheduled_sum = sum(is_scheduled.values())
        penalty_expr =  compact_pen_weight * compact_term + spread_pen_weight * spread_term
        # 注意：CP-SAT 要求目标为整数/布尔变量的线性表达式
        model.Maximize(BIG_W * scheduled_sum - penalty_expr)

        # --- 4. 求解（Solve）---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        status = solver.Solve(model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            res = []
            unscheduled_count = 0
            
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
                        if assigned: break
                else:
                    unscheduled_count += 1
            
            status_text = f"Solved ({solver.StatusName(status)}). Unscheduled: {unscheduled_count}"
            return res, status_text
        else:
            return [], "未找到解（即使考虑软约束）"