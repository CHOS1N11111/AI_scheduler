#  streamlit run app.py
import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 引入自定义模块
from data_adapter import load_real_data, generate_empty_template
from data_gen import generate_mock_data
from solvers import GreedySolver, GASolver, CPSATSolver

# =============================================
# 1. 页面全局配置
# =============================================
st.set_page_config(
    page_title="AI 智能排课控制台", 
    layout="wide", 
    page_icon="🎓",
    initial_sidebar_state="expanded"
)

# 初始化 Session State (防止刷新丢失数据)
if 'courses_data' not in st.session_state:
    st.session_state.courses_data = []

# =============================================
# 2. 侧边栏：资源与引擎配置 (Sidebar)
# =============================================
with st.sidebar:
    st.title("⚙️ 资源与算法配置")
    
    st.markdown("### 🏫 教室资源池")
    st.info("在此模拟教务处的硬件资源限制")
    
    # 动态配置教室
    c1, c2 = st.columns(2)
    with c1:
        num_multi = st.number_input("多媒体教室数", value=6, min_value=1, help="R系列大教室")
        cap_multi = st.number_input("多媒体容量", value=100)
    with c2:
        num_lab = st.number_input("机房数量", value=3, min_value=0, help="计算机实验室")
        cap_lab = st.number_input("机房容量", value=60)
        
    # 构建教室数据结构
    rooms = []
    for i in range(num_multi):
        rooms.append({"id": f"R{101+i}", "cap": cap_multi, "type": "multimedia"})
    for i in range(num_lab):
        rooms.append({"id": f"Lab_A{i+1}", "cap": cap_lab, "type": "lab"})
        
    st.caption(f"✅ 当前可用教室总数: {len(rooms)} 间")
    
    st.divider()
    
    st.markdown("### 🧠 求解引擎选择")
    solver_mode = st.selectbox("核心算法", 
        ["OR-Tools CP-SAT (精确建模)", "Genetic Algorithm (进化算法)", "Greedy (贪心基线)"],
        index=0
    )
    
    # 根据选择显示不同的算法参数
    if "Genetic" in solver_mode:
        st.markdown("#### 🧬 进化参数")
        ga_pop = st.slider("种群大小 (Population)", 20, 200, 50)
        ga_gen = st.slider("进化代数 (Generations)", 50, 500, 100)
        w_hard = st.number_input("硬约束权重", value=1000)
    elif "OR-Tools" in solver_mode:
        st.markdown("#### 🤖 求解参数")
        cp_timeout = st.slider("最大求解时间 (秒)", 10, 60, 30)

# =============================================
# 3. 主界面：数据管理 (Data Management)
# =============================================
st.title("🎓 电信学院 - 智能排课仿真控制台")
st.markdown("本系统支持 **模拟压力测试** 与 **真实数据导入**，并提供多维度算法对比分析。")

with st.container():
    st.subheader("📂 第一步：加载课程任务")
    
    # Tab 切换数据来源
    tab_sim, tab_real = st.tabs(["🎲 随机模拟生成 (演示用)", "📥 导入真实课表"])
    
    # >>> 模式 A: 随机模拟 <<<
    with tab_sim:
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            sim_count = st.slider("生成课程数量", 10, 100, 30, help="模拟多少个班级的课")
        with col_sim2:
            sim_lab_ratio = st.slider("实验课比例 (机房需求)", 0.0, 1.0, 0.3, help="拉高此值测试机房压力")
        with col_sim3:
            sim_week_ratio = st.slider("单双周课程比例", 0.0, 1.0, 0.2, help="拉高此值测试时间复用能力")
            
        if st.button("🎲 生成模拟数据", type="primary"):
            with st.spinner("正在生成虚拟教务数据..."):
                mock_pkg = generate_mock_data(
                    num_courses=sim_count, 
                    prob_lab=sim_lab_ratio, 
                    prob_odd_even=sim_week_ratio
                )
                st.session_state.courses_data = mock_pkg["courses"]
                st.success(f"✅ 已生成 {len(st.session_state.courses_data)} 条模拟数据！")

    # >>> 模式 B: 真实导入 <<<
    with tab_real:
        col_up, col_dl = st.columns([3, 1])
        with col_up:
            uploaded_file = st.file_uploader("上传 Excel/CSV 文件", type=['csv', 'xlsx'])
        with col_dl:
            st.download_button("下载数据模板", generate_empty_template(), "template.csv", "text/csv")
            
        if uploaded_file:
            real_courses, err = load_real_data(uploaded_file)
            if err:
                st.error(err)
            else:
                st.session_state.courses_data = real_courses
                st.success(f"✅ 导入成功！共 {len(real_courses)} 条。")

    # --- 数据预览与手动微调 (Data Editor) ---
    if st.session_state.courses_data:
        with st.expander(f"🔍 查看/编辑待排课程 ({len(st.session_state.courses_data)}门) - 可直接修改表格", expanded=False):
            # 转换列表为字符串以便编辑
            df_preview = pd.DataFrame(st.session_state.courses_data)
            
            # 将 class_groups (列表) 转换为字符串，将 blocked_hours (列表) 转换为字符串
            if "class_groups" in df_preview.columns:
                df_preview["class_groups"] = df_preview["class_groups"].apply(
                    lambda x: ",".join(x) if isinstance(x, list) else str(x)
                )
            if "blocked_hours" in df_preview.columns:
                df_preview["blocked_hours"] = df_preview["blocked_hours"].apply(
                    lambda x: ";".join(x) if isinstance(x, list) else str(x)
                )
            
            edited_df = st.data_editor(
                df_preview,
                column_config={
                    "type": st.column_config.SelectboxColumn("教室需求", options=["multimedia", "lab"], required=True),
                    "weeks": st.column_config.SelectboxColumn("周次模式", options=["all", "odd", "even"], required=True),
                    "size": st.column_config.NumberColumn("人数", min_value=1, max_value=500),
                    "class_groups": st.column_config.TextColumn("适用专业 (逗号分隔)", help="如: 计科2301,电气2302"),
                    "link_id": st.column_config.TextColumn("连堂ID", help="ID相同的课必须连着上"),
                    "blocked_hours": st.column_config.TextColumn("教师禁排时间", help="用分号分隔，如: Mon_08:00;Tue_10:00"),
                },
                use_container_width=True,
                num_rows="dynamic"
            )
            
            # 转换回列表格式并同步到 session state
            if "class_groups" in edited_df.columns:
                edited_df["class_groups"] = edited_df["class_groups"].apply(
                    lambda x: [m.strip() for m in str(x).split(",")] if x and str(x) != "nan" else []
                )
            if "blocked_hours" in edited_df.columns:
                edited_df["blocked_hours"] = edited_df["blocked_hours"].apply(
                    lambda x: [b.strip() for b in str(x).split(";")] if x and str(x) != "nan" else []
                )
            
            st.session_state.courses_data = edited_df.to_dict(orient="records")
            
            # 简单的负载提示
            lab_req = len([c for c in st.session_state.courses_data if c['type']=='lab'])
            st.caption(f"📊 负载分析: 总课程 {len(st.session_state.courses_data)} | 机房需求 {lab_req} | 多媒体需求 {len(st.session_state.courses_data)-lab_req}")

# =============================================
# 4. 主界面：系统运行 (Execution)
# =============================================
st.divider()
st.subheader("🚀 第二步：智能排课计算")

col_btn, col_info = st.columns([1, 4])
with col_btn:
    start_btn = st.button("开始排课", type="primary", use_container_width=True)

if start_btn:
    if not st.session_state.courses_data:
        st.error("❌ 请先在上方生成或导入数据！")
    else:
        # 准备数据包
        # 时间片定义 (固定)
        times = [f"{d}_{t}" for d in ["Mon", "Tue", "Wed", "Thu", "Fri"] for t in ["08:00", "10:00", "14:00", "16:00", "19:00"]]
        
        problem_data = {
            "metadata": {"times": times},
            "rooms": rooms, # 来自 Sidebar 配置
            "courses": st.session_state.courses_data # 来自 Step 1
        }
        
        # 运行变量初始化
        result_schedule = []
        msg = ""
        history = []
        start_ts = time.time()
        
        try:
            # --- 算法分支 ---
            if "Greedy" in solver_mode:
                solver = GreedySolver(problem_data)
                result_schedule, msg = solver.solve()
                
            elif "Genetic" in solver_mode:
                # 进度条
                progress_bar = st.progress(0)
                status_txt = st.empty()
                
                # 回调函数
                def ga_callback(g, c): 
                    progress_bar.progress((g+1)/ga_gen)
                    status_txt.text(f"🧬 进化中... Generation {g+1}/{ga_gen} | Conflict Cost: {c:.2f}")
                
                solver = GASolver(problem_data, {"hard": w_hard, "soft": 10})
                result_schedule, history = solver.run(ga_pop, ga_gen, ga_callback)
                msg = f"进化完成 (Final Cost: {history[-1]})"
                
            elif "OR-Tools" in solver_mode:
                with st.spinner("🤖 正在构建数学模型并求解 (CP-SAT)..."):
                    solver = CPSATSolver(problem_data)
                    result_schedule, msg = solver.solve(time_limit=cp_timeout)
                    
            duration = time.time() - start_ts
            
            # --- 结果处理 ---
            if not result_schedule and "No Solution" in msg:
                st.error(f"❌ 排课失败: {msg}")
            else:
                st.success(f"✅ 排课完成! {msg} | 耗时: {duration:.3f}s")
                
                # =============================================
                # 5. 结果可视化 (Visualization)
                # =============================================
                st.markdown("### 📊 排课结果看板")
                
                tab_schedule, tab_analysis, tab_visual = st.tabs(["📅 最终课表视图", "📈 算法收敛分析", "🔍 交互可视化"])
                
                # >>> Tab 1: 课表视图 <<<
                with tab_schedule:
                    if not result_schedule:
                        st.warning("结果为空。")
                    else:
                        # 数据格式化（同时保留 course id 以便检测冲突）
                        display_list = []
                        for item in result_schedule:
                            c = item.get('course', {})
                            cid = c.get('id') or c.get('name')
                            content = f"📚{c.get('name','')}\n👤{c.get('teacher','')}"
                            if c.get('weeks') and c.get('weeks') != 'all':
                                content += f"\n[{'单周' if c.get('weeks')=='odd' else '双周'}]"
                            display_list.append({
                                "Time": item.get('time'),
                                "Room": item.get('room'),
                                "Content": content,
                                "CourseID": cid
                            })
                        
                        df_res = pd.DataFrame(display_list)
                        # 计算可视化中的冲突课程（同一 Time+Room 下出现多条记录）
                        conflict_ids = set()
                        if not df_res.empty:
                            grp = df_res.groupby(["Time", "Room"]).size()
                            conflict_cells = grp[grp > 1].index.tolist()
                            if conflict_cells:
                                for t, r in conflict_cells:
                                    ids = df_res[(df_res['Time'] == t) & (df_res['Room'] == r)]['CourseID'].tolist()
                                    for x in ids:
                                        if x is not None:
                                            conflict_ids.add(x)
                        # 透视表
                        pivot = df_res.pivot_table(
                            index="Time", columns="Room", values="Content",
                            aggfunc=lambda x: " || ".join(x)
                        ).fillna("-")
                        
                        # 排序：确保所有时间都显示，即使没有课程
                        pivot = pivot.reindex(times)
                        pivot = pivot.fillna("-")  # 缺失的时间点填充为 "-"
                        
                        # 样式高亮函数
                        def highlight_cells(val):
                            style = "white-space: pre-wrap; font-size: 12px; color: black;"
                            if "||" in str(val): # 冲突
                                return style + "background-color: #ff4b4b; border: 2px solid red;"
                            # 单/双周课程：保留浅蓝
                            if "[" in str(val) and "all" not in str(val): # 单双周
                                return style + "background-color: #e6f3ff; border-left: 4px solid #2196F3;"
                            # 普通课程：改为浅黄色以便更容易区分
                            if val != "-": # 普通课程
                                return style + "background-color: #fff9c4; border-radius: 4px;"
                            return style

                        st.dataframe(pivot.style.applymap(highlight_cells), height=600, use_container_width=True)
                        st.download_button("📥 导出课表 Excel", df_res.to_csv().encode('utf-8'), "schedule.csv")

                # >>> Tab 2: 算法分析 (优化后的统计逻辑) <<<
                with tab_analysis:
                    # 1. 数据准备
                    total_courses = len(st.session_state.courses_data)
                    
                    # 获取所有课程 ID 的集合
                    all_ids = set()
                    for c in st.session_state.courses_data:
                        cid = c.get("id") or c.get("name")
                        if cid: all_ids.add(cid)

                    # 获取“已出现在课表中的”课程 ID (不管是否有冲突)
                    scheduled_ids = set()
                    # 获取“发生冲突的”课程 ID
                    conflict_ids = set()
                    
                    if result_schedule:
                        # 临时 DataFrame 用于检测冲突
                        df_res_temp = pd.DataFrame([
                            {
                                "CourseID": item.get('course', {}).get('id') or item.get('course', {}).get('name'),
                                "Time": item.get('time'),
                                "Room": item.get('room')
                            } 
                            for item in result_schedule
                        ])
                        
                        # 记录所有已排 ID
                        scheduled_ids = set(df_res_temp['CourseID'].dropna().tolist())
                        
                        # 检测冲突: 同一时间同一教室出现多次
                        if not df_res_temp.empty:
                            grp = df_res_temp.groupby(["Time", "Room"]).size()
                            conflict_points = grp[grp > 1].index.tolist() # 哪些时间地点爆了
                            
                            for t, r in conflict_points:
                                # 找出在这个爆炸点涉及的所有课程
                                c_ids = df_res_temp[(df_res_temp['Time'] == t) & (df_res_temp['Room'] == r)]['CourseID'].tolist()
                                for cid in c_ids:
                                    conflict_ids.add(cid)

                    # 2. 核心指标计算
                    # 真正成功的 = 在课表里 - 在冲突列表里
                    success_ids = scheduled_ids - conflict_ids
                    # 完全丢失的 = 总数 - 在课表里
                    missing_ids = all_ids - scheduled_ids
                    
                    success_count = len(success_ids)
                    missing_count = len(missing_ids)
                    conflict_count = len(conflict_ids)
                    
                    # 计算成功率
                    success_rate = success_count / total_courses if total_courses > 0 else 0

                    # 3. 界面展示
                    st.markdown("#### 📊 排课质量仪表盘")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("总课程数", total_courses)
                    with m2:
                        st.metric("✅ 成功安排", success_count, delta=f"{success_rate:.1%}")
                    with m3:
                        st.metric("🚫 冲突课程", conflict_count, delta_color="inverse", help="虽然排进去了，但和其他课撞车了")
                    with m4:
                        st.metric("💨 未排/丢弃", missing_count, delta_color="inverse", help="没找到位置，直接被扔掉了")

                    st.divider()

                    c1, c2 = st.columns([2, 1])
                    
                    with c1:
                        st.markdown("#### ⚠️ 问题课程详情")
                        if missing_count == 0 and conflict_count == 0:
                            st.success("🎉 完美！所有课程均已妥善安排，无遗漏、无冲突。")
                        else:
                            # 制作一个错误报告 DataFrame
                            error_report = []
                            
                            # A. 冲突的课
                            if conflict_ids:
                                for item in result_schedule:
                                    c = item.get('course', {})
                                    cid = c.get('id') or c.get('name')
                                    if cid in conflict_ids:
                                        error_report.append({
                                            "课程名称": c.get('name'),
                                            "教师": c.get('teacher'),
                                            "状态": "❌ 发生冲突",
                                            "详情": f"在 {item.get('time')} {item.get('room')} 撞车"
                                        })
                            
                            # B. 丢失的课
                            if missing_ids:
                                for c in st.session_state.courses_data:
                                    cid = c.get('id') or c.get('name')
                                    if cid in missing_ids:
                                        error_report.append({
                                            "课程名称": c.get('name'),
                                            "教师": c.get('teacher'),
                                            "状态": "💨 未能排入",
                                            "详情": "资源不足或约束过严(如时间/教室不匹配)"
                                        })
                            
                            if error_report:
                                st.dataframe(pd.DataFrame(error_report), use_container_width=True)

                    with c2:
                        st.markdown("#### 📈 算法收敛趋势")
                        # 只有 GA 才有 History
                        if history and len(history) > 1:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.plot(history, color='#2ca02c', linewidth=2, label='Cost')
                            ax.set_title("Optimization Curve", fontsize=10)
                            ax.set_xlabel("Gen", fontsize=8)
                            ax.set_ylabel("Penalty", fontsize=8)
                            ax.grid(True, linestyle='--', alpha=0.5)
                            st.pyplot(fig)
                        elif "Genetic" in solver_mode:
                            st.info("迭代过少，无趋势图。")
                        else:
                            st.info(f"当前使用 **{solver_mode.split()[0]}** 算法，\n不产生收敛曲线。\n\n它通过逻辑判断直接给出结果。")
                            
                # >>> Tab 3: 交互可视化 (Heatmap + 按对象查看) <<<
                with tab_visual:
                    st.markdown("### 🔍 交互可视化：教室利用率与按对象查看")
                    if not result_schedule:
                        st.info("请先运行排课引擎以生成结果，然后查看可视化。")
                    else:
                        # 构建 schedule dataframe（重用 display_list 的结构）
                        vis_list = []
                        for item in result_schedule:
                            c = item.get('course', {})
                            vis_list.append({
                                'Time': item.get('time'),
                                'Room': item.get('room'),
                                'Teacher': c.get('teacher'),
                                'CourseName': c.get('name'),
                                'CourseID': c.get('id') or c.get('name')
                            })
                        vis_df = pd.DataFrame(vis_list)

                        # Heatmap: room x time occupancy (0/1)
                        occ_df = pd.DataFrame(0, index=[r['id'] for r in rooms], columns=times)
                        if not vis_df.empty:
                            for _, row in vis_df.iterrows():
                                t = row['Time']
                                r = row['Room']
                                if r in occ_df.index and t in occ_df.columns:
                                    occ_df.at[r, t] = occ_df.at[r, t] + 1

                        st.markdown('#### 教室利用热力图（颜色越深表示越繁忙）')
                        fig, ax = plt.subplots(figsize=(max(8, len(times)*0.6), max(3, len(rooms)*0.35)))
                        im = ax.imshow(occ_df.values, aspect='auto', cmap='Reds', interpolation='nearest')
                        ax.set_xticks(range(len(occ_df.columns)))
                        ax.set_xticklabels(occ_df.columns, rotation=45, ha='right')
                        ax.set_yticks(range(len(occ_df.index)))
                        ax.set_yticklabels(occ_df.index)
                        ax.set_title('教室 x 时间 利用热力图')
                        plt.colorbar(im, ax=ax, orientation='vertical', label='课程数')
                        st.pyplot(fig)

                        st.divider()
                        st.markdown('#### 按对象查看（教师 / 教室）')
                        view_mode = st.selectbox('选择查看对象', ['按老师查看', '按教室查看'])
                        if view_mode == '按老师查看':
                            teachers = sorted(set([c.get('teacher') for c in st.session_state.courses_data if c.get('teacher')]))
                            sel_t = st.selectbox('选择老师', ['全部'] + teachers)
                            if sel_t == '全部':
                                fdf = vis_df.copy()
                            else:
                                fdf = vis_df[vis_df['Teacher'] == sel_t]
                            if fdf.empty:
                                st.info('该老师暂无排课')
                            else:
                                pivot_t = fdf.pivot_table(index='Time', columns='Room', values='CourseName', aggfunc=lambda x: ' || '.join(x)).reindex(times).fillna('-')
                                st.table(pivot_t)

                        else:
                            room_ids = [r['id'] for r in rooms]
                            sel_r = st.selectbox('选择教室', ['全部'] + room_ids)
                            if sel_r == '全部':
                                fdf = vis_df.copy()
                            else:
                                fdf = vis_df[vis_df['Room'] == sel_r]
                            if fdf.empty:
                                st.info('该教室暂无排课')
                            else:
                                pivot_r = fdf.pivot_table(index='Time', columns='Room', values='CourseName', aggfunc=lambda x: ' || '.join(x)).reindex(times).fillna('-')
                                st.table(pivot_r)
        except Exception as e:
            st.error(f"❌ 系统运行出错: {e}")
            # 打印堆栈以便调试
            import traceback
            st.code(traceback.format_exc())