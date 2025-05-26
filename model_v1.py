from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpInteger

class full_model:
    def __init__(self):
        self.model = LpProblem("RMC_Delivery_Scheduling", LpMinimize)

        # ========== 参数定义 ==========
        self.b = 10  # 卡车队大小
        self.MN = 2  # 厂站搅拌机数量
        self.M = ['m1', 'm2']  # 工地集合
        self.d = {'m1': 5, 'm2': 3}  # 各工地需求
        self.B = 1e6  # 大数

        # 时间参数（分钟制，8:00=480）
        self.start_time = 480
        self.node_times = {i: self.start_time + 2*i for i in range(60)}  # 时间节点
        self.travel_time = {'m1': 14, 'm2': 22}  # 到工地的行程时间
        self.placement_time = {'m1': 4, 'm2': 6}  # 工地卸货时间
        self.cold_joint_time = {'m1': 4, 'm2': 4}  # 冷缝时间
        self.normal_end_time = 960  # 正常工作结束时间
        self.tf_m = {m: self.normal_end_time for m in self.M}  # 工地正常结束时间

        # 成本参数
        self.c1 =100  # 厂站正常成本/分钟
        self.c1_m = {'m1': 80, 'm2': 80}  # 工地正常成本/分钟
        self.c2 = 133# 厂站加班成本/分钟 
        self.c2_m = {'m1': 106, 'm2': 106}  # 工地加班成本/分钟

        # 定义各类弧的成本系数
        self.c_AP = 15  # 供应弧成本系数
        self.c_AT = 5   # 停留弧成本系数
        self.c_AS = 20  # 返回弧成本系数
        self.c_AD = 30  # 运输弧成本系数


        # ========== 集合定义 ==========
        self.s = 5555  # 开始节点
        self.f = 6666  # 结束节点
        self.NR = list(self.node_times.keys())  # 厂站节点
        self.N = [self.s, self.f] + self.NR # 所有节点

        # 弧集合定义
        self.AR = [(self.s, self.f)]  # Remainder Arcs (剩余弧)
        self.AP = [(self.s, n) for n in self.NR]  # Providing Arcs (供应弧)
        self.AS = [(self.NR[i], self.NR[i + 1]) for i in range(len(self.NR) - 1)]  # Staying Arcs (停留弧)
        self.AT = [(n, self.f) for n in self.NR]  # Return Arcs (返回弧)
        self.AD = {m: [] for m in self.M}  # Delivery Arcs (运输弧)
        for m in self.M:
            # 生成运输弧
            time_cost = self.placement_time[m] + 2 * self.travel_time[m]
            time_remain = self.NR[-1] * 2
            for n in self.NR:
                if time_remain < time_cost:
                    break
                time_remain = time_remain - 2
                self.AD[m].append((n, n + time_cost // 2))
        
        # 构建AM集合：键是NR中每个节点，值是从该点出发的运输弧
        self.AM = {}
        for n in self.NR:
            self.AM[n] = []
            for m in self.M:
                time_cost = self.placement_time[m] + 2 * self.travel_time[m]
                time_remain = (self.NR[-1] - n) * 2
                if time_remain < time_cost:
                    break
                self.AM[n].append((n, n + time_cost // 2))


        # 定义成本参数 
        self.c_ij = {}

        # 剩余弧成本
        for arc in self.AR:
            self.c_ij[arc] = 0
        
        # 供应弧成本
        for arc in self.AP:
            self.c_ij[arc] = self.c_AP
        
        # 停留弧成本
        for arc in self.AT:
            self.c_ij[arc] = self.c_AT 
        
        # 返回弧成本
        for arc in self.AS:
            self.c_ij[arc] = 0
        
        # 运输弧成本（不区分工地m）
        for m in self.M:
            for arc in self.AD[m]:
                self.c_ij[arc] = self.c_AD * (arc[1] - arc[0]) * 2  # 时间差乘以成本系数

        # ========== 变量定义 ==========
        # 流量变量
        self.x = {}
        # 整数变量：剩余弧、供应弧、停留弧、返回弧
        for arc in self.AR + self.AP + self.AT + self.AS:
            self.x[arc] = LpVariable(f"x_{arc[0]}_{arc[1]}", 0, None, LpInteger)
        for m in self.M:
            for arc in self.AD[m]:
                self.x[arc] = LpVariable(f"x_{arc[0]}_{arc[1]}", 0, None, LpInteger)
                
        # 构建冷缝束和排队束
        self.CPB = {m: {} for m in self.M}  # 冷缝束集合 Cold-joint-Preventing Bundle
        self.QPB = {m: {} for m in self.M}  # 排队束集合 Queuing-Preventing Bundle

        for m in self.M:
            t1 = self.cold_joint_time[m]
            t2 = self.placement_time[m]
            # 计算束大小
            cpb_size = (t1 + t2) // 2  # 冷缝束大小
            qpb_size = t2 // 2  # 排队束大小
            # 生成冷缝束
            for a in range(self.travel_time[m] // 2, len(self.NR), cpb_size):
                # 计算冷缝束的节点范围
                end_node = min(a + int(cpb_size), len(self.NR) - 1)
                # 创建冷缝束的键
                self.CPB[m][a] = []
                for node_idx in range(a, end_node + 1):
                    node = node_idx - self.travel_time[m] // 2
                    end_node_val = node + self.travel_time[m] + self.placement_time[m] // 2
                    if end_node_val < len(self.NR):
                        self.CPB[m][a].append((node, end_node_val))
            # 生成排队束
            for a in range(self.travel_time[m] // 2, len(self.NR), qpb_size):
                # 计算排队束的节点范围
                end_node = min(a + int(qpb_size), len(self.NR) - 1)
                # 创建排队束的键
                self.QPB[m][a] = []
                for node_idx in range(a, end_node + 1):
                    node = node_idx - self.travel_time[m] // 2
                    end_node_val = node + self.travel_time[m] + self.placement_time[m] // 2
                    if end_node_val < len(self.NR):
                        self.QPB[m][a].append((node, end_node_val))

        # 补充定义冷缝束的第一个和最后一个弧的时间
        self.ta_am = {}  # 冷缝束a的最后一个弧到达工地m的时间
        self.tl_am = {}  # 冷缝束a的第一个弧离开工地m的时间
        for m in self.M:
            self.ta_am[m] = {}
            self.tl_am[m] = {}
            for cpb in self.CPB[m].keys():
                self.tl_am[m][cpb] = cpb * 2 + self.placement_time[m]
                self.ta_am[m][cpb] = cpb * 2 + self.cold_joint_time[m] + self.placement_time[m] - 2

        # 时间变量
        self.ts = LpVariable("ts", 0, None)  # 厂站开始时间
        self.te = LpVariable("te", 0, None)  # 厂站结束时间
        self.ts_m = {m: LpVariable(f"ts_{m}", 0, None) for m in self.M}  # 工地开始时间
        self.te_m = {m: LpVariable(f"te_{m}", 0, None) for m in self.M}  # 工地结束时间

        # 加班变量
        self.w = LpVariable("w", 0, None)  # 厂站加班时间
        self.w_m = {m: LpVariable(f"w_{m}", 0, None) for m in self.M}  # 工地加班时间

        # 二进制指示变量
        self.y_if = {n: LpVariable(f"y_if_{n}", 0, 1, LpBinary) for n in self.NR}  # 返回弧激活
        self.y_si = {n: LpVariable(f"y_si_{n}", 0, 1, LpBinary) for n in self.NR}  # 供应弧激活
        self.y1_am = {(cpb, m): LpVariable(f"y1_{cpb}_{m}", 0, 1, LpBinary) for m in self.M for cpb in self.CPB[m].keys()}  # 冷缝束指示变量1
        self.y2_am = {(cpb, m): LpVariable(f"y2_{cpb}_{m}", 0, 1, LpBinary) for m in self.M for cpb in self.CPB[m].keys()}  # 冷缝束指示变量2


        # ========== 目标函数 ==========
        self.model += (
            lpSum(self.c_ij.get(arc, 0) * self.x[arc] for arc in self.x) +  # 运输成本
            self.c1* (self.te - self.ts) +  # 厂站正常时间成本
            lpSum(self.c1_m[m] * (self.te_m[m] - self.ts_m[m]) for m in self.M) +  # 工地正常时间成本
            self.c2 * self.w +  # 厂站加班成本
            lpSum(self.c2_m[m] * self.w_m[m] for m in self.M)  # 工地加班成本
        )

        # ========== 约束条件 ==========
        # 1. 流量平衡约束
        for i in self.N:
            inflow = lpSum(self.x[(k, i)] for k in self.N if (k, i) in self.x)
            outflow = lpSum(self.x[(i, j)] for j in self.N if (i, j) in self.x)
            if i == self.s:
                self.model += (outflow - inflow == self.b)  # 约束2: 从起点流出b辆车
            elif i == self.f:
                self.model += (outflow - inflow == -self.b)  # 约束2: 到终点流入b辆车
            else:
                self.model += (outflow - inflow == 0)  # 约束2: 中间节点流入=流出

        # 2. 厂站容量约束
        for n in self.NR:
            self.model += lpSum(self.x[(i, j)] for (i, j) in self.AM[n]) <= self.MN  # 约束3: 同时发车数≤搅拌机数

        # 3. 需求约束
        for m in self.M:
            # 修复：确保只计算到达工地m的弧
            self.model += lpSum(self.x[(i, j)] for (i, j) in self.AD[m]) >= self.d[m]  # 约束4: 到工地的运输量≥需求

        # 4. 时间约束
        # 厂站结束时间
        for n in self.NR:
            self.model += self.te >= self.node_times[n] - self.B * (1 - self.y_if[n])  # 约束5: 结束时间≥最后一辆车的时间
            self.model += self.x[(n, self.f)] <= self.b * self.y_if[n]  # 约束6: 激活y_if当有车从n返回

        # 厂站开始时间 
        for n in self.NR:
            self.model += self.ts <= self.node_times[n] + self.B * (1 - self.y_si[n])  # 约束7: 开始时间≤第一辆车的时间
            self.model += self.x[(self.s, n)] <= self.b * self.y_si[n]  # 约束8: 激活y_si当有车从s出发到n

        # 工地时间约束
        for m in self.M:
            # 工地开始时间：最早到达工地的车辆时间
            for (i, j) in self.AD[m]:
                    # 约束9: 如果有车从n到m，则工地开始时间≤该车到达时间+大数*(1-标识变量)
                self.model += self.ts_m[m] <= self.node_times[i] + self.B * (1 - self.x[(i, j)])
            # 工地结束时间：最后离开工地的车辆时间 = 到达时间 + 卸货时间
            for (i, j) in self.AD[m]:
                # 约束10: 如果有车从n到m，则工地结束时间≥该车离开时间-大数*(1-标识变量)
                self.model += self.te_m[m] >= self.node_times[i] + self.placement_time[m] - self.B * (1 - self.x[(i, j)])

        # 5. 冷缝约束 
        for m in self.M:
            for cpb in self.CPB[m].keys():
                # 约束13: 冷缝束内流量约束
                self.model += lpSum(self.x[arc] for arc in self.CPB[m][cpb]) >= self.y1_am[(cpb, m)] + self.y2_am[(cpb, m)] - 1
                self.model += lpSum(self.x[arc] for arc in self.CPB[m][cpb]) <= self.B * (self.y1_am[(cpb, m)] + self.y2_am[(cpb, m)] - 1)
        
                # 约束14: 时间窗口约束
                self.model += self.te_m[m] <= self.tl_am[m][cpb] + self.B * self.y1_am[(cpb, m)]
        
                # 约束15: 时间窗口约束
                self.model += self.ts_m[m] >= self.ta_am[m][cpb] - self.B * self.y2_am[(cpb, m)]

        # 6. 排队约束
        for m in self.M:
            for qpb in self.QPB[m].keys():
                # 约束16: 每个排队束中最多只能有一辆车
                self.model += lpSum(self.x[arc] for arc in self.QPB[m][qpb]) <= 1

        # 7. 加班约束
        # 约束11: 厂站加班时间 = max(0, 结束时间 - 正常结束时间)
        self.model += self.w >= self.te - self.normal_end_time
        

        # 工地加班约束
        for m in self.M:
            # 约束12: 工地加班时间 = max(0, 结束时间 - 正常结束时间)
            self.model += self.w_m[m] >= self.te_m[m] - self.tf_m[m]

        # 8. 变量域约束
        # 约束17: 时间变量非负
        self.model += self.ts >= 0
        self.model += self.te >= 0
        self.model += self.w >=0  # 加班时间非负
        for m in self.M:
            self.model += self.ts_m[m] >= 0
            self.model += self.te_m[m] >= 0
            self.model += self.w_m[m] >= 0  # 加班时间非负

        # 约束18: 运输弧流量为二进制变量
        for m in self.M:
            for arc in self.AD[m]:
                self.model += self.x[arc] <= 1  # 二进制变量上界
                self.model += self.x[arc] >= 0  # 二进制变量下界

        # 约束19: 其他弧流量为非负整数（移除重复约束）
        for arc in self.AR + self.AP + self.AT + self.AS:
            self.model += self.x[arc] >= 0  # 非负整数流量

        # 约束20: 冷缝束指示变量为二进制变量
        for m in self.M:
            for cpb in self.CPB[m].keys():
                self.model += self.y1_am[(cpb, m)] <= 1
                self.model += self.y1_am[(cpb, m)] >= 0
                self.model += self.y2_am[(cpb, m)] <= 1
                self.model += self.y2_am[(cpb, m)] >= 0

        # 约束21: 其他二进制变量约束
        for n in self.NR:
            self.model += self.y_if[n] <= 1
            self.model += self.y_if[n] >= 0
            self.model += self.y_si[n] <= 1
            self.model += self.y_si[n] >= 0

