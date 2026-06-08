"""
NSGA-II多目标遗传算法优化模块
用于污水处理工艺多目标优化
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import copy
import random

from .asm1_model import ASM1Parameters
from .reactor_units import ProcessFlowSheet, ReactorType
from .process_templates import InfluentConfig
from .solver import solve_steady_state, SolverConfig
from .analysis import (
    calculate_sludge_production,
    calculate_energy_consumption,
    check_compliance,
    STANDARDS,
    ComplianceResult,
)


@dataclass
class OptimizationVariable:
    """优化变量定义"""
    name: str
    display_name: str
    unit: str
    min_value: float
    max_value: float
    default_value: float
    description: str = ""


@dataclass
class OptimizationObjective:
    """优化目标定义"""
    name: str
    display_name: str
    unit: str
    direction: str  # 'minimize' or 'maximize'
    description: str = ""


@dataclass
class OptimizationConfig:
    """优化配置"""
    variables: List[OptimizationVariable] = field(default_factory=list)
    objectives: List[OptimizationObjective] = field(default_factory=list)
    objective_weights: Dict[str, float] = field(default_factory=dict)
    use_pareto: bool = True
    population_size: int = 50
    max_generations: int = 100
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    crossover_distribution_index: float = 20.0
    mutation_distribution_index: float = 20.0
    constraint_standard: str = '一级A'


@dataclass
class Individual:
    """种群个体"""
    variables: np.ndarray  # 决策变量值 [num_vars]
    objectives: np.ndarray = field(default_factory=lambda: np.array([]))  # 目标值 [num_objs]
    constraints: np.ndarray = field(default_factory=lambda: np.array([]))  # 约束违反值
    constraint_violation: float = 0.0  # 总约束违反量
    rank: int = 0  # 非支配排序层级
    crowding_distance: float = 0.0  # 拥挤距离
    is_feasible: bool = True  # 是否满足约束
    converged: bool = False  # 求解是否收敛
    effluent_quality: Dict[str, float] = field(default_factory=dict)
    energy_result: Optional[object] = None
    sludge_result: Optional[object] = None
    compliance_result: Optional[ComplianceResult] = None


@dataclass
class OptimizationResult:
    """优化结果"""
    config: OptimizationConfig
    pareto_front: List[Individual] = field(default_factory=list)
    all_populations: List[List[Individual]] = field(default_factory=list)
    best_fitness_history: List[float] = field(default_factory=list)
    avg_fitness_history: List[float] = field(default_factory=list)
    total_evaluations: int = 0
    was_aborted: bool = False


DEFAULT_VARIABLES = [
    OptimizationVariable(
        name='DO_setpoint',
        display_name='好氧池DO设定值',
        unit='mg/L',
        min_value=0.5,
        max_value=4.0,
        default_value=2.0,
        description='好氧池溶解氧浓度设定值',
    ),
    OptimizationVariable(
        name='internal_return_ratio',
        display_name='缺氧池内回流比',
        unit='%',
        min_value=50.0,
        max_value=400.0,
        default_value=200.0,
        description='混合液内回流比',
    ),
    OptimizationVariable(
        name='SRT',
        display_name='好氧池SRT',
        unit='天',
        min_value=5.0,
        max_value=30.0,
        default_value=15.0,
        description='污泥停留时间',
    ),
    OptimizationVariable(
        name='return_sludge_ratio',
        display_name='回流污泥比',
        unit='%',
        min_value=50.0,
        max_value=150.0,
        default_value=50.0,
        description='二沉池污泥回流比',
    ),
]


DEFAULT_OBJECTIVES = [
    OptimizationObjective(
        name='NH3_N',
        display_name='出水NH3-N',
        unit='mg/L',
        direction='minimize',
        description='出水氨氮浓度',
    ),
    OptimizationObjective(
        name='TN',
        display_name='出水TN',
        unit='mg/L',
        direction='minimize',
        description='出水总氮浓度',
    ),
    OptimizationObjective(
        name='energy',
        display_name='日均能耗',
        unit='kWh/d',
        direction='minimize',
        description='系统日均电耗',
    ),
    OptimizationObjective(
        name='sludge',
        display_name='日产泥量',
        unit='kg DS/d',
        direction='minimize',
        description='每日剩余污泥产量',
    ),
]


class NSGA2Optimizer:
    """NSGA-II多目标遗传算法优化器"""

    def __init__(
        self,
        config: OptimizationConfig,
        pfs: ProcessFlowSheet,
        influent: InfluentConfig,
        asm1_params: ASM1Parameters,
        solver_config: Optional[SolverConfig] = None,
    ):
        self.config = config
        self.pfs = pfs
        self.influent = influent
        self.asm1_params = asm1_params
        self.solver_config = solver_config or SolverConfig()
        self.solver_config.max_iterations = 50
        self.solver_config.tolerance = 1e-3
        self.solver_config.use_engineering_tolerance = True
        
        self.num_vars = len(config.variables)
        self.num_objs = len(config.objectives)
        self._stop_event = False

    def stop(self):
        """提前终止优化"""
        self._stop_event = True

    def _initialize_population(self, population_size: int) -> List[Individual]:
        """初始化种群"""
        population = []
        for _ in range(population_size):
            variables = np.zeros(self.num_vars)
            for i, var in enumerate(self.config.variables):
                variables[i] = random.uniform(var.min_value, var.max_value)
            population.append(Individual(variables=variables))
        return population

    def _decode_variables(self, individual: Individual) -> Dict[str, float]:
        """将个体变量解码为参数字典"""
        params = {}
        for i, var in enumerate(self.config.variables):
            params[var.name] = individual.variables[i]
        return params

    def _apply_parameters(self, params: Dict[str, float]) -> ProcessFlowSheet:
        """将参数应用到工艺流程"""
        pfs_copy = copy.deepcopy(self.pfs)
        
        do_setpoint = params.get('DO_setpoint', 2.0)
        internal_return = params.get('internal_return_ratio', 200.0) / 100.0
        srt = params.get('SRT', 15.0)
        return_ratio = params.get('return_sludge_ratio', 50.0) / 100.0
        
        for reactor in pfs_copy.reactors:
            if reactor.reactor_type == ReactorType.AEROBIC:
                reactor.operation.DO_setpoint = do_setpoint
                reactor.operation.SRT = srt
            
            if hasattr(reactor.operation, 'internal_return_ratio'):
                reactor.operation.internal_return_ratio = internal_return
            
            if hasattr(reactor.operation, 'return_sludge_ratio'):
                if reactor.reactor_type == ReactorType.SECONDARY or reactor.is_biological():
                    reactor.operation.return_sludge_ratio = return_ratio
        
        for reactor in pfs_copy.reactors:
            if reactor.is_biological() and hasattr(reactor.operation, 'SRT'):
                reactor.operation.SRT = srt
        
        return pfs_copy

    def _evaluate_individual(self, individual: Individual) -> Individual:
        """评估单个个体的适应度"""
        params = self._decode_variables(individual)
        pfs_copy = self._apply_parameters(params)
        
        try:
            result = solve_steady_state(
                pfs_copy, self.influent, self.asm1_params, self.solver_config
            )
            
            individual.converged = result.converged
            
            if not result.converged:
                individual.is_feasible = False
                individual.constraint_violation = 1e6
                individual.objectives = np.full(self.num_objs, 1e6)
                return individual
            
            individual.effluent_quality = result.effluent_quality.copy()
            
            compliance = check_compliance(
                result.effluent_quality, self.config.constraint_standard
            )
            individual.compliance_result = compliance
            
            constraint_violations = []
            standard = STANDARDS.get(self.config.constraint_standard, STANDARDS['一级A'])
            
            constraints_to_check = [
                ('COD', standard.COD),
                ('NH3_N', standard.NH3_N),
                ('TN', standard.TN),
                ('TP', standard.TP),
            ]
            
            total_violation = 0.0
            for key, limit in constraints_to_check:
                value = result.effluent_quality.get(key, 1e6)
                violation = max(0, value - limit)
                if violation > 0:
                    total_violation += (violation / limit) ** 2
                constraint_violations.append(violation)
            
            individual.constraints = np.array(constraint_violations)
            individual.constraint_violation = total_violation
            individual.is_feasible = total_violation < 1e-6
            
            try:
                Q = self.influent.Q_base
                sludge_result = calculate_sludge_production(
                    pfs_copy, result.reactor_states, Q, self.asm1_params
                )
                energy_result = calculate_energy_consumption(
                    pfs_copy, result.reactor_states, self.influent, self.asm1_params
                )
                individual.sludge_result = sludge_result
                individual.energy_result = energy_result
            except:
                sludge_result = None
                energy_result = None
            
            objectives = np.zeros(self.num_objs)
            for i, obj in enumerate(self.config.objectives):
                if obj.name == 'NH3_N':
                    val = result.effluent_quality.get('NH3_N', 1e6)
                elif obj.name == 'TN':
                    val = result.effluent_quality.get('TN', 1e6)
                elif obj.name == 'energy':
                    val = energy_result.total_kwh_d if energy_result else 1e6
                elif obj.name == 'sludge':
                    val = sludge_result.daily_sludge_kg if sludge_result else 1e6
                else:
                    val = result.effluent_quality.get(obj.name, 1e6)
                
                if obj.direction == 'maximize':
                    val = -val
                
                objectives[i] = val
            
            individual.objectives = objectives
            
            if not individual.is_feasible:
                penalty = 1e3 * (1 + individual.constraint_violation)
                individual.objectives = individual.objectives + penalty
            
        except Exception as e:
            individual.converged = False
            individual.is_feasible = False
            individual.constraint_violation = 1e6
            individual.objectives = np.full(self.num_objs, 1e6)
        
        return individual

    def _evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """评估整个种群"""
        return [self._evaluate_individual(ind) for ind in population]

    def _fast_non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """快速非支配排序"""
        fronts = [[]]
        for i, p in enumerate(population):
            p.domination_count = 0
            p.dominated_set = []
            
            for j, q in enumerate(population):
                if i == j:
                    continue
                
                if self._dominates(p, q):
                    p.dominated_set.append(j)
                elif self._dominates(q, p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for j in p.dominated_set:
                    q = population[j]
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]

    def _dominates(self, p: Individual, q: Individual) -> bool:
        """判断个体p是否支配个体q"""
        if not p.is_feasible and not q.is_feasible:
            return p.constraint_violation < q.constraint_violation
        if not p.is_feasible:
            return False
        if not q.is_feasible:
            return True
        
        better_in_all = True
        better_in_at_least_one = False
        
        for i in range(self.num_objs):
            if p.objectives[i] > q.objectives[i]:
                better_in_all = False
            elif p.objectives[i] < q.objectives[i]:
                better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one

    def _calculate_crowding_distance(self, front: List[Individual]):
        """计算拥挤距离"""
        n = len(front)
        if n <= 2:
            for ind in front:
                ind.crowding_distance = float('inf')
            return
        
        for ind in front:
            ind.crowding_distance = 0.0
        
        for i in range(self.num_objs):
            front.sort(key=lambda x: x.objectives[i])
            
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            min_obj = front[0].objectives[i]
            max_obj = front[-1].objectives[i]
            obj_range = max_obj - min_obj
            
            if obj_range < 1e-10:
                continue
            
            for j in range(1, n - 1):
                distance = (front[j + 1].objectives[i] - front[j - 1].objectives[i]) / obj_range
                front[j].crowding_distance += distance

    def _tournament_selection(self, population: List[Individual], tournament_size: int = 2) -> Individual:
        """锦标赛选择"""
        candidates = random.sample(population, tournament_size)
        
        best = candidates[0]
        for cand in candidates[1:]:
            if self._compare_individuals(cand, best) < 0:
                best = cand
        
        return best

    def _compare_individuals(self, a: Individual, b: Individual) -> int:
        """比较两个个体，返回-1表示a更好，1表示b更好，0表示相等"""
        if a.rank != b.rank:
            return -1 if a.rank < b.rank else 1
        
        if a.crowding_distance != b.crowding_distance:
            return -1 if a.crowding_distance > b.crowding_distance else 1
        
        return 0

    def _sbx_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """模拟二进制交叉(SBX)"""
        eta = self.config.crossover_distribution_index
        
        child1_vars = parent1.variables.copy()
        child2_vars = parent2.variables.copy()
        
        if random.random() < self.config.crossover_probability:
            for i in range(self.num_vars):
                if random.random() < 0.5:
                    y1 = parent1.variables[i]
                    y2 = parent2.variables[i]
                    
                    if abs(y1 - y2) > 1e-10:
                        var = self.config.variables[i]
                        yl = var.min_value
                        yu = var.max_value
                        
                        if y1 > y2:
                            y1, y2 = y2, y1
                        
                        rand = random.random()
                        
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - beta ** (-(eta + 1.0))
                        
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        
                        c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                        
                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - beta ** (-(eta + 1.0))
                        
                        if rand <= 1.0 / alpha:
                            beta_q = (rand * alpha) ** (1.0 / (eta + 1.0))
                        else:
                            beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))
                        
                        c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                        
                        c1 = max(yl, min(yu, c1))
                        c2 = max(yl, min(yu, c2))
                        
                        child1_vars[i] = c1
                        child2_vars[i] = c2
        
        return Individual(variables=child1_vars), Individual(variables=child2_vars)

    def _polynomial_mutation(self, individual: Individual) -> Individual:
        """多项式变异"""
        eta = self.config.mutation_distribution_index
        mutated_vars = individual.variables.copy()
        
        for i in range(self.num_vars):
            if random.random() < self.config.mutation_probability:
                var = self.config.variables[i]
                y = individual.variables[i]
                yl = var.min_value
                yu = var.max_value
                
                delta1 = (y - yl) / (yu - yl)
                delta2 = (yu - y) / (yu - yl)
                
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                
                y = y + delta_q * (yu - yl)
                y = max(yl, min(yu, y))
                mutated_vars[i] = y
        
        return Individual(variables=mutated_vars)

    def _combine_populations(self, parent_pop: List[Individual], offspring_pop: List[Individual]) -> List[Individual]:
        """合并父代和子代种群"""
        combined = parent_pop + offspring_pop
        
        fronts = self._fast_non_dominated_sort(combined)
        
        new_population = []
        front_idx = 0
        
        while len(new_population) + len(fronts[front_idx]) <= self.config.population_size:
            self._calculate_crowding_distance(fronts[front_idx])
            new_population.extend(fronts[front_idx])
            front_idx += 1
            
            if front_idx >= len(fronts):
                break
        
        if len(new_population) < self.config.population_size and front_idx < len(fronts):
            self._calculate_crowding_distance(fronts[front_idx])
            fronts[front_idx].sort(key=lambda x: x.crowding_distance, reverse=True)
            needed = self.config.population_size - len(new_population)
            new_population.extend(fronts[front_idx][:needed])
        
        return new_population

    def _calculate_weighted_fitness(self, individual: Individual) -> float:
        """计算加权适应度（用于收敛曲线）"""
        if not self.config.use_pareto and self.config.objective_weights:
            fitness = 0.0
            for i, obj in enumerate(self.config.objectives):
                weight = self.config.objective_weights.get(obj.name, 1.0)
                fitness += weight * individual.objectives[i]
            return fitness
        else:
            if len(individual.objectives) > 0:
                return np.sum(individual.objectives)
            return float('inf')

    def optimize(
        self,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        stop_check_callback: Optional[Callable[[], bool]] = None,
    ) -> OptimizationResult:
        """
        运行NSGA-II优化
        
        参数:
            progress_callback: 进度回调(current_gen, max_gen, best_fitness, avg_fitness)
            stop_check_callback: 终止检查回调，返回True表示需要终止
        
        返回:
            OptimizationResult
        """
        self._stop_event = False
        
        result = OptimizationResult(config=self.config)
        
        population = self._initialize_population(self.config.population_size)
        population = self._evaluate_population(population)
        result.total_evaluations = len(population)
        
        fronts = self._fast_non_dominated_sort(population)
        for front in fronts:
            self._calculate_crowding_distance(front)
        
        best_fitness = min(self._calculate_weighted_fitness(ind) for ind in population)
        avg_fitness = np.mean([self._calculate_weighted_fitness(ind) for ind in population])
        result.best_fitness_history.append(best_fitness)
        result.avg_fitness_history.append(avg_fitness)
        result.all_populations.append(copy.deepcopy(population))
        
        if progress_callback is not None:
            progress_callback(0, self.config.max_generations, best_fitness, avg_fitness)
        
        for generation in range(self.config.max_generations):
            if self._stop_event:
                result.was_aborted = True
                break
            
            if stop_check_callback is not None and stop_check_callback():
                result.was_aborted = True
                break
            
            offspring = []
            for _ in range(self.config.population_size // 2):
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child1, child2 = self._sbx_crossover(parent1, parent2)
                child1 = self._polynomial_mutation(child1)
                child2 = self._polynomial_mutation(child2)
                offspring.extend([child1, child2])
            
            while len(offspring) < self.config.population_size:
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child1, child2 = self._sbx_crossover(parent1, parent2)
                offspring.append(child1)
            
            offspring = self._evaluate_population(offspring)
            result.total_evaluations += len(offspring)
            
            population = self._combine_populations(population, offspring)
            
            best_fitness = min(self._calculate_weighted_fitness(ind) for ind in population)
            avg_fitness = np.mean([self._calculate_weighted_fitness(ind) for ind in population])
            result.best_fitness_history.append(best_fitness)
            result.avg_fitness_history.append(avg_fitness)
            result.all_populations.append(copy.deepcopy(population))
            
            if progress_callback is not None:
                progress_callback(
                    generation + 1, self.config.max_generations, best_fitness, avg_fitness
                )
        
        final_fronts = self._fast_non_dominated_sort(population)
        if len(final_fronts) > 0:
            pareto_front = final_fronts[0]
            pareto_front = [ind for ind in pareto_front if ind.converged]
            result.pareto_front = pareto_front
        
        return result


def get_default_config() -> OptimizationConfig:
    """获取默认优化配置"""
    return OptimizationConfig(
        variables=copy.deepcopy(DEFAULT_VARIABLES),
        objectives=copy.deepcopy(DEFAULT_OBJECTIVES),
        objective_weights={'NH3_N': 0.3, 'TN': 0.3, 'energy': 0.2, 'sludge': 0.2},
        use_pareto=True,
        population_size=50,
        max_generations=100,
    )


def calculate_composite_score(individual: Individual, weights: Dict[str, float]) -> float:
    """
    计算综合评分（用于推荐排序）
    
    对各目标值进行归一化后加权求和
    """
    if not individual.converged or not individual.effluent_quality:
        return float('inf')
    
    objectives = individual.objectives
    if len(objectives) == 0:
        return float('inf')
    
    normalized = []
    for i, obj in enumerate(DEFAULT_OBJECTIVES):
        if obj.name == 'NH3_N':
            max_val = 5.0
            val = individual.effluent_quality.get('NH3_N', max_val)
            norm = min(val / max_val, 1.0)
        elif obj.name == 'TN':
            max_val = 15.0
            val = individual.effluent_quality.get('TN', max_val)
            norm = min(val / max_val, 1.0)
        elif obj.name == 'energy':
            max_val = 2000.0
            val = individual.energy_result.total_kwh_d if individual.energy_result else max_val
            norm = min(val / max_val, 1.0)
        elif obj.name == 'sludge':
            max_val = 1000.0
            val = individual.sludge_result.daily_sludge_kg if individual.sludge_result else max_val
            norm = min(val / max_val, 1.0)
        else:
            norm = 0.5
        
        weight = weights.get(obj.name, 0.25)
        normalized.append(weight * norm)
    
    score = sum(normalized)
    
    if not individual.is_feasible:
        score += 10.0
    
    return score
