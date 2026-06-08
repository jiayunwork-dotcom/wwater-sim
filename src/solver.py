"""
求解器模块
包含稳态求解(Newton-Raphson)和动态仿真(ODE BDF)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import copy
from scipy.integrate import solve_ivp

from .asm1_model import ASM1Parameters, NUM_COMPONENTS, COMPONENT_INDEX, aggregate_to_wq_indices
from .reactor_units import (
    ProcessFlowSheet,
    ReactorUnit,
    CSTRReactor,
    SecondaryClarifier,
    MembraneUnit,
    SimpleTreatment,
    ReactorType,
)
from .process_templates import InfluentConfig


@dataclass
class SolverConfig:
    """求解器配置"""
    max_iterations: int = 100
    tolerance: float = 1e-4
    relaxation: float = 0.3
    simulation_days: float = 7.0
    output_interval_days: float = 0.01
    method: str = 'BDF'
    rtol: float = 1e-6
    atol: float = 1e-8
    use_damped_newton: bool = True
    min_relaxation: float = 0.05
    warmup_days: float = 0.0
    steady_state_method: str = 'hybrid'  # 'newton', 'dynamic', 'hybrid'
    dynamic_steady_days: float = 30.0
    check_steady_every_days: float = 5.0
    steady_state_rtol: float = 1e-3
    use_engineering_tolerance: bool = True  # 是否使用工程收敛标准
    engineering_tolerance_factor: float = 50000.0  # 工程容差是tolerance的倍数


@dataclass
class SteadyStateResult:
    """稳态求解结果"""
    converged: bool
    iterations: int
    final_residual: float
    residual_history: List[float]
    reactor_states: List[np.ndarray]
    effluent_quality: Dict[str, float]
    message: str = ""


@dataclass
class DynamicResult:
    """动态仿真结果"""
    success: bool
    time_days: np.ndarray
    reactor_states: List[np.ndarray]
    effluent_quality_history: List[Dict[str, float]]
    message: str = ""


def calculate_system_residual(C_flat: np.ndarray, pfs: ProcessFlowSheet,
                               C_in: np.ndarray, Q: float,
                               params: ASM1Parameters) -> np.ndarray:
    """
    计算系统的残差向量 (稳态时 dC/dt = 0)
    
    参数:
        C_flat: 展平的反应器浓度 [num_reactors * 13]
        pfs: 工艺流程
        C_in: 进水浓度 [13]
        Q: 进水流量
        params: ASM1参数
    
    返回:
        残差向量
    """
    num_reactors = len(pfs.reactors)
    residual = np.zeros(num_reactors * NUM_COMPONENTS)
    
    return_sludge = None
    internal_return = None
    
    secondary_idx = None
    for i, reactor in enumerate(pfs.reactors):
        if isinstance(reactor, SecondaryClarifier):
            secondary_idx = i
            break
    
    membrane_idx = None
    for i, reactor in enumerate(pfs.reactors):
        if isinstance(reactor, MembraneUnit):
            membrane_idx = i
            break
    
    for i, reactor in enumerate(pfs.reactors):
        C_reactor = C_flat[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS]
        
        if i == 0:
            reactor_in = C_in
        else:
            prev_C = C_flat[(i - 1) * NUM_COMPONENTS:i * NUM_COMPONENTS]
            
            if isinstance(pfs.reactors[i - 1], SecondaryClarifier):
                prev_reactor = pfs.reactors[i - 1]
                prev_reactor.C = prev_C
                effluent, rs_C, _ = prev_reactor.process(prev_C, Q, params)
                reactor_in = effluent
            elif isinstance(pfs.reactors[i - 1], MembraneUnit):
                prev_reactor = pfs.reactors[i - 1]
                prev_reactor.C = prev_C
                effluent, _ = prev_reactor.process(prev_C, Q, params)
                reactor_in = effluent
            elif isinstance(pfs.reactors[i - 1], SimpleTreatment):
                prev_reactor = pfs.reactors[i - 1]
                prev_reactor.C = prev_C
                reactor_in = prev_reactor.process(prev_C, Q, params)
            else:
                reactor_in = prev_C
        
        return_sludge_to_use = None
        if secondary_idx is not None and i == 0:
            sec_C = C_flat[secondary_idx * NUM_COMPONENTS:(secondary_idx + 1) * NUM_COMPONENTS]
            sec_reactor = pfs.reactors[secondary_idx]
            sec_reactor.C = sec_C
            effluent, rs_C, _ = sec_reactor.process(sec_C, Q, params)
            Q_r = sec_reactor.operation.return_sludge_ratio * Q
            return_sludge_to_use = (Q_r, rs_C)
        
        if membrane_idx is not None and i == 0:
            mem_C = C_flat[membrane_idx * NUM_COMPONENTS:(membrane_idx + 1) * NUM_COMPONENTS]
            mem_reactor = pfs.reactors[membrane_idx]
            mem_reactor.C = mem_C
            _, retentate = mem_reactor.process(mem_C, Q, params)
            Q_r = mem_reactor.operation.return_sludge_ratio * Q
            return_sludge_to_use = (Q_r, retentate)
        
        internal_return_to_use = None
        if hasattr(reactor.operation, 'internal_return_ratio') and reactor.operation.internal_return_ratio > 0:
            if i == 1 and len(pfs.reactors) > 2:
                aerobic_idx = 2
                if aerobic_idx < num_reactors:
                    aer_C = C_flat[aerobic_idx * NUM_COMPONENTS:(aerobic_idx + 1) * NUM_COMPONENTS]
                    Q_ir = reactor.operation.internal_return_ratio * Q
                    internal_return_to_use = (Q_ir, aer_C)
        
        if isinstance(reactor, CSTRReactor):
            dCdt = reactor.calculate_derivatives(
                reactor_in, Q, params,
                return_sludge=return_sludge_to_use,
                internal_return=internal_return_to_use,
            )
            dCdt[:NUM_COMPONENTS] = 0
            actual_dCdt = reactor.calculate_derivatives(
                reactor_in, Q, params,
                return_sludge=return_sludge_to_use,
                internal_return=internal_return_to_use,
            )
            residual[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = actual_dCdt
        else:
            if isinstance(reactor, SecondaryClarifier):
                effluent, rs_C, _ = reactor.process(reactor_in, Q, params)
                residual[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = C_reactor - reactor_in
            elif isinstance(reactor, MembraneUnit):
                effluent, _ = reactor.process(reactor_in, Q, params)
                residual[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = C_reactor - reactor_in
            elif isinstance(reactor, SimpleTreatment):
                effluent = reactor.process(reactor_in, Q, params)
                residual[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = C_reactor - reactor_in
    
    return residual


def _get_initial_guess(pfs: ProcessFlowSheet, C_in: np.ndarray, Q: float, 
                        params: ASM1Parameters) -> np.ndarray:
    """
    生成更好的初始猜测：逐级计算，前一单元的出水作为后一单元的进水
    """
    num_reactors = len(pfs.reactors)
    n = num_reactors * NUM_COMPONENTS
    C_guess = np.zeros(n)
    
    C_prev = C_in.copy()
    
    for i, reactor in enumerate(pfs.reactors):
        if isinstance(reactor, (SecondaryClarifier, MembraneUnit, SimpleTreatment)):
            if hasattr(reactor, 'process'):
                try:
                    if isinstance(reactor, SecondaryClarifier):
                        effluent, _, _ = reactor.process(C_prev, Q, params)
                    elif isinstance(reactor, MembraneUnit):
                        effluent, _ = reactor.process(C_prev, Q, params)
                    else:
                        effluent = reactor.process(C_prev, Q, params)
                    C_guess[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = effluent
                    C_prev = effluent.copy()
                    continue
                except:
                    pass
        
        C_guess[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = C_prev * 0.6
        
        if reactor.reactor_type == ReactorType.AEROBIC:
            SNH_idx = COMPONENT_INDEX['SNH']
            SNO_idx = COMPONENT_INDEX['SNO']
            SO_idx = COMPONENT_INDEX['SO']
            C_guess[i * NUM_COMPONENTS + SNH_idx] = max(C_prev[SNH_idx] * 0.2, 1.0)
            C_guess[i * NUM_COMPONENTS + SNO_idx] = max(C_prev[SNH_idx] * 0.6, 5.0)
            C_guess[i * NUM_COMPONENTS + SO_idx] = reactor.operation.DO_setpoint if hasattr(reactor.operation, 'DO_setpoint') else 2.0
        
        if reactor.is_anoxic():
            SNO_idx = COMPONENT_INDEX['SNO']
            C_guess[i * NUM_COMPONENTS + SNO_idx] = max(C_prev[SNO_idx] * 0.3, 0.5)
        
        C_guess[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = np.maximum(
            C_guess[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS], 0.1
        )
        
        C_prev = C_guess[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS].copy()
    
    return C_guess


def _solve_steady_by_newton(pfs: ProcessFlowSheet, C_in: np.ndarray, Q: float,
                            params: ASM1Parameters, C_guess: np.ndarray,
                            config: SolverConfig,
                            progress_callback: Optional[Callable[[int, float], None]] = None) -> Tuple[bool, np.ndarray, List[float]]:
    """
    使用Newton-Raphson方法求解稳态（内部函数）
    """
    num_reactors = len(pfs.reactors)
    n = num_reactors * NUM_COMPONENTS
    
    residual_history = []
    converged = False
    C_current = C_guess.copy()
    best_residual = np.inf
    best_C = C_current.copy()
    current_relaxation = config.relaxation
    
    for iteration in range(config.max_iterations):
        residual = calculate_system_residual(C_current, pfs, C_in, Q, params)
        residual_norm = np.linalg.norm(residual) / (n + 1e-10)
        residual_history.append(residual_norm)
        
        if residual_norm < best_residual:
            best_residual = residual_norm
            best_C = C_current.copy()
        
        if progress_callback is not None:
            progress_callback(iteration + 1, residual_norm)
        
        if residual_norm < config.tolerance:
            converged = True
            break
        
        if iteration > 5 and residual_norm > residual_history[-2] * 1.5:
            current_relaxation = max(current_relaxation * 0.5, config.min_relaxation)
        
        if iteration > 10 and len(residual_history) > 5 and residual_history[-5] < residual_history[-1]:
            current_relaxation = max(current_relaxation * 0.8, config.min_relaxation)
        
        J = np.zeros((n, n))
        eps = 1e-6
        for j in range(n):
            pert = eps * max(C_current[j], 1e-10)
            C_perturbed = C_current.copy()
            C_perturbed[j] += pert
            residual_perturbed = calculate_system_residual(C_perturbed, pfs, C_in, Q, params)
            J[:, j] = (residual_perturbed - residual) / pert
        
        try:
            J_reg = J + 1e-4 * np.eye(n)
            delta = np.linalg.solve(J_reg, -residual)
        except np.linalg.LinAlgError:
            J_reg = J + 1e-2 * np.eye(n)
            delta = np.linalg.solve(J_reg, -residual)
        
        max_delta = 100.0
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_delta:
            delta = delta * (max_delta / delta_norm)
        
        if config.use_damped_newton:
            C_candidate = C_current + current_relaxation * delta
            C_candidate = np.maximum(C_candidate, 0.0)
            
            r_candidate = calculate_system_residual(C_candidate, pfs, C_in, Q, params)
            rn_candidate = np.linalg.norm(r_candidate) / (n + 1e-10)
            
            inner_iter = 0
            while rn_candidate > residual_norm and inner_iter < 5:
                current_relaxation *= 0.5
                C_candidate = C_current + current_relaxation * delta
                C_candidate = np.maximum(C_candidate, 0.0)
                r_candidate = calculate_system_residual(C_candidate, pfs, C_in, Q, params)
                rn_candidate = np.linalg.norm(r_candidate) / (n + 1e-10)
                inner_iter += 1
            
            C_current = C_candidate
            current_relaxation = min(current_relaxation * 1.2, config.relaxation)
        else:
            C_new = C_current + current_relaxation * delta
            C_new = np.maximum(C_new, 0.0)
            C_current = C_new
    
    if not converged and best_residual < residual_history[-1]:
        C_current = best_C
        residual_history[-1] = best_residual
    
    return converged, C_current, residual_history


def _check_dynamic_steady(reactor_states: List[np.ndarray], check_idx: int, 
                          rtol: float = 1e-3) -> bool:
    """
    检查动态仿真是否达到稳态：比较最后两段时间的平均浓度变化
    """
    if check_idx < 2:
        return False
    
    n = len(reactor_states)
    half = check_idx // 2
    
    for i in range(n):
        last_half = reactor_states[i][half:check_idx, :]
        first_half = reactor_states[i][:half, :]
        
        mean_last = np.mean(last_half, axis=0)
        mean_first = np.mean(first_half, axis=0)
        
        diff = np.abs(mean_last - mean_first)
        denom = np.maximum(np.abs(mean_first), 1e-6)
        rel_diff = diff / denom
        
        if np.any(rel_diff > rtol):
            return False
    
    return True


def _solve_steady_by_dynamic(pfs: ProcessFlowSheet, influent: InfluentConfig,
                              params: ASM1Parameters, C_guess: np.ndarray,
                              config: SolverConfig) -> Tuple[bool, np.ndarray, List[float], int]:
    """
    使用动态仿真逼近稳态（内部函数）
    """
    num_reactors = len(pfs.reactors)
    n = num_reactors * NUM_COMPONENTS
    Q = influent.get_Q(0)
    C_in = influent.get_C(0)
    
    residual_history = []
    converged = False
    iterations = 0
    
    initial_states = [C_guess[i*NUM_COMPONENTS:(i+1)*NUM_COMPONENTS] for i in range(num_reactors)]
    
    total_days = 0.0
    segment_days = config.check_steady_every_days
    
    C_current = C_guess.copy()
    
    while total_days < config.dynamic_steady_days:
        dyn_config = SolverConfig(
            simulation_days=segment_days,
            output_interval_days=segment_days / 20,
            rtol=1e-4,
            atol=1e-6,
            method=config.method
        )
        
        dyn_result = run_dynamic_simulation(
            pfs, influent, params,
            initial_states=initial_states,
            config=dyn_config
        )
        
        if not dyn_result.success:
            break
        
        iterations += 1
        total_days += segment_days
        
        final_states = [dyn_result.reactor_states[i][-1] for i in range(num_reactors)]
        C_current = np.concatenate(final_states)
        
        residual = calculate_system_residual(C_current, pfs, C_in, Q, params)
        residual_norm = np.linalg.norm(residual) / (n + 1e-10)
        residual_history.append(residual_norm)
        
        if residual_norm < config.tolerance:
            converged = True
            break
        
        if _check_dynamic_steady(dyn_result.reactor_states, len(dyn_result.time_days), config.steady_state_rtol):
            converged = True
            break
        
        initial_states = final_states
    
    return converged, C_current, residual_history, iterations


def solve_steady_state(pfs: ProcessFlowSheet, influent: InfluentConfig,
                       params: ASM1Parameters,
                       config: Optional[SolverConfig] = None,
                       progress_callback: Optional[Callable[[int, float], None]] = None) -> SteadyStateResult:
    """
    求解稳态，支持多种方法：
    - 'newton': 仅Newton-Raphson迭代
    - 'dynamic': 动态仿真逼近稳态（最稳健）
    - 'hybrid': 先动态预热，再Newton迭代（默认）
    
    参数:
        pfs: 工艺流程
        influent: 进水配置
        params: ASM1参数
        config: 求解器配置
        progress_callback: 进度回调(迭代次数, 残差)
    
    返回:
        SteadyStateResult
    """
    if config is None:
        config = SolverConfig()
    
    Q = influent.get_Q(0)
    C_in = influent.get_C(0)
    
    num_reactors = len(pfs.reactors)
    
    C_guess = _get_initial_guess(pfs, C_in, Q, params)
    method = getattr(config, 'steady_state_method', 'hybrid')
    iterations = 0
    residual_history = []
    converged = False
    C_current = C_guess.copy()
    method_used = method
    
    if method in ['dynamic', 'hybrid']:
        print(f"   使用动态仿真方法...")
        
        if method == 'hybrid' and getattr(config, 'warmup_days', 0) > 0:
            print(f"   预热动态仿真 {config.warmup_days} 天...")
            warmup_config = SolverConfig(
                simulation_days=config.warmup_days,
                output_interval_days=config.warmup_days,
                rtol=1e-3,
                atol=1e-5
            )
            warmup_result = run_dynamic_simulation(
                pfs, influent, params,
                initial_states=[C_guess[i*NUM_COMPONENTS:(i+1)*NUM_COMPONENTS] for i in range(num_reactors)],
                config=warmup_config
            )
            if warmup_result.success:
                final_states = [warmup_result.reactor_states[i][-1] for i in range(num_reactors)]
                C_guess = np.concatenate(final_states)
                print(f"   预热完成")
        
        if method == 'dynamic':
            converged, C_current, residual_history, iterations = _solve_steady_by_dynamic(
                pfs, influent, params, C_guess, config
            )
        else:  # hybrid
            converged, C_current, residual_history, iterations = _solve_steady_by_dynamic(
                pfs, influent, params, C_guess, config
            )
            
            if not converged:
                print(f"   动态仿真未完全收敛，尝试Newton-Raphson迭代...")
                n_conv, C_new, res_hist = _solve_steady_by_newton(
                    pfs, C_in, Q, params, C_current, config, progress_callback
                )
                iterations += len(res_hist)
                residual_history.extend(res_hist)
                if n_conv or res_hist[-1] < residual_history[-2] if len(residual_history) > 1 else True:
                    converged = n_conv
                    C_current = C_new
                
                if not converged:
                    converged = residual_history[-1] < config.tolerance * 10
    
    else:  # newton only
        if getattr(config, 'warmup_days', 0) > 0:
            print(f"   预热动态仿真 {config.warmup_days} 天...")
            warmup_config = SolverConfig(
                simulation_days=config.warmup_days,
                output_interval_days=config.warmup_days,
                rtol=1e-3,
                atol=1e-5
            )
            warmup_result = run_dynamic_simulation(
                pfs, influent, params,
                initial_states=[C_guess[i*NUM_COMPONENTS:(i+1)*NUM_COMPONENTS] for i in range(num_reactors)],
                config=warmup_config
            )
            if warmup_result.success:
                final_states = [warmup_result.reactor_states[i][-1] for i in range(num_reactors)]
                C_guess = np.concatenate(final_states)
                print(f"   预热完成")
        
        converged, C_current, residual_history = _solve_steady_by_newton(
            pfs, C_in, Q, params, C_guess, config, progress_callback
        )
        iterations = len(residual_history)
    
    reactor_states = []
    for i in range(num_reactors):
        state = C_current[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS]
        state = np.maximum(state, 0.0)
        reactor_states.append(state)
        pfs.reactors[i].C = state.copy()
        pfs.reactors[i].C_out = state.copy()
    
    effluent_state = reactor_states[-1]
    effluent_quality = aggregate_to_wq_indices(effluent_state)
    
    avg_residual = residual_history[-1]
    
    if converged:
        message = f"收敛！迭代{iterations}次，残差范数: {residual_history[-1]:.2e} (方法: {method_used})"
    elif config.use_engineering_tolerance and avg_residual < config.tolerance * config.engineering_tolerance_factor:
        converged = True
        message = f"工程收敛（容差×{config.engineering_tolerance_factor:.0f}）！迭代{iterations}次，残差: {residual_history[-1]:.2e}，阈值: {config.tolerance * config.engineering_tolerance_factor:.2e}"
    else:
        message = f"未收敛，迭代{iterations}次，残差: {residual_history[-1]:.2e}，阈值: {config.tolerance:.2e}。结果可作为近似稳态使用。"
    
    return SteadyStateResult(
        converged=converged,
        iterations=iterations,
        final_residual=residual_history[-1] if residual_history else np.inf,
        residual_history=residual_history,
        reactor_states=reactor_states,
        effluent_quality=effluent_quality,
        message=message,
    )


class DynamicSimulator:
    """动态仿真器"""
    
    def __init__(self, pfs: ProcessFlowSheet, influent: InfluentConfig,
                 params: ASM1Parameters, config: Optional[SolverConfig] = None):
        self.pfs = pfs
        self.influent = influent
        self.params = params
        self.config = config or SolverConfig()
        self._pause_event = False
        self._stop_event = False
    
    def pause(self):
        self._pause_event = True
    
    def resume(self):
        self._pause_event = False
    
    def stop(self):
        self._stop_event = True
    
    def _ode_function(self, t_days: float, C_flat: np.ndarray) -> np.ndarray:
        """
        ODE右侧函数: dC/dt = f(t, C)
        
        参数:
            t_days: 时间(天)
            C_flat: 展平的浓度 [num_reactors * 13]
        
        返回:
            dCdt_flat: 导数
        """
        if self._stop_event:
            return np.zeros_like(C_flat)
        
        t_hours = t_days * 24
        Q = self.influent.get_Q(t_hours)
        C_in = self.influent.get_C(t_hours)
        
        num_reactors = len(self.pfs.reactors)
        dCdt_flat = np.zeros_like(C_flat)
        
        return_sludge = None
        internal_return = None
        
        secondary_idx = None
        for i, reactor in enumerate(self.pfs.reactors):
            if isinstance(reactor, SecondaryClarifier):
                secondary_idx = i
                break
        
        membrane_idx = None
        for i, reactor in enumerate(self.pfs.reactors):
            if isinstance(reactor, MembraneUnit):
                membrane_idx = i
                break
        
        for i, reactor in enumerate(self.pfs.reactors):
            C_reactor = C_flat[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS]
            
            if i == 0:
                reactor_in = C_in
            else:
                prev_C = C_flat[(i - 1) * NUM_COMPONENTS:i * NUM_COMPONENTS]
                
                if isinstance(self.pfs.reactors[i - 1], SecondaryClarifier):
                    prev_reactor = self.pfs.reactors[i - 1]
                    effluent, rs_C, _ = prev_reactor.process(prev_C, Q, self.params)
                    reactor_in = effluent
                elif isinstance(self.pfs.reactors[i - 1], MembraneUnit):
                    prev_reactor = self.pfs.reactors[i - 1]
                    effluent, _ = prev_reactor.process(prev_C, Q, self.params)
                    reactor_in = effluent
                elif isinstance(self.pfs.reactors[i - 1], SimpleTreatment):
                    prev_reactor = self.pfs.reactors[i - 1]
                    reactor_in = prev_reactor.process(prev_C, Q, self.params)
                else:
                    reactor_in = prev_C
            
            return_sludge_to_use = None
            if secondary_idx is not None and i == 0:
                sec_C = C_flat[secondary_idx * NUM_COMPONENTS:(secondary_idx + 1) * NUM_COMPONENTS]
                sec_reactor = self.pfs.reactors[secondary_idx]
                effluent, rs_C, _ = sec_reactor.process(sec_C, Q, self.params)
                Q_r = sec_reactor.operation.return_sludge_ratio * Q
                return_sludge_to_use = (Q_r, rs_C)
            
            if membrane_idx is not None and i == 0:
                mem_C = C_flat[membrane_idx * NUM_COMPONENTS:(membrane_idx + 1) * NUM_COMPONENTS]
                mem_reactor = self.pfs.reactors[membrane_idx]
                _, retentate = mem_reactor.process(mem_C, Q, self.params)
                Q_r = mem_reactor.operation.return_sludge_ratio * Q
                return_sludge_to_use = (Q_r, retentate)
            
            internal_return_to_use = None
            if hasattr(reactor.operation, 'internal_return_ratio') and reactor.operation.internal_return_ratio > 0:
                if i == 1 and len(self.pfs.reactors) > 2:
                    aerobic_idx = 2
                    if aerobic_idx < num_reactors:
                        aer_C = C_flat[aerobic_idx * NUM_COMPONENTS:(aerobic_idx + 1) * NUM_COMPONENTS]
                        Q_ir = reactor.operation.internal_return_ratio * Q
                        internal_return_to_use = (Q_ir, aer_C)
            
            if isinstance(reactor, CSTRReactor):
                reactor.C = C_reactor.copy()
                dCdt = reactor.calculate_derivatives(
                    reactor_in, Q, self.params,
                    return_sludge=return_sludge_to_use,
                    internal_return=internal_return_to_use,
                )
                dCdt_flat[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = dCdt
            else:
                dCdt_flat[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = 0.0
        
        return dCdt_flat
    
    def run(self, initial_states: Optional[List[np.ndarray]] = None,
            progress_callback: Optional[Callable[[float], None]] = None) -> DynamicResult:
        """
        运行动态仿真
        
        参数:
            initial_states: 初始状态，每个反应器的浓度
            progress_callback: 进度回调(已仿真天数)
        
        返回:
            DynamicResult
        """
        self._pause_event = False
        self._stop_event = False
        
        num_reactors = len(self.pfs.reactors)
        n = num_reactors * NUM_COMPONENTS
        
        if initial_states is None:
            C0 = np.zeros(n)
            C_in = self.influent.get_C(0)
            for i in range(num_reactors):
                C0[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS] = C_in * 0.5
        else:
            C0 = np.concatenate(initial_states)
        
        t_span = (0, self.config.simulation_days)
        t_eval = np.arange(0, self.config.simulation_days, self.config.output_interval_days)
        
        success = False
        message = ""
        
        try:
            sol = solve_ivp(
                self._ode_function,
                t_span,
                C0,
                method=self.config.method,
                t_eval=t_eval,
                rtol=self.config.rtol,
                atol=self.config.atol,
                dense_output=True,
            )
            success = sol.success
            if not success:
                message = sol.message
        except Exception as e:
            success = False
            message = str(e)
        
        if not success:
            return DynamicResult(
                success=False,
                time_days=np.array([]),
                reactor_states=[],
                effluent_quality_history=[],
                message=message,
            )
        
        time_days = sol.t
        states = sol.y
        
        reactor_states = []
        for i in range(num_reactors):
            reactor_states.append(
                states[i * NUM_COMPONENTS:(i + 1) * NUM_COMPONENTS, :].T
            )
        
        effluent_history = []
        for t_idx in range(len(time_days)):
            effluent_state = reactor_states[-1][t_idx, :]
            effluent_history.append(aggregate_to_wq_indices(effluent_state))
        
        return DynamicResult(
            success=True,
            time_days=time_days,
            reactor_states=reactor_states,
            effluent_quality_history=effluent_history,
            message="仿真完成",
        )


def run_dynamic_simulation(pfs: ProcessFlowSheet, influent: InfluentConfig,
                           params: ASM1Parameters,
                           initial_states: Optional[List[np.ndarray]] = None,
                           config: Optional[SolverConfig] = None,
                           progress_callback: Optional[Callable[[float], None]] = None) -> DynamicResult:
    """
    便捷函数：运行动态仿真
    """
    simulator = DynamicSimulator(pfs, influent, params, config)
    return simulator.run(initial_states, progress_callback)
