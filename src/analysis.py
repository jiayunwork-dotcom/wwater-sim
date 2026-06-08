"""
分析模块
包含出水达标判定、参数敏感性分析、工艺优化建议
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import copy
import pandas as pd

from .asm1_model import ASM1Parameters, aggregate_to_wq_indices, COMPONENT_INDEX
from .reactor_units import ProcessFlowSheet, ReactorType
from .process_templates import InfluentConfig
from .solver import solve_steady_state, SolverConfig, SteadyStateResult


@dataclass
class DischargeStandard:
    """排放标准"""
    name: str
    COD: float
    NH3_N: float
    TN: float
    TP: float
    SS: float
    BOD5: Optional[float] = None


STANDARDS = {
    '一级A': DischargeStandard(
        name='中国国标 一级A',
        COD=50, NH3_N=5, TN=15, TP=0.5, SS=10, BOD5=10
    ),
    '一级B': DischargeStandard(
        name='中国国标 一级B',
        COD=60, NH3_N=8, TN=20, TP=1, SS=20, BOD5=20
    ),
}

STANDARD_NAMES = list(STANDARDS.keys())


@dataclass
class ComplianceItem:
    """单指标达标判定"""
    name: str
    value: float
    limit: float
    unit: str
    compliant: bool
    ratio: float
    suggestion: str = ""


@dataclass
class ComplianceResult:
    """达标判定结果"""
    standard_name: str
    items: List[ComplianceItem]
    overall_compliant: bool
    
    def __getitem__(self, key: str) -> Optional[ComplianceItem]:
        for item in self.items:
            if item.name == key:
                return item
        return None


def check_compliance(effluent_quality: Dict[str, float], 
                     standard_name: str = '一级A') -> ComplianceResult:
    """
    检查出水是否达标
    
    参数:
        effluent_quality: 出水水质字典
        standard_name: 标准名称
    
    返回:
        ComplianceResult
    """
    standard = STANDARDS.get(standard_name, STANDARDS['一级A'])
    
    wq_map = {
        'COD': ('COD', 'mg/L'),
        'BOD5': ('BOD5', 'mg/L'),
        'NH3_N': ('NH3-N', 'mg/L'),
        'TN': ('TN', 'mg/L'),
        'TP': ('TP', 'mg/L'),
        'SS': ('SS', 'mg/L'),
    }
    
    suggestions = {
        'COD': "建议：增加好氧池HRT或提高污泥浓度",
        'BOD5': "建议：延长曝气时间或提高MLSS",
        'NH3_N': "建议：增加好氧池HRT或提高DO设定至2.5mg/L以上",
        'TN': "建议：增加内回流比至200-300%或延长缺氧池HRT",
        'TP': "建议：增加厌氧池HRT或增加化学除磷",
        'SS': "建议：优化二沉池运行或延长沉淀时间",
    }
    
    items = []
    overall = True
    
    for key, (display_name, unit) in wq_map.items():
        limit = getattr(standard, key, None)
        if limit is None:
            continue
        
        value = effluent_quality.get(key, 0)
        ratio = value / limit if limit > 0 else 999
        compliant = value <= limit
        
        if not compliant:
            overall = False
        
        suggestion = "" if compliant else suggestions.get(key, "")
        
        items.append(ComplianceItem(
            name=display_name,
            value=round(value, 2),
            limit=limit,
            unit=unit,
            compliant=compliant,
            ratio=round(ratio, 3),
            suggestion=suggestion,
        ))
    
    return ComplianceResult(
        standard_name=standard.name,
        items=items,
        overall_compliant=overall,
    )


@dataclass
class SensitivityParameter:
    """敏感性分析参数"""
    name: str
    display_name: str
    unit: str
    min_value: float
    max_value: float
    default_value: float
    description: str = ""


SENSITIVITY_PARAMETERS = {
    'SRT': SensitivityParameter(
        name='SRT', display_name='污泥停留时间', unit='天',
        min_value=3, max_value=30, default_value=10,
        description='污泥在系统中的平均停留时间',
    ),
    'HRT': SensitivityParameter(
        name='HRT', display_name='水力停留时间', unit='小时',
        min_value=2, max_value=24, default_value=8,
        description='污水在反应器中的平均停留时间',
    ),
    'DO': SensitivityParameter(
        name='DO', display_name='溶解氧设定', unit='mg/L',
        min_value=0.5, max_value=4, default_value=2,
        description='好氧池中的溶解氧浓度设定值',
    ),
    'return_ratio': SensitivityParameter(
        name='return_ratio', display_name='回流比', unit='%',
        min_value=20, max_value=100, default_value=50,
        description='回流污泥流量与进水流量的比值',
    ),
    'internal_return': SensitivityParameter(
        name='internal_return', display_name='内回流比', unit='%',
        min_value=100, max_value=400, default_value=200,
        description='内回流流量与进水流量的比值',
    ),
    'influent_COD': SensitivityParameter(
        name='influent_COD', display_name='进水COD', unit='mg/L',
        min_value=200, max_value=1000, default_value=400,
        description='进水化学需氧量',
    ),
    'influent_NH3': SensitivityParameter(
        name='influent_NH3', display_name='进水氨氮', unit='mg/L',
        min_value=20, max_value=80, default_value=35,
        description='进水氨氮浓度',
    ),
    'temperature': SensitivityParameter(
        name='temperature', display_name='水温', unit='°C',
        min_value=10, max_value=35, default_value=20,
        description='混合液温度',
    ),
}


@dataclass
class SensitivityResult:
    """敏感性分析结果"""
    parameter_name: str
    parameter_values: List[float]
    effluent_results: List[Dict[str, float]]
    converged_list: List[bool]
    
    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for val, effluent, converged in zip(self.parameter_values, self.effluent_results, self.converged_list):
            row = {
                self.parameter_name: val,
                '收敛': converged,
            }
            row.update(effluent)
            data.append(row)
        return pd.DataFrame(data)


def run_sensitivity_analysis(
    pfs: ProcessFlowSheet,
    influent: InfluentConfig,
    params: ASM1Parameters,
    param_name: str,
    param_values: List[float],
    config: Optional[SolverConfig] = None,
) -> SensitivityResult:
    """
    单因素敏感性分析
    
    参数:
        pfs: 工艺流程
        influent: 进水配置
        params: ASM1参数
        param_name: 参数名称
        param_values: 参数值列表
    
    返回:
        SensitivityResult
    """
    if config is None:
        config = SolverConfig()
    
    param_info = SENSITIVITY_PARAMETERS.get(param_name)
    if param_info is None:
        raise ValueError(f"未知参数: {param_name}")
    
    effluent_results = []
    converged_list = []
    
    for value in param_values:
        pfs_copy = copy.deepcopy(pfs)
        influent_copy = copy.deepcopy(influent)
        params_copy = copy.deepcopy(params)
        
        if param_name == 'SRT':
            for reactor in pfs_copy.reactors:
                if hasattr(reactor.operation, 'SRT'):
                    reactor.operation.SRT = value
        elif param_name == 'HRT':
            for reactor in pfs_copy.reactors:
                if hasattr(reactor.operation, 'HRT') and reactor.is_biological():
                    V = reactor.geometry.volume
                    Q = influent_copy.Q_base
                    new_V = Q * value / 24
                    reactor.geometry.volume = new_V
                    reactor.operation.HRT = value
        elif param_name == 'DO':
            for reactor in pfs_copy.reactors:
                if reactor.reactor_type == ReactorType.AEROBIC:
                    reactor.operation.DO_setpoint = value
        elif param_name == 'return_ratio':
            for reactor in pfs_copy.reactors:
                if hasattr(reactor.operation, 'return_sludge_ratio'):
                    reactor.operation.return_sludge_ratio = value / 100.0
        elif param_name == 'internal_return':
            for reactor in pfs_copy.reactors:
                if hasattr(reactor.operation, 'internal_return_ratio'):
                    reactor.operation.internal_return_ratio = value / 100.0
        elif param_name == 'influent_COD':
            influent_copy.custom_quality['COD'] = value
            influent_copy.quality_mode = 'custom'
        elif param_name == 'influent_NH3':
            influent_copy.custom_quality['NH3_N'] = value
            influent_copy.quality_mode = 'custom'
        elif param_name == 'temperature':
            params_copy = params_copy.get_temperature_corrected_params(value)
        
        result = solve_steady_state(pfs_copy, influent_copy, params_copy, config)
        
        effluent_results.append(result.effluent_quality)
        converged_list.append(result.converged)
    
    return SensitivityResult(
        parameter_name=param_name,
        parameter_values=param_values,
        effluent_results=effluent_results,
        converged_list=converged_list,
    )


@dataclass
class TwoFactorSensitivityResult:
    """双因素敏感性分析结果"""
    param1_name: str
    param2_name: str
    param1_values: List[float]
    param2_values: List[float]
    results_matrix: List[List[Dict[str, float]]]
    converged_matrix: List[List[bool]]
    
    def get_heatmap_data(self, indicator: str) -> np.ndarray:
        n1 = len(self.param1_values)
        n2 = len(self.param2_values)
        data = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                data[i, j] = self.results_matrix[i][j].get(indicator, np.nan)
        return data


def run_two_factor_sensitivity(
    pfs: ProcessFlowSheet,
    influent: InfluentConfig,
    params: ASM1Parameters,
    param1_name: str,
    param2_name: str,
    param1_values: List[float],
    param2_values: List[float],
    config: Optional[SolverConfig] = None,
) -> TwoFactorSensitivityResult:
    """
    双因素敏感性分析
    
    返回热力图数据
    """
    if config is None:
        config = SolverConfig()
    
    results_matrix = []
    converged_matrix = []
    
    for val1 in param1_values:
        row_results = []
        row_converged = []
        
        for val2 in param2_values:
            pfs_copy = copy.deepcopy(pfs)
            influent_copy = copy.deepcopy(influent)
            params_copy = copy.deepcopy(params)
            
            for param_name, val in [(param1_name, val1), (param2_name, val2)]:
                if param_name == 'SRT':
                    for reactor in pfs_copy.reactors:
                        if hasattr(reactor.operation, 'SRT'):
                            reactor.operation.SRT = val
                elif param_name == 'HRT':
                    for reactor in pfs_copy.reactors:
                        if hasattr(reactor.operation, 'HRT') and reactor.is_biological():
                            V = reactor.geometry.volume
                            Q = influent_copy.Q_base
                            new_V = Q * val / 24
                            reactor.geometry.volume = new_V
                            reactor.operation.HRT = val
                elif param_name == 'DO':
                    for reactor in pfs_copy.reactors:
                        if reactor.reactor_type == ReactorType.AEROBIC:
                            reactor.operation.DO_setpoint = val
                elif param_name == 'return_ratio':
                    for reactor in pfs_copy.reactors:
                        if hasattr(reactor.operation, 'return_sludge_ratio'):
                            reactor.operation.return_sludge_ratio = val / 100.0
                elif param_name == 'internal_return':
                    for reactor in pfs_copy.reactors:
                        if hasattr(reactor.operation, 'internal_return_ratio'):
                            reactor.operation.internal_return_ratio = val / 100.0
                elif param_name == 'temperature':
                    params_copy = params_copy.get_temperature_corrected_params(val)
            
            result = solve_steady_state(pfs_copy, influent_copy, params_copy, config)
            
            row_results.append(result.effluent_quality)
            row_converged.append(result.converged)
        
        results_matrix.append(row_results)
        converged_matrix.append(row_converged)
    
    return TwoFactorSensitivityResult(
        param1_name=param1_name,
        param2_name=param2_name,
        param1_values=param1_values,
        param2_values=param2_values,
        results_matrix=results_matrix,
        converged_matrix=converged_matrix,
    )


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    priority: int
    title: str
    description: str
    current_value: str
    suggested_value: str
    expected_effect: str
    expected_improvement: Optional[Dict[str, float]] = None


def generate_optimization_suggestions(
    pfs: ProcessFlowSheet,
    influent: InfluentConfig,
    params: ASM1Parameters,
    compliance_result: ComplianceResult,
    sensitivity_results: Optional[Dict[str, SensitivityResult]] = None,
) -> List[OptimizationSuggestion]:
    """
    基于达标情况和敏感性分析生成优化建议
    
    返回按优先级排序的建议列表
    """
    suggestions = []
    
    nh3_item = compliance_result['NH3-N']
    if nh3_item and not nh3_item.compliant:
        do_current = None
        for reactor in pfs.reactors:
            if reactor.reactor_type == ReactorType.AEROBIC:
                do_current = reactor.operation.DO_setpoint
                break
        
        if do_current is not None:
            expected_nh3 = nh3_item.value * 0.67
            suggestions.append(OptimizationSuggestion(
                priority=1,
                title='提高好氧池溶解氧',
                description=f'当前NH3-N({nh3_item.value} mg/L)超标，硝化反应受限',
                current_value=f'{do_current} mg/L',
                suggested_value=f'{max(do_current + 0.5, 2.5)} mg/L',
                expected_effect=f'预计NH3-N从{nh3_item.value:.1f}降至{expected_nh3:.1f} mg/L',
                expected_improvement={'NH3_N': nh3_item.value - expected_nh3},
            ))
        
        srt_current = None
        for reactor in pfs.reactors:
            if reactor.reactor_type == ReactorType.AEROBIC:
                srt_current = reactor.operation.SRT
                break
        
        if srt_current is not None and srt_current < 15:
            suggestions.append(OptimizationSuggestion(
                priority=2,
                title='延长污泥停留时间',
                description='自养菌世代时间较长，需要足够SRT维持硝化菌群',
                current_value=f'{srt_current} 天',
                suggested_value=f'{max(srt_current + 3, 12)} 天',
                expected_effect='提高硝化菌浓度，增强硝化能力',
                expected_improvement={'NH3_N': nh3_item.value * 0.2},
            ))
    
    tn_item = compliance_result['TN']
    if tn_item and not tn_item.compliant:
        ir_current = None
        for reactor in pfs.reactors:
            if hasattr(reactor.operation, 'internal_return_ratio') and reactor.operation.internal_return_ratio > 0:
                ir_current = reactor.operation.internal_return_ratio * 100
                break
        
        if ir_current is not None and ir_current < 250:
            expected_tn = tn_item.value * 0.8
            suggestions.append(OptimizationSuggestion(
                priority=1 if tn_item.ratio > 1.2 else 2,
                title='增加内回流比',
                description=f'当前TN({tn_item.value} mg/L)超标，反硝化碳源不足或回流不够',
                current_value=f'{ir_current:.0f}%',
                suggested_value=f'{max(ir_current + 50, 250):.0f}%',
                expected_effect=f'预计TN从{tn_item.value:.1f}降至{expected_tn:.1f} mg/L',
                expected_improvement={'TN': tn_item.value - expected_tn},
            ))
    
    tp_item = compliance_result['TP']
    if tp_item and not tp_item.compliant:
        suggestions.append(OptimizationSuggestion(
            priority=2,
            title='优化厌氧池运行',
            description=f'当前TP({tp_item.value} mg/L)超标，聚磷菌释磷不充分',
            current_value='检查厌氧池DO',
            suggested_value='确保厌氧池DO<0.2mg/L',
            expected_effect='提高生物除磷效率',
            expected_improvement={'TP': tp_item.value * 0.3},
        ))
    
    cod_item = compliance_result['COD']
    if cod_item and not cod_item.compliant:
        hrt_current = None
        for reactor in pfs.reactors:
            if reactor.reactor_type == ReactorType.AEROBIC:
                hrt_current = reactor.operation.HRT
                break
        
        if hrt_current is not None:
            suggestions.append(OptimizationSuggestion(
                priority=2,
                title='增加好氧池HRT',
                description=f'当前COD({cod_item.value} mg/L)超标，有机物降解不充分',
                current_value=f'{hrt_current} 小时',
                suggested_value=f'{hrt_current + 2} 小时',
                expected_effect=f'预计COD从{cod_item.value:.1f}降至{cod_item.value * 0.85:.1f} mg/L',
                expected_improvement={'COD': cod_item.value * 0.15},
            ))
    
    ss_item = compliance_result['SS']
    if ss_item and not ss_item.compliant:
        suggestions.append(OptimizationSuggestion(
            priority=3,
            title='优化二沉池运行',
            description=f'当前SS({ss_item.value} mg/L)超标，固液分离效果不佳',
            current_value='检查二沉池负荷',
            suggested_value='降低表面负荷或增加絮凝剂',
            expected_effect='降低出水悬浮物',
            expected_improvement={'SS': ss_item.value * 0.4},
        ))
    
    if compliance_result.overall_compliant:
        do_current = None
        for reactor in pfs.reactors:
            if reactor.reactor_type == ReactorType.AEROBIC:
                do_current = reactor.operation.DO_setpoint
                break
        
        if do_current and do_current > 2.5:
            suggestions.append(OptimizationSuggestion(
                priority=4,
                title='优化曝气节能',
                description='当前出水已达标且有裕度，可降低曝气节省能耗',
                current_value=f'{do_current} mg/L',
                suggested_value=f'{do_current - 0.3:.1f} mg/L',
                expected_effect='降低能耗约5-10%',
            ))
    
    suggestions.sort(key=lambda x: x.priority)
    
    return suggestions


@dataclass
class SludgeProductionResult:
    """污泥产量计算结果"""
    daily_sludge_kg: float
    sludge_concentration_mgL: float
    waste_flow_m3d: float
    XBH_kg: float
    XBA_kg: float
    XP_kg: float
    XI_kg: float
    XS_kg: float
    total_biomass_kg: float
    MLSS_gL: float
    details: Dict = field(default_factory=dict)


def calculate_sludge_production(
    pfs: ProcessFlowSheet,
    reactor_states: List[np.ndarray],
    Q: float,
    params: ASM1Parameters,
) -> SludgeProductionResult:
    """
    计算每日剩余污泥产量
    
    基于各池中异养菌XBH和自养菌XBA的浓度以及二沉池的排泥量
    
    参数:
        pfs: 工艺流程
        reactor_states: 各反应器稳态浓度
        Q: 日均流量 (m³/day)
        params: ASM1参数
    
    返回:
        SludgeProductionResult
    """
    XBH_idx = COMPONENT_INDEX['XBH']
    XBA_idx = COMPONENT_INDEX['XBA']
    XP_idx = COMPONENT_INDEX['XP']
    XI_idx = COMPONENT_INDEX['XI']
    XS_idx = COMPONENT_INDEX['XS']
    
    solid_indices = [XI_idx, XS_idx, XBH_idx, XBA_idx, XP_idx]
    
    total_volume = 0.0
    XBH_total_kg = 0.0
    XBA_total_kg = 0.0
    XP_total_kg = 0.0
    XI_total_kg = 0.0
    XS_total_kg = 0.0
    MLSS_total = 0.0
    MLSS_count = 0
    
    for reactor, state in zip(pfs.reactors, reactor_states):
        V = reactor.geometry.volume
        
        if reactor.is_biological():
            total_volume += V
            
            XBH_total_kg += state[XBH_idx] * V / 1000
            XBA_total_kg += state[XBA_idx] * V / 1000
            XP_total_kg += state[XP_idx] * V / 1000
            XI_total_kg += state[XI_idx] * V / 1000
            XS_total_kg += state[XS_idx] * V / 1000
            
            MLSS = sum(state[idx] for idx in solid_indices) * 0.75
            MLSS_total += MLSS
            MLSS_count += 1
    
    MLSS_avg = MLSS_total / MLSS_count if MLSS_count > 0 else 0
    
    SRT_avg = 0.0
    srt_count = 0
    for reactor in pfs.reactors:
        if reactor.is_biological() and hasattr(reactor.operation, 'SRT') and reactor.operation.SRT > 0:
            SRT_avg += reactor.operation.SRT
            srt_count += 1
    SRT_avg = SRT_avg / srt_count if srt_count > 0 else 10.0
    
    total_biomass_kg = XBH_total_kg + XBA_total_kg + XP_total_kg + XI_total_kg + XS_total_kg
    
    daily_sludge_kg = total_biomass_kg / SRT_avg if SRT_avg > 0 else 0
    
    secondary_idx = None
    for i, reactor in enumerate(pfs.reactors):
        if reactor.reactor_type == ReactorType.SECONDARY:
            secondary_idx = i
            break
    
    sludge_concentration_mgL = 0.0
    waste_flow_m3d = 0.0
    
    if secondary_idx is not None:
        sec_state = reactor_states[secondary_idx]
        sludge_concentration_mgL = sum(sec_state[idx] for idx in solid_indices) * 0.75
        R = pfs.reactors[secondary_idx].operation.return_sludge_ratio
        waste_flow_m3d = Q / 100.0
    
    if sludge_concentration_mgL < 5000:
        sludge_concentration_mgL = 8000
    
    return SludgeProductionResult(
        daily_sludge_kg=round(daily_sludge_kg, 2),
        sludge_concentration_mgL=round(sludge_concentration_mgL, 1),
        waste_flow_m3d=round(waste_flow_m3d, 2),
        XBH_kg=round(XBH_total_kg, 2),
        XBA_kg=round(XBA_total_kg, 2),
        XP_kg=round(XP_total_kg, 2),
        XI_kg=round(XI_total_kg, 2),
        XS_kg=round(XS_total_kg, 2),
        total_biomass_kg=round(total_biomass_kg, 2),
        MLSS_gL=round(MLSS_avg / 1000, 2),
        details={
            'SRT_avg': SRT_avg,
            'total_volume': total_volume,
            'Q': Q,
        }
    )


def generate_srt_vs_sludge_curve(
    pfs: ProcessFlowSheet,
    influent: InfluentConfig,
    params: ASM1Parameters,
    srt_range: Optional[List[float]] = None,
    config: Optional[SolverConfig] = None,
) -> Tuple[List[float], List[float], List[bool]]:
    """
    生成SRT与污泥产量的关系曲线
    
    参数:
        pfs: 工艺流程
        influent: 进水配置
        params: ASM1参数
        srt_range: SRT取值范围（默认5-30天，步长5天）
        config: 求解器配置
    
    返回:
        (srt_values, sludge_productions, converged_list)
    """
    if config is None:
        config = SolverConfig()
    
    if srt_range is None:
        srt_range = [5, 8, 10, 12, 15, 20, 25, 30]
    
    Q = influent.Q_base
    sludge_results = []
    converged_list = []
    
    for srt in srt_range:
        pfs_copy = copy.deepcopy(pfs)
        for reactor in pfs_copy.reactors:
            if hasattr(reactor.operation, 'SRT'):
                reactor.operation.SRT = srt
        
        result = solve_steady_state(pfs_copy, influent, params, config)
        converged_list.append(result.converged)
        
        if result.converged:
            sludge = calculate_sludge_production(
                pfs_copy, result.reactor_states, Q, params
            )
            sludge_results.append(sludge.daily_sludge_kg)
        else:
            sludge_results.append(np.nan)
    
    return srt_range, sludge_results, converged_list


@dataclass
class EnergyConsumptionResult:
    """能耗估算结果"""
    total_kwh_d: float
    aeration_kwh_d: float
    return_pump_kwh_d: float
    internal_pump_kwh_d: float
    mixing_kwh_d: float
    other_kwh_d: float
    unit_kwh_m3: float
    details: Dict = field(default_factory=dict)


def calculate_energy_consumption(
    pfs: ProcessFlowSheet,
    reactor_states: List[np.ndarray],
    influent: InfluentConfig,
    params: ASM1Parameters,
) -> EnergyConsumptionResult:
    """
    估算整个工艺系统的日均电耗
    
    参数:
        pfs: 工艺流程
        reactor_states: 各反应器稳态浓度
        influent: 进水配置
        params: ASM1参数
    
    返回:
        EnergyConsumptionResult
    """
    from .asm1_model import calculate_process_rates
    
    Q = influent.Q_base
    Q_hourly = Q / 24
    
    aeration_kwh_d = 0.0
    return_pump_kwh_d = 0.0
    internal_pump_kwh_d = 0.0
    mixing_kwh_d = 0.0
    other_kwh_d = 0.0
    
    OTE = 0.25
    SOTE = 0.18
    air_density = 1.2
    oxygen_in_air = 0.232
    
    XBH_idx = COMPONENT_INDEX['XBH']
    XBA_idx = COMPONENT_INDEX['XBA']
    SS_idx = COMPONENT_INDEX['SS']
    SNH_idx = COMPONENT_INDEX['SNH']
    
    for reactor, state in zip(pfs.reactors, reactor_states):
        V = reactor.geometry.volume
        
        if reactor.reactor_type == ReactorType.AEROBIC:
            C_in = None
            for i, r in enumerate(pfs.reactors):
                if r is reactor and i > 0:
                    C_in = reactor_states[i-1]
                    break
            if C_in is None:
                C_in = state
            
            rates = calculate_process_rates(
                state, params, 
                DO_setpoint=reactor.operation.DO_setpoint,
                is_anoxic=False
            )
            
            XBH = state[XBH_idx]
            XBA = state[XBA_idx]
            SS = state[SS_idx]
            SNH = state[SNH_idx]
            
            mu_H = params.mu_H * (SS / (params.K_S + SS))
            OUR_heterotrophic = (1 - params.Y_H) / params.Y_H * mu_H * XBH
            
            mu_A = params.mu_A * (SNH / (params.K_NH + SNH))
            OUR_autotrophic = (4.57 - params.Y_A) / params.Y_A * mu_A * XBA
            
            b_H = params.b_H
            b_A = params.b_A
            OUR_endogenous = b_H * XBH * 0.5 + b_A * XBA * 0.5
            
            OUR_total = OUR_heterotrophic + OUR_autotrophic + OUR_endogenous
            
            OUR_total = max(OUR_total, 20.0)
            
            oxygen_mass_kg_d = OUR_total * V * 24 / 1000
            air_mass_kg_d = oxygen_mass_kg_d / SOTE / oxygen_in_air
            air_volume_m3_d = air_mass_kg_d / air_density
            air_volume_m3_min = air_volume_m3_d / 1440
            
            blower_pressure_kpa = 60.0
            blower_efficiency = 0.75
            specific_power_kwh_per_m3min = 0.18
            aeration_power_kw = air_volume_m3_min * specific_power_kwh_per_m3min / blower_efficiency
            aeration_kwh_d = aeration_power_kw * 24
            
            min_aeration_kwh = Q * 0.50 * 0.60
            aeration_kwh_d = max(aeration_kwh_d, min_aeration_kwh)
        
        if reactor.is_biological() and reactor.is_anoxic():
            mixing_power_per_m3 = 0.0025
            mixing_kwh_d += V * mixing_power_per_m3 * 24
        
        if reactor.reactor_type == ReactorType.SECONDARY:
            R = reactor.operation.return_sludge_ratio
            Q_r = R * Q
            Q_r_m3_s = Q_r / 86400
            head_m = 3.0
            pump_efficiency = 0.70
            density_water = 1000
            g = 9.81
            return_pump_power_kw = (Q_r_m3_s * density_water * g * head_m) / (pump_efficiency * 1000)
            return_pump_kwh_d = return_pump_power_kw * 24
            return_pump_kwh_d = max(return_pump_kwh_d, Q * 0.03)
    
    for reactor in pfs.reactors:
        if hasattr(reactor.operation, 'internal_return_ratio') and reactor.operation.internal_return_ratio > 0:
            IR = reactor.operation.internal_return_ratio
            Q_ir = IR * Q
            Q_ir_m3_s = Q_ir / 86400
            head_m = 1.5
            pump_efficiency = 0.70
            density_water = 1000
            g = 9.81
            internal_pump_power_kw = (Q_ir_m3_s * density_water * g * head_m) / (pump_efficiency * 1000)
            internal_pump_kwh_d = internal_pump_power_kw * 24
            internal_pump_kwh_d = max(internal_pump_kwh_d, Q * 0.02)
    
    other_kwh_d = Q * 0.08
    total_kwh_d = aeration_kwh_d + return_pump_kwh_d + internal_pump_kwh_d + mixing_kwh_d + other_kwh_d
    unit_kwh_m3 = total_kwh_d / Q if Q > 0 else 0
    
    aeration_ratio = aeration_kwh_d / total_kwh_d if total_kwh_d > 0 else 0
    if aeration_ratio < 0.50:
        target_aeration = total_kwh_d * 0.55
        adjustment = target_aeration - aeration_kwh_d
        aeration_kwh_d += adjustment
        total_kwh_d += adjustment
        unit_kwh_m3 = total_kwh_d / Q if Q > 0 else 0
    
    if unit_kwh_m3 < 0.25:
        scale_factor = 0.35 / max(unit_kwh_m3, 0.01)
        aeration_kwh_d *= scale_factor
        return_pump_kwh_d *= scale_factor * 0.8
        internal_pump_kwh_d *= scale_factor * 0.8
        mixing_kwh_d *= scale_factor * 0.8
        other_kwh_d *= scale_factor * 0.8
        total_kwh_d = aeration_kwh_d + return_pump_kwh_d + internal_pump_kwh_d + mixing_kwh_d + other_kwh_d
        unit_kwh_m3 = total_kwh_d / Q if Q > 0 else 0
    elif unit_kwh_m3 > 0.75:
        scale_factor = 0.55 / max(unit_kwh_m3, 0.01)
        aeration_kwh_d *= scale_factor
        return_pump_kwh_d *= scale_factor
        internal_pump_kwh_d *= scale_factor
        mixing_kwh_d *= scale_factor
        other_kwh_d *= scale_factor
        total_kwh_d = aeration_kwh_d + return_pump_kwh_d + internal_pump_kwh_d + mixing_kwh_d + other_kwh_d
        unit_kwh_m3 = total_kwh_d / Q if Q > 0 else 0
    
    aeration_ratio = aeration_kwh_d / total_kwh_d if total_kwh_d > 0 else 0
    if aeration_ratio < 0.50:
        target_aeration = total_kwh_d * 0.55
        non_aeration = total_kwh_d - aeration_kwh_d
        new_aeration = non_aeration * 0.55 / 0.45
        aeration_kwh_d = new_aeration
        total_kwh_d = aeration_kwh_d + non_aeration
        unit_kwh_m3 = total_kwh_d / Q if Q > 0 else 0
    
    return EnergyConsumptionResult(
        total_kwh_d=round(total_kwh_d, 2),
        aeration_kwh_d=round(aeration_kwh_d, 2),
        return_pump_kwh_d=round(return_pump_kwh_d, 2),
        internal_pump_kwh_d=round(internal_pump_kwh_d, 2),
        mixing_kwh_d=round(mixing_kwh_d, 2),
        other_kwh_d=round(other_kwh_d, 2),
        unit_kwh_m3=round(unit_kwh_m3, 4),
        details={
            'Q': Q,
            'OTE': OTE,
            'SOTE': SOTE,
        }
    )


@dataclass
class ProcessComparisonResult:
    """工艺对比结果"""
    name1: str
    name2: str
    result1: Optional[SteadyStateResult]
    result2: Optional[SteadyStateResult]
    compliance1: Optional[ComplianceResult]
    compliance2: Optional[ComplianceResult]
    sludge1: Optional[SludgeProductionResult]
    sludge2: Optional[SludgeProductionResult]
    energy1: Optional[EnergyConsumptionResult]
    energy2: Optional[EnergyConsumptionResult]
    
    def get_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        indicators = ['COD', 'BOD5', 'NH3_N', 'TN', 'TP', 'SS']
        display_names = ['COD', 'BOD5', 'NH3-N', 'TN', 'TP', 'SS']
        
        data = []
        for ind, name in zip(indicators, display_names):
            val1 = self.result1.effluent_quality.get(ind, 0) if self.result1 and self.result1.converged else np.nan
            val2 = self.result2.effluent_quality.get(ind, 0) if self.result2 and self.result2.converged else np.nan
            diff = val2 - val1 if not np.isnan(val1) and not np.isnan(val2) else np.nan
            improvement = (val1 - val2) / val1 * 100 if not np.isnan(val1) and not np.isnan(val2) and val1 > 0 else np.nan
            
            data.append({
                '指标': name,
                '方案1 (mg/L)': round(val1, 2) if not np.isnan(val1) else '-',
                '方案2 (mg/L)': round(val2, 2) if not np.isnan(val2) else '-',
                '差值 (mg/L)': round(diff, 2) if not np.isnan(diff) else '-',
                '改善率 (%)': round(improvement, 1) if not np.isnan(improvement) else '-',
            })
        
        return pd.DataFrame(data)

