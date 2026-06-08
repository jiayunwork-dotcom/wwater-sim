"""
分析模块
包含出水达标判定、参数敏感性分析、工艺优化建议
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import copy
import pandas as pd

from .asm1_model import ASM1Parameters, aggregate_to_wq_indices
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
