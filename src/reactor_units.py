"""
反应器单元模型模块
包含CSTR、二沉池、膜组件等单元
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import copy

from .asm1_model import (
    ASM1Parameters,
    calculate_cstr_derivatives,
    calculate_reaction_contributions,
    COMPONENT_INDEX,
    NUM_COMPONENTS,
    aggregate_to_wq_indices,
)


class ReactorType(Enum):
    BAR_SCREEN = "bar_screen"
    GRIT = "grit"
    PRIMARY = "primary"
    ANAEROBIC = "anaerobic"
    ANOXIC = "anoxic"
    AEROBIC = "aerobic"
    SECONDARY = "secondary"
    DISINFECTION = "disinfection"
    MEMBRANE = "membrane"


REACTOR_TYPE_NAMES = {
    ReactorType.BAR_SCREEN: "格栅",
    ReactorType.GRIT: "沉砂池",
    ReactorType.PRIMARY: "初沉池",
    ReactorType.ANAEROBIC: "厌氧池",
    ReactorType.ANOXIC: "缺氧池",
    ReactorType.AEROBIC: "好氧池",
    ReactorType.SECONDARY: "二沉池",
    ReactorType.DISINFECTION: "消毒池",
    ReactorType.MEMBRANE: "膜组件",
}

REACTOR_TYPE_ICONS = {
    ReactorType.BAR_SCREEN: "🔲",
    ReactorType.GRIT: "🏖️",
    ReactorType.PRIMARY: "🔄",
    ReactorType.ANAEROBIC: "⚫",
    ReactorType.ANOXIC: "🟣",
    ReactorType.AEROBIC: "🔵",
    ReactorType.SECONDARY: "🔻",
    ReactorType.DISINFECTION: "💧",
    ReactorType.MEMBRANE: "🔬",
}


@dataclass
class ReactorGeometry:
    """反应器几何参数"""
    volume: float = 1000.0
    area: float = 100.0
    height: float = 5.0


@dataclass
class ReactorOperation:
    """反应器运行参数"""
    HRT: float = 8.0
    SRT: float = 10.0
    DO_setpoint: float = 2.0
    return_sludge_ratio: float = 0.5
    internal_return_ratio: float = 0.0


@dataclass
class ReactorUnit:
    """反应器单元基类"""
    name: str
    reactor_type: ReactorType
    geometry: ReactorGeometry = field(default_factory=ReactorGeometry)
    operation: ReactorOperation = field(default_factory=ReactorOperation)
    C: np.ndarray = field(default_factory=lambda: np.zeros(NUM_COMPONENTS))
    C_out: np.ndarray = field(default_factory=lambda: np.zeros(NUM_COMPONENTS))
    
    def get_type_name(self) -> str:
        return REACTOR_TYPE_NAMES.get(self.reactor_type, str(self.reactor_type))
    
    def get_icon(self) -> str:
        return REACTOR_TYPE_ICONS.get(self.reactor_type, "⬜")
    
    def is_biological(self) -> bool:
        return self.reactor_type in [
            ReactorType.ANAEROBIC, ReactorType.ANOXIC, ReactorType.AEROBIC
        ]
    
    def needs_aeration(self) -> bool:
        return self.reactor_type == ReactorType.AEROBIC
    
    def is_anoxic(self) -> bool:
        return self.reactor_type in [ReactorType.ANAEROBIC, ReactorType.ANOXIC]


class CSTRReactor(ReactorUnit):
    """连续搅拌反应器(CSTR)"""
    
    def __init__(self, name: str, reactor_type: ReactorType,
                 geometry: Optional[ReactorGeometry] = None,
                 operation: Optional[ReactorOperation] = None):
        geometry = geometry or ReactorGeometry()
        operation = operation or ReactorOperation()
        super().__init__(name, reactor_type, geometry, operation)
        self.sludge_waste_flow = 0.0
    
    def calculate_derivatives(self, C_in: np.ndarray, Q: float, params: ASM1Parameters,
                              return_sludge: Optional[Tuple[float, np.ndarray]] = None,
                              internal_return: Optional[Tuple[float, np.ndarray]] = None) -> np.ndarray:
        """
        计算CSTR的dC/dt
        
        参数:
            C_in: 进水浓度 [13]
            Q: 进水流量 (m³/day)
            params: ASM1参数
            return_sludge: 回流污泥 (流量, 浓度)
            internal_return: 内回流 (流量, 浓度)
        
        返回:
            dCdt: 时间导数 [13]
        """
        additional_inputs = []
        
        if return_sludge is not None:
            Q_r, C_r = return_sludge
            additional_inputs.append((Q_r, C_r))
        
        if internal_return is not None:
            Q_ir, C_ir = internal_return
            additional_inputs.append((Q_ir, C_ir))
        
        is_anoxic = self.is_anoxic()
        DO_sp = self.operation.DO_setpoint if self.needs_aeration() else 0.0
        
        dCdt = calculate_cstr_derivatives(
            self.C, C_in, Q, self.geometry.volume,
            params, DO_sp, is_anoxic, additional_inputs
        )
        
        if self.operation.SRT > 0 and self.reactor_type in [ReactorType.AEROBIC, ReactorType.ANOXIC, ReactorType.ANAEROBIC]:
            V = self.geometry.volume
            SRT = self.operation.SRT
            waste_rate = V / SRT
            MLSS_indices = [
                COMPONENT_INDEX['XI'],
                COMPONENT_INDEX['XS'],
                COMPONENT_INDEX['XBH'],
                COMPONENT_INDEX['XBA'],
                COMPONENT_INDEX['XP'],
            ]
            for idx in MLSS_indices:
                dCdt[idx] -= (waste_rate / V) * self.C[idx]
        
        return dCdt
    
    def update_state(self, dCdt: np.ndarray, dt: float) -> None:
        """更新反应器状态"""
        self.C += dCdt * dt
        self.C = np.maximum(self.C, 0.0)
        self.C_out = self.C.copy()


class SecondaryClarifier(ReactorUnit):
    """二沉池 - 简化一维通量模型"""
    
    def __init__(self, name: str,
                 geometry: Optional[ReactorGeometry] = None,
                 operation: Optional[ReactorOperation] = None):
        geometry = geometry or ReactorGeometry(volume=500, area=100, height=4)
        operation = operation or ReactorOperation(HRT=2, SRT=0, DO_setpoint=0, return_sludge_ratio=0.5)
        super().__init__(name, ReactorType.SECONDARY, geometry, operation)
        
        self.settling_velocity = 10.0
        self.effluent_SS_fraction = 0.01
        self.return_sludge_concentration_factor = 2.5
    
    def process(self, C_in: np.ndarray, Q: float, params: Optional[ASM1Parameters] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        二沉池固液分离处理
        
        参数:
            C_in: 进水浓度 [13]
            Q: 进水流量 (m³/day)
        
        返回:
            effluent: 出水浓度 [13]
            return_sludge: 回流污泥浓度 [13]
            waste_sludge_flow: 排泥流量 (m³/day)
        """
        A = self.geometry.area
        v_s = self.settling_velocity
        
        solid_indices = [
            COMPONENT_INDEX['XI'],
            COMPONENT_INDEX['XS'],
            COMPONENT_INDEX['XBH'],
            COMPONENT_INDEX['XBA'],
            COMPONENT_INDEX['XP'],
        ]
        
        self.C = C_in.copy()
        self.C_out = C_in.copy()
        
        for idx in solid_indices:
            flux_in = C_in[idx] * Q
            settling_flux = v_s * A * C_in[idx]
            
            actual_settling = min(settling_flux, flux_in * 0.99)
            
            effluent_solids = flux_in - actual_settling
            effluent_solids = max(effluent_solids, flux_in * self.effluent_SS_fraction)
            
            self.C_out[idx] = effluent_solids / Q if Q > 0 else 0
            
            R = self.operation.return_sludge_ratio
            Q_r = R * Q
            Q_e = Q
            Q_w = Q / 100.0
            
            solids_to_sludge = flux_in - effluent_solids
            if Q_r + Q_w > 0:
                return_sludge = solids_to_sludge / (Q_r + Q_w)
                return_sludge_conc = return_sludge * self.return_sludge_concentration_factor
            else:
                return_sludge_conc = 0
            
            self.return_sludge_C = C_in.copy()
            for s_idx in solid_indices:
                self.return_sludge_C[s_idx] = return_sludge_conc
            
        dissolved_indices = [i for i in range(NUM_COMPONENTS) if i not in solid_indices]
        for idx in dissolved_indices:
            self.C_out[idx] = C_in[idx]
            self.return_sludge_C[idx] = C_in[idx]
        
        return self.C_out, self.return_sludge_C, Q_w


class MembraneUnit(ReactorUnit):
    """膜组件单元"""
    
    def __init__(self, name: str,
                 geometry: Optional[ReactorGeometry] = None,
                 operation: Optional[ReactorOperation] = None):
        geometry = geometry or ReactorGeometry(volume=200, area=1000, height=3)
        operation = operation or ReactorOperation(HRT=1, SRT=0, DO_setpoint=0, return_sludge_ratio=1.0)
        super().__init__(name, ReactorType.MEMBRANE, geometry, operation)
        
        self.membrane_flux = 15.0
        self.SS_removal = 0.999
        self.BOD_removal = 0.1
    
    def process(self, C_in: np.ndarray, Q: float, params: Optional[ASM1Parameters] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        膜过滤处理
        
        返回:
            effluent, retentate
        """
        self.C = C_in.copy()
        self.C_out = C_in.copy()
        
        solid_indices = [
            COMPONENT_INDEX['XI'],
            COMPONENT_INDEX['XS'],
            COMPONENT_INDEX['XBH'],
            COMPONENT_INDEX['XBA'],
            COMPONENT_INDEX['XP'],
        ]
        
        for idx in solid_indices:
            removed = C_in[idx] * self.SS_removal
            self.C_out[idx] = C_in[idx] - removed
        
        bod_indices = [COMPONENT_INDEX['SS'], COMPONENT_INDEX['XS']]
        for idx in bod_indices:
            removed = C_in[idx] * self.BOD_removal
            self.C_out[idx] -= removed
        
        R = self.operation.return_sludge_ratio
        Q_perm = Q
        Q_ret = R * Q
        
        retentate = C_in.copy()
        for idx in solid_indices:
            total_in = C_in[idx] * Q
            total_perm = self.C_out[idx] * Q_perm
            retentate[idx] = (total_in - total_perm) / Q_ret if Q_ret > 0 else 0
        
        self.retentate_C = retentate
        
        return self.C_out, self.retentate


class SimpleTreatment(ReactorUnit):
    """简单处理单元(沉砂池/初沉池/消毒池)"""
    
    def __init__(self, name: str, reactor_type: ReactorType,
                 removal_rates: Optional[Dict[str, float]] = None,
                 geometry: Optional[ReactorGeometry] = None,
                 operation: Optional[ReactorOperation] = None):
        geometry = geometry or ReactorGeometry(volume=100, area=50, height=3)
        operation = operation or ReactorOperation(HRT=0.5, SRT=0, DO_setpoint=0)
        super().__init__(name, reactor_type, geometry, operation)
        
        default_removals = {
            ReactorType.BAR_SCREEN: {'SS': 0.1, 'BOD5': 0.05, 'COD': 0.03},
            ReactorType.GRIT: {'SS': 0.2, 'BOD5': 0.05},
            ReactorType.PRIMARY: {'SS': 0.5, 'BOD5': 0.3, 'COD': 0.25},
            ReactorType.DISINFECTION: {'NH3_N': 0.1, 'BOD5': 0.05},
        }
        
        self.removal_rates = removal_rates or default_removals.get(reactor_type, {})
    
    def process(self, C_in: np.ndarray, Q: float, params: Optional[ASM1Parameters] = None) -> np.ndarray:
        """简单去除处理"""
        self.C = C_in.copy()
        self.C_out = C_in.copy()
        
        wq = aggregate_to_wq_indices(C_in)
        
        for param, removal in self.removal_rates.items():
            if param == 'SS':
                indices = [COMPONENT_INDEX['XI'], COMPONENT_INDEX['XS'],
                           COMPONENT_INDEX['XBH'], COMPONENT_INDEX['XBA'],
                           COMPONENT_INDEX['XP']]
                for idx in indices:
                    self.C_out[idx] *= (1 - removal)
            elif param == 'COD':
                indices = [COMPONENT_INDEX['SI'], COMPONENT_INDEX['SS'],
                           COMPONENT_INDEX['XI'], COMPONENT_INDEX['XS']]
                for idx in indices:
                    self.C_out[idx] *= (1 - removal)
            elif param == 'BOD5':
                self.C_out[COMPONENT_INDEX['SS']] *= (1 - removal * 0.8)
                self.C_out[COMPONENT_INDEX['XS']] *= (1 - removal * 0.6)
            elif param == 'NH3_N':
                self.C_out[COMPONENT_INDEX['SNH']] *= (1 - removal)
        
        return self.C_out


@dataclass
class ProcessFlowSheet:
    """工艺流程配置"""
    reactors: List[ReactorUnit] = field(default_factory=list)
    connections: List[Tuple[int, int]] = field(default_factory=list)
    
    def add_reactor(self, reactor: ReactorUnit) -> int:
        """添加反应器，返回索引"""
        self.reactors.append(reactor)
        return len(self.reactors) - 1
    
    def connect(self, from_idx: int, to_idx: int) -> None:
        """连接两个反应器"""
        self.connections.append((from_idx, to_idx))
    
    def get_reactor_names(self) -> List[str]:
        return [r.name for r in self.reactors]
    
    def get_aerobic_reactors(self) -> List[int]:
        return [i for i, r in enumerate(self.reactors) 
                if r.reactor_type == ReactorType.AEROBIC]
    
    def get_secondary_clarifiers(self) -> List[int]:
        return [i for i, r in enumerate(self.reactors)
                if r.reactor_type == ReactorType.SECONDARY]
    
    def get_membrane_units(self) -> List[int]:
        return [i for i, r in enumerate(self.reactors)
                if r.reactor_type == ReactorType.MEMBRANE]
    
    def validate(self) -> Tuple[bool, str]:
        """验证工艺流程"""
        if len(self.reactors) < 1:
            return False, "至少需要一个处理单元"
        return True, ""
    
    def get_volume(self) -> float:
        """获取总容积"""
        return sum(r.geometry.volume for r in self.reactors)
    
    def get_total_HRT(self, Q: float) -> float:
        """获取总HRT(小时)"""
        return self.get_volume() / Q * 24 if Q > 0 else 0
    
    def to_dict(self) -> Dict:
        """导出为字典"""
        return {
            'reactors': [
                {
                    'name': r.name,
                    'type': r.reactor_type.value,
                    'volume': r.geometry.volume,
                    'area': r.geometry.area,
                    'height': r.geometry.height,
                    'HRT': r.operation.HRT,
                    'SRT': r.operation.SRT,
                    'DO': r.operation.DO_setpoint,
                    'return_ratio': r.operation.return_sludge_ratio,
                    'internal_return': r.operation.internal_return_ratio,
                }
                for r in self.reactors
            ],
            'connections': self.connections,
        }


def create_reactor_by_type(reactor_type: ReactorType, name: str, **kwargs) -> ReactorUnit:
    """根据类型创建反应器"""
    geometry = ReactorGeometry(
        volume=kwargs.get('volume', 1000),
        area=kwargs.get('area', 100),
        height=kwargs.get('height', 5),
    )
    operation = ReactorOperation(
        HRT=kwargs.get('HRT', 8),
        SRT=kwargs.get('SRT', 10),
        DO_setpoint=kwargs.get('DO_setpoint', 2.0),
        return_sludge_ratio=kwargs.get('return_sludge_ratio', 0.5),
        internal_return_ratio=kwargs.get('internal_return_ratio', 0.0),
    )
    
    if reactor_type == ReactorType.SECONDARY:
        return SecondaryClarifier(name, geometry, operation)
    elif reactor_type == ReactorType.MEMBRANE:
        return MembraneUnit(name, geometry, operation)
    elif reactor_type in [ReactorType.BAR_SCREEN, ReactorType.GRIT, ReactorType.PRIMARY, ReactorType.DISINFECTION]:
        return SimpleTreatment(name, reactor_type, geometry=geometry, operation=operation)
    elif reactor_type in [ReactorType.ANAEROBIC, ReactorType.ANOXIC, ReactorType.AEROBIC]:
        return CSTRReactor(name, reactor_type, geometry, operation)
    else:
        return CSTRReactor(name, reactor_type, geometry, operation)
