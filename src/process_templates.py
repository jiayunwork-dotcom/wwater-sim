"""
工艺流程模板模块
预置A2O、SBR、MBR三种典型工艺
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
import numpy as np

from .reactor_units import (
    ReactorType,
    ProcessFlowSheet,
    create_reactor_by_type,
    ReactorUnit,
    CSTRReactor,
)
from .asm1_model import NUM_COMPONENTS, get_typical_influent


@dataclass
class ProcessTemplate:
    """工艺模板"""
    name: str
    description: str
    create_function: Callable[[], ProcessFlowSheet]
    
    def create(self) -> ProcessFlowSheet:
        return self.create_function()


def create_A2O_process() -> ProcessFlowSheet:
    """
    创建A2O工艺: 厌氧池 → 缺氧池 → 好氧池 → 二沉池
    带污泥回流和内回流
    """
    pfs = ProcessFlowSheet()
    
    anaerobic = create_reactor_by_type(
        ReactorType.ANAEROBIC, "厌氧池",
        volume=800, area=200, height=4,
        HRT=2, SRT=15, DO_setpoint=0.0,
        return_sludge_ratio=0.5, internal_return_ratio=0.0,
    )
    pfs.add_reactor(anaerobic)
    
    anoxic = create_reactor_by_type(
        ReactorType.ANOXIC, "缺氧池",
        volume=1000, area=250, height=4,
        HRT=2.5, SRT=15, DO_setpoint=0.0,
        return_sludge_ratio=0.0, internal_return_ratio=2.0,
    )
    pfs.add_reactor(anoxic)
    
    aerobic = create_reactor_by_type(
        ReactorType.AEROBIC, "好氧池",
        volume=2000, area=500, height=4,
        HRT=5, SRT=15, DO_setpoint=2.0,
        return_sludge_ratio=0.0, internal_return_ratio=0.0,
    )
    pfs.add_reactor(aerobic)
    
    secondary = create_reactor_by_type(
        ReactorType.SECONDARY, "二沉池",
        volume=600, area=300, height=4,
        HRT=1.5, SRT=0, DO_setpoint=0,
        return_sludge_ratio=0.5,
    )
    pfs.add_reactor(secondary)
    
    pfs.connect(0, 1)
    pfs.connect(1, 2)
    pfs.connect(2, 3)
    
    return pfs


def create_SBR_process() -> ProcessFlowSheet:
    """
    创建SBR工艺: 单池间歇运行
    循环: 进水 → 曝气 → 沉淀 → 排水
    """
    pfs = ProcessFlowSheet()
    
    sbr_reactor = create_reactor_by_type(
        ReactorType.AEROBIC, "SBR反应池",
        volume=3000, area=750, height=4,
        HRT=8, SRT=12, DO_setpoint=2.0,
        return_sludge_ratio=0.0,
    )
    pfs.add_reactor(sbr_reactor)
    
    return pfs


def create_MBR_process() -> ProcessFlowSheet:
    """
    创建MBR工艺: 厌氧池 → 好氧池 → 膜组件
    膜组件替代二沉池，泥水分离
    """
    pfs = ProcessFlowSheet()
    
    anaerobic = create_reactor_by_type(
        ReactorType.ANAEROBIC, "厌氧池",
        volume=600, area=150, height=4,
        HRT=1.5, SRT=20, DO_setpoint=0.0,
        return_sludge_ratio=0.5,
    )
    pfs.add_reactor(anaerobic)
    
    aerobic = create_reactor_by_type(
        ReactorType.AEROBIC, "好氧池",
        volume=2500, area=625, height=4,
        HRT=6, SRT=20, DO_setpoint=2.5,
        return_sludge_ratio=0.0,
    )
    pfs.add_reactor(aerobic)
    
    membrane = create_reactor_by_type(
        ReactorType.MEMBRANE, "膜组件",
        volume=400, area=2000, height=3,
        HRT=1, SRT=0, DO_setpoint=0,
        return_sludge_ratio=3.0,
    )
    pfs.add_reactor(membrane)
    
    pfs.connect(0, 1)
    pfs.connect(1, 2)
    
    return pfs


PROCESS_TEMPLATES = {
    'A2O': ProcessTemplate(
        name='A2O工艺',
        description='厌氧-缺氧-好氧工艺，适合脱氮除磷',
        create_function=create_A2O_process,
    ),
    'SBR': ProcessTemplate(
        name='SBR工艺',
        description='序批式活性污泥法，间歇运行',
        create_function=create_SBR_process,
    ),
    'MBR': ProcessTemplate(
        name='MBR工艺',
        description='膜生物反应器，高效固液分离',
        create_function=create_MBR_process,
    ),
}


def get_template_names() -> List[str]:
    return [t.name for t in PROCESS_TEMPLATES.values()]


def create_process_by_name(name: str) -> Optional[ProcessFlowSheet]:
    for key, template in PROCESS_TEMPLATES.items():
        if template.name == name or key == name:
            return template.create()
    return None


@dataclass
class InfluentConfig:
    """进水配置"""
    flow_mode: str = 'constant'
    Q_base: float = 1000.0
    quality_mode: str = 'typical'
    influent_type: str = 'domestic'
    custom_quality: Dict = field(default_factory=lambda: {
        'COD': 400, 'BOD5': 200, 'NH3_N': 35,
        'TN': 45, 'TP': 5, 'SS': 200,
    })
    diurnal_curve: np.ndarray = field(default_factory=lambda: np.ones(24))
    diurnal_flow_curve: np.ndarray = field(default_factory=lambda: np.ones(24))
    
    def get_Q(self, t_hours: float) -> float:
        """获取指定时间的流量 (m³/day)"""
        if self.flow_mode == 'constant':
            return self.Q_base
        
        t_mod = t_hours % 24
        idx = int(t_mod)
        if idx >= 24:
            idx = 23
        factor = self.diurnal_flow_curve[idx]
        return self.Q_base * factor
    
    def get_C(self, t_hours: float) -> np.ndarray:
        """获取指定时间的进水浓度 [13]"""
        if self.quality_mode == 'typical':
            C_base = get_typical_influent(self.influent_type)
        else:
            from .asm1_model import create_influent_from_quality
            C_base = create_influent_from_quality(**self.custom_quality)
        
        if self.flow_mode == 'constant':
            return C_base
        
        t_mod = t_hours % 24
        idx = int(t_mod)
        if idx >= 24:
            idx = 23
        factor = self.diurnal_curve[idx]
        return C_base * factor
    
    def set_diurnal_pattern(self, pattern_type: str = 'morning_evening_peak'):
        """设置日变化模式"""
        if pattern_type == 'morning_evening_peak':
            hours = np.arange(24)
            flow_curve = np.ones(24)
            conc_curve = np.ones(24)
            
            morning_peak = np.exp(-0.5 * ((hours - 8) / 2) ** 2)
            evening_peak = np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)
            night_trough = 0.6 + 0.4 * np.exp(-0.5 * ((hours - 3) / 3) ** 2)
            
            flow_curve = 0.5 + morning_peak * 0.8 + evening_peak * 0.7
            flow_curve = flow_curve / flow_curve.mean()
            
            conc_curve = night_trough + morning_peak * 0.5 + evening_peak * 0.4
            conc_curve = conc_curve / conc_curve.mean()
            
            self.diurnal_flow_curve = flow_curve
            self.diurnal_curve = conc_curve
        else:
            self.diurnal_flow_curve = np.ones(24)
            self.diurnal_curve = np.ones(24)
    
    def to_dict(self) -> Dict:
        return {
            'flow_mode': self.flow_mode,
            'Q_base': self.Q_base,
            'quality_mode': self.quality_mode,
            'influent_type': self.influent_type,
            'custom_quality': self.custom_quality,
            'diurnal_curve': self.diurnal_curve.tolist(),
            'diurnal_flow_curve': self.diurnal_flow_curve.tolist(),
        }
