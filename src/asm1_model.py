"""
ASM1活性污泥模型核心模块
基于IWA发布的Activated Sludge Model No.1
包含13个组分、8个生化反应过程
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json


NUM_COMPONENTS = 13

COMPONENT_NAMES = [
    'SI',    # 0: 溶解性惰性有机物 (mg COD/L)
    'XI',    # 1: 颗粒性惰性有机物 (mg COD/L)
    'SS',    # 2: 易降解有机物 (mg COD/L)
    'XS',    # 3: 缓慢降解有机物 (mg COD/L)
    'XBH',   # 4: 活性异养菌 (mg COD/L)
    'XBA',   # 5: 活性自养菌 (mg COD/L)
    'XP',    # 6: 颗粒性代谢产物 (mg COD/L)
    'SO',    # 7: 溶解氧 (mg O2/L)
    'SNO',   # 8: 硝酸盐 (mg N/L)
    'SNH',   # 9: 氨氮 (mg N/L)
    'SND',   # 10: 溶解性可降解有机氮 (mg N/L)
    'XND',   # 11: 颗粒性可降解有机氮 (mg N/L)
    'SALK',  # 12: 碱度 (mol HCO3-/m³)
]

COMPONENT_INDEX = {name: i for i, name in enumerate(COMPONENT_NAMES)}

COMPONENT_UNITS = [
    'mg COD/L', 'mg COD/L', 'mg COD/L', 'mg COD/L',
    'mg COD/L', 'mg COD/L', 'mg COD/L', 'mg O2/L',
    'mg N/L', 'mg N/L', 'mg N/L', 'mg N/L', 'mol/m³'
]

COMPONENT_DESCRIPTIONS = [
    '溶解性惰性有机物',
    '颗粒性惰性有机物',
    '易降解有机物',
    '缓慢降解有机物',
    '活性异养菌',
    '活性自养菌',
    '颗粒性代谢产物',
    '溶解氧',
    '硝酸盐',
    '氨氮',
    '溶解性可降解有机氮',
    '颗粒性可降解有机氮',
    '碱度',
]

NUM_PROCESSES = 8

PROCESS_NAMES = [
    '好氧异养菌生长',
    '缺氧异养菌生长(反硝化)',
    '好氧自养菌生长(硝化)',
    '异养菌衰减',
    '自养菌衰减',
    '可溶性有机氮氨化',
    '颗粒有机物水解',
    '颗粒有机氮水解',
]


@dataclass
class ASM1Parameters:
    """
    ASM1动力学参数和化学计量系数
    默认值为20°C标准值
    """
    
    Y_H: float = 0.67
    Y_A: float = 0.24
    f_P: float = 0.08
    i_XB: float = 0.08
    i_XP: float = 0.06
    
    mu_H: float = 6.0
    K_S: float = 20.0
    K_O_H: float = 0.2
    K_NO: float = 0.5
    b_H: float = 0.62
    eta_g: float = 0.8
    eta_h: float = 0.4
    k_h: float = 3.0
    K_X: float = 0.03
    mu_A: float = 0.8
    K_O_A: float = 0.4
    K_NH: float = 1.0
    b_A: float = 0.20
    k_a: float = 0.08
    S_ALK: float = 7.0
    
    theta_mu_H: float = 1.072
    theta_mu_A: float = 1.103
    theta_b_H: float = 1.029
    theta_b_A: float = 1.030
    theta_k_h: float = 1.072
    theta_k_a: float = 1.072
    
    temperature: float = 20.0
    
    description: Dict = field(default_factory=lambda: {
        'Y_H': ('异养菌产率系数', 'mg COD/mg COD'),
        'Y_A': ('自养菌产率系数', 'mg COD/mg N'),
        'f_P': ('衰减产生的惰性颗粒组分比例', '无量纲'),
        'i_XB': ('菌体中的氮含量', 'mg N/mg COD'),
        'i_XP': ('惰性颗粒产物中的氮含量', 'mg N/mg COD'),
        'mu_H': ('异养菌最大比生长速率', '1/day'),
        'K_S': ('异养菌半饱和系数(易降解COD)', 'mg COD/L'),
        'K_O_H': ('异养菌氧半饱和系数', 'mg O2/L'),
        'K_NO': ('硝酸盐半饱和系数', 'mg N/L'),
        'b_H': ('异养菌衰减系数', '1/day'),
        'eta_g': ('缺氧条件下最大比生长速率修正系数', '无量纲'),
        'eta_h': ('缺氧条件下水解速率修正系数', '无量纲'),
        'k_h': ('颗粒有机物水解速率', '1/day'),
        'K_X': ('颗粒有机物水解半饱和系数', 'mg COD/mg COD'),
        'mu_A': ('自养菌最大比生长速率', '1/day'),
        'K_O_A': ('自养菌氧半饱和系数', 'mg O2/L'),
        'K_NH': ('氨氮半饱和系数', 'mg N/L'),
        'b_A': ('自养菌衰减系数', '1/day'),
        'k_a': ('氨化速率', 'L/(mg COD.day)'),
    })
    
    def get_temperature_corrected_params(self, T: Optional[float] = None) -> 'ASM1Parameters':
        """
        根据Arrhenius公式修正动力学参数
        """
        if T is None:
            T = self.temperature
        delta_T = T - 20.0
        
        params = ASM1Parameters()
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float)):
                setattr(params, key, value)
        
        params.mu_H = self.mu_H * (self.theta_mu_H ** delta_T)
        params.mu_A = self.mu_A * (self.theta_mu_A ** delta_T)
        params.b_H = self.b_H * (self.theta_b_H ** delta_T)
        params.b_A = self.b_A * (self.theta_b_A ** delta_T)
        params.k_h = self.k_h * (self.theta_k_h ** delta_T)
        params.k_a = self.k_a * (self.theta_k_a ** delta_T)
        params.temperature = T
        
        return params
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if k != 'description'}
    
    def from_dict(self, data: Dict) -> None:
        for key, value in data.items():
            if hasattr(self, key) and key != 'description':
                setattr(self, key, value)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def from_json(self, json_str: str) -> None:
        data = json.loads(json_str)
        self.from_dict(data)
    
    def reset_to_default(self) -> None:
        default = ASM1Parameters()
        for key in self.__dict__.keys():
            if key != 'description':
                setattr(self, key, getattr(default, key))


def get_stoichiometric_matrix() -> np.ndarray:
    """
    ASM1化学计量矩阵 (13组分 x 8过程)
    行: 组分, 列: 反应过程
    """
    v = np.zeros((NUM_COMPONENTS, NUM_PROCESSES))
    
    Y_H = 0.67
    Y_A = 0.24
    f_P = 0.08
    i_XB = 0.08
    i_XP = 0.06
    
    # 过程1: 好氧异养菌生长 (列0)
    v[COMPONENT_INDEX['SS'], 0] = -(1/Y_H)
    v[COMPONENT_INDEX['XBH'], 0] = 1.0
    v[COMPONENT_INDEX['SO'], 0] = -(1-Y_H)/Y_H
    v[COMPONENT_INDEX['SNH'], 0] = -i_XB
    v[COMPONENT_INDEX['SALK'], 0] = -i_XB/14
    
    # 过程2: 缺氧异养菌生长 (列1)
    v[COMPONENT_INDEX['SS'], 1] = -(1/Y_H)
    v[COMPONENT_INDEX['XBH'], 1] = 1.0
    v[COMPONENT_INDEX['SNO'], 1] = -(1-Y_H)/(2.86*Y_H)
    v[COMPONENT_INDEX['SNH'], 1] = -i_XB
    v[COMPONENT_INDEX['SALK'], 1] = -i_XB/14 + (1-Y_H)/(2.86*Y_H)*0.0714
    
    # 过程3: 好氧自养菌生长 (列2)
    v[COMPONENT_INDEX['XBA'], 2] = 1.0
    v[COMPONENT_INDEX['SNH'], 2] = -(i_XB + 1/Y_A)
    v[COMPONENT_INDEX['SO'], 2] = -(4.57-Y_A)/Y_A
    v[COMPONENT_INDEX['SALK'], 2] = -(i_XB + 1/Y_A)/14
    
    # 过程4: 异养菌衰减 (列3)
    v[COMPONENT_INDEX['XBH'], 3] = -1.0
    v[COMPONENT_INDEX['XP'], 3] = f_P
    v[COMPONENT_INDEX['SND'], 3] = i_XB - f_P*i_XP
    v[COMPONENT_INDEX['XI'], 3] = 1 - f_P
    
    # 过程5: 自养菌衰减 (列4)
    v[COMPONENT_INDEX['XBA'], 4] = -1.0
    v[COMPONENT_INDEX['XP'], 4] = f_P
    v[COMPONENT_INDEX['SND'], 4] = i_XB - f_P*i_XP
    v[COMPONENT_INDEX['XI'], 4] = 1 - f_P
    
    # 过程6: 可溶性有机氮氨化 (列5)
    v[COMPONENT_INDEX['SND'], 5] = -1.0
    v[COMPONENT_INDEX['SNH'], 5] = 1.0
    v[COMPONENT_INDEX['SALK'], 5] = 1/14
    
    # 过程7: 颗粒有机物水解 (列6)
    v[COMPONENT_INDEX['XS'], 6] = -1.0
    v[COMPONENT_INDEX['SS'], 6] = 1.0
    
    # 过程8: 颗粒有机氮水解 (列7)
    v[COMPONENT_INDEX['XND'], 7] = -1.0
    v[COMPONENT_INDEX['SND'], 7] = 1.0
    
    return v


def calculate_process_rates(C: np.ndarray, params: ASM1Parameters, 
                            DO_setpoint: float = 2.0, is_anoxic: bool = False) -> np.ndarray:
    """
    计算8个反应过程的速率
    
    参数:
        C: 组分浓度数组 [13]
        params: ASM1参数
        DO_setpoint: 溶解氧设定值 (mg O2/L)
        is_anoxic: 是否为缺氧/厌氧条件
    
    返回:
        rates: 反应速率数组 [8]
    """
    SI, XI, SS, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK = C
    
    if is_anoxic:
        SO_eff = 0.0
    else:
        SO_eff = max(SO, DO_setpoint) if DO_setpoint > 0 else max(SO, 0.0)
    
    rates = np.zeros(NUM_PROCESSES)
    
    rates[0] = params.mu_H * (SS / (params.K_S + SS)) * \
               (SO_eff / (params.K_O_H + SO_eff)) * XBH
    
    rates[1] = params.mu_H * (SS / (params.K_S + SS)) * \
               (params.K_O_H / (params.K_O_H + SO_eff)) * \
               (SNO / (params.K_NO + SNO)) * params.eta_g * XBH
    
    rates[2] = params.mu_A * (SNH / (params.K_NH + SNH)) * \
               (SO_eff / (params.K_O_A + SO_eff)) * XBA
    
    rates[3] = params.b_H * XBH
    rates[4] = params.b_A * XBA
    
    rates[5] = params.k_a * SND * XBH
    
    rates[6] = params.k_h * (XS / (XBH + 1e-10)) / \
               (params.K_X + (XS / (XBH + 1e-10))) * XBH
    if SO_eff < params.K_O_H:
        rates[6] *= params.eta_h
    
    rates[7] = params.k_h * (XND / (XBH + 1e-10)) / \
               (params.K_X + (XND / (XBH + 1e-10))) * XBH
    if SO_eff < params.K_O_H:
        rates[7] *= params.eta_h
    
    return rates


def calculate_reaction_contributions(C: np.ndarray, params: ASM1Parameters,
                                     DO_setpoint: float = 2.0, 
                                     is_anoxic: bool = False) -> np.ndarray:
    """
    计算各组分的反应贡献项
    
    返回:
        dC_reaction: 各组分的反应速率 [13]
    """
    stoich_matrix = get_stoichiometric_matrix()
    rates = calculate_process_rates(C, params, DO_setpoint, is_anoxic)
    return stoich_matrix @ rates


def calculate_cstr_derivatives(C: np.ndarray, C_in: np.ndarray, Q: float, V: float,
                               params: ASM1Parameters, DO_setpoint: float = 2.0,
                               is_anoxic: bool = False, 
                               additional_inputs: Optional[List[Tuple[float, np.ndarray]]] = None) -> np.ndarray:
    """
    计算CSTR反应器的dC/dt
    
    dCi/dt = (Q/V)*(Cin_i - Ci) + 反应贡献 + 其他输入
    
    参数:
        C: 当前组分浓度 [13]
        C_in: 进水组分浓度 [13]
        Q: 进水流量 (m³/day)
        V: 反应器容积 (m³)
        params: ASM1参数
        DO_setpoint: DO设定值
        is_anoxic: 是否为缺氧/厌氧
        additional_inputs: 其他输入物流 [(流量, 浓度)]
    
    返回:
        dCdt: 各组分的时间导数 [13]
    """
    reaction_term = calculate_reaction_contributions(C, params, DO_setpoint, is_anoxic)
    HRT = V / Q if Q > 0 else 1e6
    
    advection_term = (1.0 / HRT) * (C_in - C)
    
    if additional_inputs is not None:
        for Q_add, C_add in additional_inputs:
            advection_term += (Q_add / V) * (C_add - C)
    
    dCdt = advection_term + reaction_term
    
    if not is_anoxic and DO_setpoint > 0:
        SO_idx = COMPONENT_INDEX['SO']
        dCdt[SO_idx] += (DO_setpoint - C[SO_idx]) * 10.0
    
    return dCdt


def aggregate_to_wq_indices(C: np.ndarray) -> Dict[str, float]:
    """
    将ASM1组分浓度聚合成常规水质指标
    
    返回:
        {
            'COD': ...,   # mg/L
            'BOD5': ...,  # mg/L
            'NH3_N': ..., # mg/L
            'TN': ...,    # mg/L
            'TP': ...,    # mg/L (假设比例)
            'SS': ...,    # mg/L
        }
    """
    SI, XI, SS, XS, XBH, XBA, XP, SO, SNO, SNH, SND, XND, SALK = C
    
    COD = SI + XI + SS + XS + XBH + XBA + XP
    
    BOD5 = (SS + 0.6 * XS) * 0.68 + (XBH + XBA) * 0.6
    
    NH3_N = SNH
    TN = SNH + SNO + SND + XND + 0.08 * (XBH + XBA) + 0.06 * XP
    
    TP = TN * 0.05 + 1.0
    
    SS = (XI + XS + XBH + XBA + XP) * 0.75
    
    return {
        'COD': round(COD, 2),
        'BOD5': round(BOD5, 2),
        'NH3_N': round(NH3_N, 2),
        'TN': round(TN, 2),
        'TP': round(TP, 2),
        'SS': round(SS, 2),
    }


def create_influent_from_quality(COD: float, BOD5: float, NH3_N: float, 
                                 TN: float, TP: float, SS: float) -> np.ndarray:
    """
    根据常规水质指标创建ASM1进水组分向量
    """
    C = np.zeros(NUM_COMPONENTS)
    
    COD_bio = BOD5 / 0.68
    COD_nb = COD - COD_bio
    
    C[COMPONENT_INDEX['SI']] = COD_nb * 0.4
    C[COMPONENT_INDEX['XI']] = (SS * 1.33) * 0.3
    C[COMPONENT_INDEX['SS']] = COD_bio * 0.75
    C[COMPONENT_INDEX['XS']] = COD_bio * 0.25
    
    C[COMPONENT_INDEX['XBH']] = SS * 0.2
    C[COMPONENT_INDEX['XBA']] = SS * 0.02
    C[COMPONENT_INDEX['XP']] = SS * 0.1
    
    C[COMPONENT_INDEX['SNO']] = 0.5
    C[COMPONENT_INDEX['SNH']] = NH3_N * 0.95
    C[COMPONENT_INDEX['SND']] = TN * 0.03
    C[COMPONENT_INDEX['XND']] = TN * 0.02
    
    C[COMPONENT_INDEX['SO']] = 0.5
    C[COMPONENT_INDEX['SALK']] = 7.0
    
    return C


def get_typical_influent(influent_type: str = 'domestic') -> np.ndarray:
    """
    获取典型进水水质
    influent_type: 'domestic'(生活污水), 'industrial'(工业混合), 'high_strength'(高浓度有机废水)
    """
    typicals = {
        'domestic': {
            'COD': 400, 'BOD5': 200, 'NH3_N': 35, 
            'TN': 45, 'TP': 5, 'SS': 200
        },
        'industrial': {
            'COD': 800, 'BOD5': 400, 'NH3_N': 50, 
            'TN': 70, 'TP': 8, 'SS': 300
        },
        'high_strength': {
            'COD': 2000, 'BOD5': 1200, 'NH3_N': 80, 
            'TN': 120, 'TP': 15, 'SS': 500
        },
    }
    
    if influent_type not in typicals:
        influent_type = 'domestic'
    
    return create_influent_from_quality(**typicals[influent_type])
