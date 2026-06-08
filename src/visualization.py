"""
可视化模块
包含各类图表绘制功能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from .asm1_model import COMPONENT_NAMES, COMPONENT_INDEX, NUM_COMPONENTS, aggregate_to_wq_indices
from .reactor_units import ProcessFlowSheet, ReactorUnit, ReactorType
from .process_templates import InfluentConfig
from .solver import SteadyStateResult, DynamicResult
from .analysis import SensitivityResult, TwoFactorSensitivityResult, ComplianceResult, ComplianceItem


COLORS = {
    'COD': '#1f77b4',
    'BOD5': '#ff7f0e',
    'NH3-N': '#2ca02c',
    'TN': '#d62728',
    'TP': '#9467bd',
    'SS': '#8c564b',
}

REACTOR_COLORS = {
    ReactorType.GRIT: '#a1c9f4',
    ReactorType.PRIMARY: '#8de5a1',
    ReactorType.ANAEROBIC: '#6b6b6b',
    ReactorType.ANOXIC: '#9467bd',
    ReactorType.AEROBIC: '#1f77b4',
    ReactorType.SECONDARY: '#ff9f9b',
    ReactorType.DISINFECTION: '#d0bbff',
    ReactorType.MEMBRANE: '#ffbb78',
}


def plot_reactor_stack(pfs: ProcessFlowSheet, reactor_states: List[np.ndarray]) -> go.Figure:
    """
    绘制各池组分浓度堆叠柱状图
    """
    reactor_names = [r.name for r in pfs.reactors]
    
    cod_fractions = {
        'SI': [], 'XI': [], 'SS': [], 'XS': [],
        'XBH': [], 'XBA': [], 'XP': [],
    }
    
    nitrogen_fractions = {
        'SNH': [], 'SNO': [], 'SND': [], 'XND': [], 'Biomass_N': [],
    }
    
    for state in reactor_states:
        SI, XI, SS, XS, XBH, XBA, XP = state[0:7]
        cod_fractions['SI'].append(SI)
        cod_fractions['XI'].append(XI)
        cod_fractions['SS'].append(SS)
        cod_fractions['XS'].append(XS)
        cod_fractions['XBH'].append(XBH)
        cod_fractions['XBA'].append(XBA)
        cod_fractions['XP'].append(XP)
        
        SNH, SNO, SND, XND = state[9], state[8], state[10], state[11]
        biomass_N = 0.08 * (XBH + XBA) + 0.06 * XP
        nitrogen_fractions['SNH'].append(SNH)
        nitrogen_fractions['SNO'].append(SNO)
        nitrogen_fractions['SND'].append(SND)
        nitrogen_fractions['XND'].append(XND)
        nitrogen_fractions['Biomass_N'].append(biomass_N)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('各池COD组分分布 (mg COD/L)', '各池氮组分分布 (mg N/L)'),
        horizontal_spacing=0.1,
    )
    
    cod_labels = ['SI(惰性溶解)', 'XI(惰性颗粒)', 'SS(易降解)', 'XS(缓慢降解)', 'XBH(异养菌)', 'XBA(自养菌)', 'XP(代谢产物)']
    cod_colors = ['#95a5a6', '#7f8c8d', '#3498db', '#2980b9', '#27ae60', '#16a085', '#f39c12']
    
    for i, (key, values) in enumerate(cod_fractions.items()):
        fig.add_trace(
            go.Bar(name=cod_labels[i], x=reactor_names, y=values,
                   marker_color=cod_colors[i], showlegend=True),
            row=1, col=1,
        )
    
    n_labels = ['SNH(氨氮)', 'SNO(硝酸盐)', 'SND(溶解有机氮)', 'XND(颗粒有机氮)', '菌体氮']
    n_colors = ['#e74c3c', '#3498db', '#9b59b6', '#8e44ad', '#2ecc71']
    
    for i, (key, values) in enumerate(nitrogen_fractions.items()):
        fig.add_trace(
            go.Bar(name=n_labels[i], x=reactor_names, y=values,
                   marker_color=n_colors[i], showlegend=True),
            row=1, col=2,
        )
    
    fig.update_layout(
        barmode='stack',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.3),
    )
    fig.update_yaxes(title_text='浓度', row=1, col=1)
    fig.update_yaxes(title_text='浓度', row=1, col=2)
    
    return fig


def plot_process_diagram(pfs: ProcessFlowSheet) -> go.Figure:
    """
    绘制工艺流程示意图
    """
    fig = go.Figure()
    
    num_reactors = len(pfs.reactors)
    spacing = 3.0
    start_x = 1.0
    
    shapes = []
    annotations = []
    
    for i, reactor in enumerate(pfs.reactors):
        x = start_x + i * spacing
        color = REACTOR_COLORS.get(reactor.reactor_type, '#cccccc')
        
        width = 1.5
        height = 1.2
        y = 0.5
        
        shapes.append(dict(
            type='rect',
            x0=x - width/2, y0=y - height/2,
            x1=x + width/2, y1=y + height/2,
            fillcolor=color,
            line=dict(color='#333333', width=2),
        ))
        
        annotations.append(dict(
            x=x, y=y + 0.05,
            text=reactor.get_icon(),
            showarrow=False,
            font=dict(size=24),
        ))
        
        annotations.append(dict(
            x=x, y=y - 0.3,
            text=f"<b>{reactor.name}</b>",
            showarrow=False,
            font=dict(size=12, color='white'),
        ))
        
        annotations.append(dict(
            x=x, y=y - 0.55,
            text=f"V={reactor.geometry.volume:.0f}m³<br>"
                 f"HRT={reactor.operation.HRT:.1f}h<br>"
                 f"{'DO=' + str(reactor.operation.DO_setpoint) + 'mg/L' if reactor.reactor_type == ReactorType.AEROBIC else ''}",
            showarrow=False,
            font=dict(size=10),
            align='center',
        ))
        
        if i < num_reactors - 1:
            x_arrow_start = x + width/2
            x_arrow_end = x + spacing - width/2
            shapes.append(dict(
                type='line',
                x0=x_arrow_start, y0=y,
                x1=x_arrow_end, y1=y,
                line=dict(width=2, color='#555555'),
            ))
            annotations.append(dict(
                x=(x_arrow_start + x_arrow_end) / 2,
                y=y + 0.1,
                text='→',
                showarrow=False,
                font=dict(size=14),
            ))
    
    sec_idx = None
    for i, r in enumerate(pfs.reactors):
        if r.reactor_type == ReactorType.SECONDARY:
            sec_idx = i
            break
    
    if sec_idx is not None and sec_idx > 0:
        x_sec = start_x + sec_idx * spacing
        x_first = start_x
        R = pfs.reactors[sec_idx].operation.return_sludge_ratio
        
        shapes.append(dict(
            type='path',
            path=f'M {x_sec} {y - height/2 - 0.2} '
                 f'L {x_sec} {y - 1.5} '
                 f'L {x_first} {y - 1.5} '
                 f'L {x_first} {y - height/2 - 0.2}',
            fillcolor=None,
            line=dict(color='red', width=2, dash='dash'),
        ))
        
        annotations.append(dict(
            x=(x_sec + x_first) / 2, y=y - 1.3,
            text=f'回流 R={R:.0%}',
            showarrow=False,
            font=dict(size=10, color='red'),
        ))
    
    if len(pfs.reactors) > 2:
        for i, reactor in enumerate(pfs.reactors):
            if reactor.operation.internal_return_ratio > 0:
                x_from = start_x + 2 * spacing
                x_to = start_x + i * spacing
                IR = reactor.operation.internal_return_ratio
                
                shapes.append(dict(
                    type='path',
                    path=f'M {x_from} {y + height/2 + 0.2} '
                         f'L {x_from} {y + 1.2} '
                         f'L {x_to} {y + 1.2} '
                         f'L {x_to} {y + height/2 + 0.2}',
                    fillcolor=None,
                    line=dict(color='green', width=2, dash='dot'),
                ))
                
                annotations.append(dict(
                    x=(x_from + x_to) / 2, y=y + 1.4,
                    text=f'内回流 IR={IR:.0%}',
                    showarrow=False,
                    font=dict(size=10, color='green'),
                ))
                break
    
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0, start_x + num_reactors * spacing]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-2, 2]),
        height=500,
        margin=dict(l=20, r=20, t=40, b=80),
        title='工艺流程示意图',
    )
    
    return fig


def plot_effluent_timeseries(dynamic_result: DynamicResult) -> go.Figure:
    """
    绘制出水指标时间序列曲线
    """
    time_days = dynamic_result.time_days
    effluent_history = dynamic_result.effluent_quality_history
    
    fig = go.Figure()
    
    indicators = ['COD', 'NH3_N', 'TN', 'TP', 'SS']
    labels = ['COD', 'NH3-N', 'TN', 'TP', 'SS']
    
    for ind, label in zip(indicators, labels):
        values = [h[ind] for h in effluent_history]
        fig.add_trace(go.Scatter(
            x=time_days, y=values,
            mode='lines', name=label,
            line=dict(color=COLORS.get(label, '#333333'), width=2),
        ))
    
    fig.update_layout(
        title='出水水质动态变化',
        xaxis_title='时间 (天)',
        yaxis_title='浓度 (mg/L)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
    )
    
    return fig


def plot_influent_diurnal(influent: InfluentConfig) -> go.Figure:
    """
    绘制进水日变化曲线
    """
    hours = np.arange(24)
    
    if influent.flow_mode == 'constant':
        flow_values = np.ones(24) * influent.Q_base
        conc_values = np.ones(24)
    else:
        flow_values = influent.diurnal_flow_curve * influent.Q_base
        conc_values = influent.diurnal_curve
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('进水流量日变化', '进水浓度日变化'),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    
    fig.add_trace(
        go.Scatter(x=hours, y=flow_values, mode='lines+markers',
                   fill='tozeroy', name='流量',
                   line=dict(color='#1f77b4')),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(x=hours, y=conc_values, mode='lines+markers',
                   fill='tozeroy', name='浓度系数',
                   line=dict(color='#2ca02c')),
        row=2, col=1,
    )
    
    fig.update_xaxes(title_text='时间 (小时)', row=2, col=1)
    fig.update_yaxes(title_text='流量 (m³/day)', row=1, col=1)
    fig.update_yaxes(title_text='浓度系数', row=2, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=False,
    )
    
    return fig


def plot_compliance_radar(compliance_result: ComplianceResult) -> go.Figure:
    """
    绘制达标雷达图
    """
    labels = []
    ratios = []
    
    for item in compliance_result.items:
        labels.append(item.name)
        ratios.append(min(item.ratio * 100, 150))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=ratios,
        theta=labels,
        fill='toself',
        name='占标率 (%)',
        line=dict(color='#1f77b4'),
        fillcolor='rgba(31, 119, 180, 0.3)',
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[100] * len(labels),
        theta=labels,
        mode='lines',
        name='达标线 (100%)',
        line=dict(color='red', dash='dash'),
        fill=None,
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 150],
                ticktext=['0%', '50%', '100%', '150%'],
                tickvals=[0, 50, 100, 150],
            ),
            angularaxis=dict(
                tickfont=dict(size=12),
            ),
        ),
        showlegend=True,
        height=500,
        title='出水水质达标雷达图(占标率%)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.1),
    )
    
    return fig


def plot_sensitivity_curves(sensitivity_result: SensitivityResult) -> go.Figure:
    """
    绘制敏感性分析响应曲线
    """
    param_info = sensitivity_result.parameter_name
    param_values = sensitivity_result.parameter_values
    
    fig = go.Figure()
    
    indicators = ['COD', 'NH3_N', 'TN', 'TP', 'SS']
    labels = ['COD', 'NH3-N', 'TN', 'TP', 'SS']
    
    for ind, label in zip(indicators, labels):
        values = [h[ind] for h in sensitivity_result.effluent_results]
        fig.add_trace(go.Scatter(
            x=param_values, y=values,
            mode='lines+markers', name=label,
            line=dict(color=COLORS.get(label, '#333333'), width=2),
        ))
    
    from .analysis import SENSITIVITY_PARAMETERS
    param_display = SENSITIVITY_PARAMETERS.get(param_info, None)
    x_label = param_display.display_name if param_display else param_info
    x_unit = param_display.unit if param_display else ''
    
    fig.update_layout(
        title=f'{x_label} 对出水水质的影响',
        xaxis_title=f'{x_label} ({x_unit})',
        yaxis_title='出水浓度 (mg/L)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
    )
    
    return fig


def plot_two_factor_heatmap(two_factor_result: TwoFactorSensitivityResult,
                            indicator: str = 'NH3_N') -> go.Figure:
    """
    绘制双因素敏感性分析热力图
    """
    heatmap_data = two_factor_result.get_heatmap_data(indicator)
    
    from .analysis import SENSITIVITY_PARAMETERS
    
    p1 = SENSITIVITY_PARAMETERS.get(two_factor_result.param1_name, None)
    p2 = SENSITIVITY_PARAMETERS.get(two_factor_result.param2_name, None)
    
    x_label = p1.display_name if p1 else two_factor_result.param1_name
    y_label = p2.display_name if p2 else two_factor_result.param2_name
    x_unit = p1.unit if p1 else ''
    y_unit = p2.unit if p2 else ''
    
    indicator_labels = {
        'COD': 'COD', 'NH3_N': 'NH3-N', 'TN': 'TN', 'TP': 'TP', 'SS': 'SS'
    }
    ind_label = indicator_labels.get(indicator, indicator)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=two_factor_result.param1_values,
        y=two_factor_result.param2_values,
        colorscale='Viridis',
        colorbar=dict(title=f'{ind_label} (mg/L)'),
        hovertemplate=f'{x_label}: %{{x}} {x_unit}<br>'
                      f'{y_label}: %{{y}} {y_unit}<br>'
                      f'{ind_label}: %{{z:.2f}} mg/L<extra></extra>',
    ))
    
    fig.update_layout(
        title=f'{x_label} 与 {y_label} 对 {ind_label} 的联合影响',
        xaxis_title=f'{x_label} ({x_unit})',
        yaxis_title=f'{y_label} ({y_unit})',
        height=550,
    )
    
    return fig


def plot_residual_convergence(steady_result: SteadyStateResult) -> go.Figure:
    """
    绘制迭代收敛曲线
    """
    iterations = list(range(1, len(steady_result.residual_history) + 1))
    residuals = steady_result.residual_history
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations, y=residuals,
        mode='lines+markers',
        name='残差范数',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=6),
    ))
    
    fig.add_hline(
        y=1e-6, line_dash='dash', line_color='green',
        annotation_text='收敛阈值 (1e-6)',
        annotation_position='bottom right',
    )
    
    fig.update_layout(
        title='稳态求解收敛过程',
        xaxis_title='迭代次数',
        yaxis_title='残差范数 (对数尺度)',
        yaxis_type='log',
        height=400,
        showlegend=False,
    )
    
    return fig
