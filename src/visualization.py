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


def plot_influent_diurnal(influent: InfluentConfig, 
                           selected_hour: Optional[int] = None,
                           show_editable_hint: bool = False) -> go.Figure:
    """
    绘制进水日变化曲线，支持高亮选中节点
    
    参数:
        influent: 进水配置
        selected_hour: 高亮显示的小时节点 (None表示不高亮)
        show_editable_hint: 是否显示可编辑提示
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
        subplot_titles=(
            '进水流量日变化' + (' (点击下方小时按钮选择节点调整)' if show_editable_hint else ''),
            '进水浓度日变化' + (' (拖拽滑块实时更新)' if show_editable_hint else '')
        ),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    
    marker_sizes_flow = [8] * 24
    marker_colors_flow = ['#1f77b4'] * 24
    marker_sizes_conc = [8] * 24
    marker_colors_conc = ['#2ca02c'] * 24
    
    if selected_hour is not None and 0 <= selected_hour < 24:
        marker_sizes_flow[selected_hour] = 18
        marker_colors_flow[selected_hour] = '#ff4b4b'
        marker_sizes_conc[selected_hour] = 18
        marker_colors_conc[selected_hour] = '#ff4b4b'
    
    fig.add_trace(
        go.Scatter(x=hours, y=flow_values, mode='lines+markers',
                   fill='tozeroy', name='流量',
                   line=dict(color='#1f77b4', width=2),
                   marker=dict(size=marker_sizes_flow, color=marker_colors_flow, 
                               line=dict(color='white', width=1))),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(x=hours, y=conc_values, mode='lines+markers',
                   fill='tozeroy', name='浓度系数',
                   line=dict(color='#2ca02c', width=2),
                   marker=dict(size=marker_sizes_conc, color=marker_colors_conc,
                               line=dict(color='white', width=1))),
        row=2, col=1,
    )
    
    if selected_hour is not None and 0 <= selected_hour < 24:
        fig.add_annotation(
            x=selected_hour, y=flow_values[selected_hour],
            text=f"<b>{selected_hour:02d}:00</b><br>{flow_values[selected_hour]:.0f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            bgcolor='rgba(255,75,75,0.9)', font=dict(color='white'),
            row=1, col=1
        )
        fig.add_annotation(
            x=selected_hour, y=conc_values[selected_hour],
            text=f"<b>{selected_hour:02d}:00</b><br>{conc_values[selected_hour]:.2f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            bgcolor='rgba(255,75,75,0.9)', font=dict(color='white'),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text='时间 (小时)', row=2, col=1, tickmode='array', 
                     tickvals=list(range(0, 24, 2)))
    fig.update_yaxes(title_text='流量 (m³/day)', row=1, col=1)
    fig.update_yaxes(title_text='浓度系数', row=2, col=1)
    
    fig.update_layout(
        height=580,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=10, r=10, t=40, b=10),
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


def plot_process_comparison(comparison_result: 'ProcessComparisonResult') -> go.Figure:
    """
    绘制两套工艺方案的出水水质对比柱状图
    
    参数:
        comparison_result: 工艺对比结果
    
    返回:
        Plotly Figure
    """
    from .analysis import ProcessComparisonResult
    
    indicators = ['COD', 'BOD5', 'NH3_N', 'TN', 'TP', 'SS']
    display_names = ['COD', 'BOD5', 'NH3-N', 'TN', 'TP', 'SS']
    
    scheme1_values = []
    scheme2_values = []
    
    for ind in indicators:
        val1 = comparison_result.result1.effluent_quality.get(ind, 0) if comparison_result.result1 and comparison_result.result1.converged else 0
        val2 = comparison_result.result2.effluent_quality.get(ind, 0) if comparison_result.result2 and comparison_result.result2.converged else 0
        scheme1_values.append(val1)
        scheme2_values.append(val2)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=comparison_result.name1,
        x=display_names,
        y=scheme1_values,
        marker_color='#1f77b4',
        text=[f'{v:.2f}' for v in scheme1_values],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name=comparison_result.name2,
        x=display_names,
        y=scheme2_values,
        marker_color='#ff7f0e',
        text=[f'{v:.2f}' for v in scheme2_values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='两套工艺方案出水水质对比',
        xaxis_title='水质指标',
        yaxis_title='浓度 (mg/L)',
        barmode='group',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        hovermode='x unified',
    )
    
    return fig


def plot_srt_vs_sludge(srt_values: List[float], 
                        sludge_values: List[float],
                        current_srt: Optional[float] = None,
                        current_sludge: Optional[float] = None,
                        converged_list: Optional[List[bool]] = None) -> go.Figure:
    """
    绘制SRT与污泥产量的关系曲线
    
    参数:
        srt_values: SRT取值列表
        sludge_values: 对应污泥产量列表
        current_srt: 当前运行工况的SRT
        current_sludge: 当前工况的污泥产量
        converged_list: 各点收敛情况
    
    返回:
        Plotly Figure
    """
    fig = go.Figure()
    
    valid_srt = []
    valid_sludge = []
    invalid_srt = []
    invalid_sludge = []
    
    if converged_list is not None:
        for srt, sludge, conv in zip(srt_values, sludge_values, converged_list):
            if conv and not np.isnan(sludge):
                valid_srt.append(srt)
                valid_sludge.append(sludge)
            else:
                invalid_srt.append(srt)
                invalid_sludge.append(sludge)
    else:
        for srt, sludge in zip(srt_values, sludge_values):
            if not np.isnan(sludge):
                valid_srt.append(srt)
                valid_sludge.append(sludge)
    
    fig.add_trace(go.Scatter(
        x=valid_srt, y=valid_sludge,
        mode='lines+markers',
        name='污泥产量',
        line=dict(color='#27ae60', width=3),
        marker=dict(size=10, color='#27ae60'),
        fill='tozeroy',
        fillcolor='rgba(39, 174, 96, 0.2)',
    ))
    
    if len(invalid_srt) > 0:
        fig.add_trace(go.Scatter(
            x=invalid_srt, y=invalid_sludge,
            mode='markers',
            name='未收敛',
            marker=dict(size=10, color='red', symbol='x'),
        ))
    
    if current_srt is not None and current_sludge is not None:
        fig.add_trace(go.Scatter(
            x=[current_srt], y=[current_sludge],
            mode='markers',
            name='当前工况',
            marker=dict(size=15, color='#e74c3c', symbol='star', line=dict(color='white', width=2)),
        ))
        
        fig.add_annotation(
            x=current_srt, y=current_sludge,
            text=f'当前: SRT={current_srt:.1f}天<br>产泥={current_sludge:.1f} kg/d',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            ax=-100, ay=-50,
            bgcolor='rgba(231, 76, 60, 0.9)',
            font=dict(color='white', size=12),
        )
    
    fig.update_layout(
        title='SRT与剩余污泥产量的关系',
        xaxis_title='污泥停留时间 SRT (天)',
        yaxis_title='剩余污泥产量 (kg DS/d)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        annotations=[
            dict(
                x=0.95, y=0.95,
                xref='paper', yref='paper',
                text='💡 SRT越长，污泥产量越低<br>但硝化效果越好，能耗越高',
                showarrow=False,
                align='right',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#cccccc',
                borderwidth=1,
                font=dict(size=11),
            )
        ]
    )
    
    return fig


def plot_energy_pie(energy_result: 'EnergyConsumptionResult') -> go.Figure:
    """
    绘制能耗分项饼图
    
    参数:
        energy_result: 能耗估算结果
    
    返回:
        Plotly Figure
    """
    from .analysis import EnergyConsumptionResult
    
    labels = ['曝气系统', '回流泵', '内回流泵', '搅拌系统', '其他']
    values = [
        energy_result.aeration_kwh_d,
        energy_result.return_pump_kwh_d,
        energy_result.internal_pump_kwh_d,
        energy_result.mixing_kwh_d,
        energy_result.other_kwh_d,
    ]
    
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#95a5a6']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=colors,
        textinfo='label+percent',
        texttemplate='%{label}<br>%{percent:.1%}<br>%{value:.1f} kWh/d',
        hovertemplate='%{label}<br>能耗: %{value:.1f} kWh/d<br>占比: %{percent:.1%}<extra></extra>',
    )])
    
    fig.update_layout(
        title='系统能耗分项占比',
        height=500,
        annotations=[
            dict(
                text=f'总能耗<br>{energy_result.total_kwh_d:.1f} kWh/d',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False,
            )
        ]
    )
    
    return fig


def plot_srt_vs_energy(srt_values: List[float],
                        energy_values: List[float],
                        current_srt: Optional[float] = None,
                        current_energy: Optional[float] = None,
                        converged_list: Optional[List[bool]] = None) -> go.Figure:
    """
    绘制SRT与能耗的关系曲线
    
    参数:
        srt_values: SRT取值列表
        energy_values: 对应能耗列表
        current_srt: 当前运行工况的SRT
        current_energy: 当前工况的能耗
        converged_list: 各点收敛情况
    
    返回:
        Plotly Figure
    """
    fig = go.Figure()
    
    valid_srt = []
    valid_energy = []
    
    if converged_list is not None:
        for srt, energy, conv in zip(srt_values, energy_values, converged_list):
            if conv and not np.isnan(energy):
                valid_srt.append(srt)
                valid_energy.append(energy)
    else:
        for srt, energy in zip(srt_values, energy_values):
            if not np.isnan(energy):
                valid_srt.append(srt)
                valid_energy.append(energy)
    
    fig.add_trace(go.Scatter(
        x=valid_srt, y=valid_energy,
        mode='lines+markers',
        name='能耗',
        line=dict(color='#e67e22', width=3),
        marker=dict(size=10, color='#e67e22'),
        fill='tozeroy',
        fillcolor='rgba(230, 126, 34, 0.2)',
    ))
    
    if current_srt is not None and current_energy is not None:
        fig.add_trace(go.Scatter(
            x=[current_srt], y=[current_energy],
            mode='markers',
            name='当前工况',
            marker=dict(size=15, color='#e74c3c', symbol='star', line=dict(color='white', width=2)),
        ))
        
        fig.add_annotation(
            x=current_srt, y=current_energy,
            text=f'当前: SRT={current_srt:.1f}天<br>能耗={current_energy:.1f} kWh/d',
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            ax=-100, ay=-50,
            bgcolor='rgba(231, 76, 60, 0.9)',
            font=dict(color='white', size=12),
        )
    
    fig.update_layout(
        title='SRT与系统能耗的关系',
        xaxis_title='污泥停留时间 SRT (天)',
        yaxis_title='系统能耗 (kWh/d)',
        height=500,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
    )
    
    return fig


def plot_pareto_front(pareto_front: List, color_by: str = 'sludge') -> go.Figure:
    """
    绘制Pareto前沿散点图
    
    参数:
        pareto_front: Pareto前沿个体列表
        color_by: 颜色映射的目标 ('sludge', 'NH3_N', 'energy', 'TN')
    
    返回:
        plotly Figure
    """
    from .nsga2_optimizer import Individual
    
    if not pareto_front or len(pareto_front) == 0:
        fig = go.Figure()
        fig.update_layout(
            title='Pareto前沿',
            xaxis_title='能耗 (kWh/d)',
            yaxis_title='出水TN (mg/L)',
            height=500,
        )
        return fig
    
    x_vals = []
    y_vals = []
    color_vals = []
    hover_texts = []
    
    color_label_map = {
        'sludge': '产泥量 (kg DS/d)',
        'NH3_N': '出水NH3-N (mg/L)',
        'energy': '能耗 (kWh/d)',
        'TN': '出水TN (mg/L)',
    }
    color_label = color_label_map.get(color_by, '产泥量 (kg DS/d)')
    
    for ind in pareto_front:
        if not ind.converged:
            continue
        
        energy = ind.energy_result.total_kwh_d if ind.energy_result else 0
        tn = ind.effluent_quality.get('TN', 0)
        
        if color_by == 'sludge':
            color_val = ind.sludge_result.daily_sludge_kg if ind.sludge_result else 0
        elif color_by == 'NH3_N':
            color_val = ind.effluent_quality.get('NH3_N', 0)
        elif color_by == 'energy':
            color_val = energy
        else:
            color_val = tn
        
        x_vals.append(energy)
        y_vals.append(tn)
        color_vals.append(color_val)
        
        do = ind.variables[0] if len(ind.variables) > 0 else 0
        irr = ind.variables[1] if len(ind.variables) > 1 else 0
        srt = ind.variables[2] if len(ind.variables) > 2 else 0
        rr = ind.variables[3] if len(ind.variables) > 3 else 0
        
        nh3 = ind.effluent_quality.get('NH3_N', 0)
        cod = ind.effluent_quality.get('COD', 0)
        tp = ind.effluent_quality.get('TP', 0)
        sludge = ind.sludge_result.daily_sludge_kg if ind.sludge_result else 0
        
        compliant = "是" if ind.is_feasible else "否"
        
        hover_text = (
            f"<b>能耗:</b> {energy:.1f} kWh/d<br>"
            f"<b>出水TN:</b> {tn:.2f} mg/L<br>"
            f"<b>出水NH3-N:</b> {nh3:.2f} mg/L<br>"
            f"<b>出水COD:</b> {cod:.2f} mg/L<br>"
            f"<b>出水TP:</b> {tp:.2f} mg/L<br>"
            f"<b>产泥量:</b> {sludge:.1f} kg DS/d<br>"
            f"<b>DO设定:</b> {do:.2f} mg/L<br>"
            f"<b>内回流比:</b> {irr:.0f}%<br>"
            f"<b>SRT:</b> {srt:.1f} 天<br>"
            f"<b>回流比:</b> {rr:.0f}%<br>"
            f"<b>达标:</b> {compliant}"
        )
        hover_texts.append(hover_text)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=12,
                color=color_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=color_label),
                line=dict(width=1, color='DarkSlateGrey'),
            ),
            text=hover_texts,
            hovertemplate='%{text}<extra></extra>',
            name='Pareto最优解',
        )
    )
    
    fig.update_layout(
        title='Pareto前沿 - 能耗 vs 出水TN',
        xaxis_title='能耗 (kWh/d)',
        yaxis_title='出水TN (mg/L)',
        height=550,
        hovermode='closest',
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    return fig


def plot_convergence_curve(
    best_fitness_history: List[float], 
    avg_fitness_history: List[float]
) -> go.Figure:
    """
    绘制收敛曲线图
    
    参数:
        best_fitness_history: 每代最优适应度
        avg_fitness_history: 每代平均适应度
    
    返回:
        plotly Figure
    """
    generations = list(range(len(best_fitness_history)))
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=best_fitness_history,
            mode='lines+markers',
            name='最优适应度',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=generations,
            y=avg_fitness_history,
            mode='lines+markers',
            name='平均适应度',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6),
        )
    )
    
    fig.update_layout(
        title='算法收敛曲线',
        xaxis_title='迭代代数',
        yaxis_title='适应度值',
        height=400,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2),
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    return fig


def plot_objective_parallel_coordinates(pareto_front: List) -> go.Figure:
    """
    绘制平行坐标图展示多目标关系
    
    参数:
        pareto_front: Pareto前沿个体列表
    
    返回:
        plotly Figure
    """
    from .nsga2_optimizer import Individual
    
    if not pareto_front or len(pareto_front) == 0:
        fig = go.Figure()
        return fig
    
    nh3_vals = []
    tn_vals = []
    energy_vals = []
    sludge_vals = []
    
    for ind in pareto_front:
        if not ind.converged:
            continue
        
        nh3_vals.append(ind.effluent_quality.get('NH3_N', 0))
        tn_vals.append(ind.effluent_quality.get('TN', 0))
        energy_vals.append(ind.energy_result.total_kwh_d if ind.energy_result else 0)
        sludge_vals.append(ind.sludge_result.daily_sludge_kg if ind.sludge_result else 0)
    
    if len(tn_vals) == 0:
        fig = go.Figure()
        return fig
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=tn_vals,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='TN (mg/L)')
            ),
            dimensions=list([
                dict(range=[min(nh3_vals), max(nh3_vals)],
                     label='NH3-N (mg/L)', values=nh3_vals),
                dict(range=[min(tn_vals), max(tn_vals)],
                     label='TN (mg/L)', values=tn_vals),
                dict(range=[min(energy_vals), max(energy_vals)],
                     label='能耗 (kWh/d)', values=energy_vals),
                dict(range=[min(sludge_vals), max(sludge_vals)],
                     label='产泥 (kg/d)', values=sludge_vals),
            ])
        )
    )
    
    fig.update_layout(
        title='多目标平行坐标图',
        height=450,
        margin=dict(l=80, r=80, t=80, b=60),
    )
    
    return fig
