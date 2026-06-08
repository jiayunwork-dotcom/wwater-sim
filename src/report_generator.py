"""
报告生成模块
支持HTML和PDF格式的仿真报告导出
"""

import numpy as np
import pandas as pd
import json
import base64
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import io
import plotly.graph_objects as go
from dataclasses import dataclass

from .asm1_model import (
    ASM1Parameters,
    COMPONENT_NAMES,
    COMPONENT_DESCRIPTIONS,
    aggregate_to_wq_indices,
)
from .reactor_units import ProcessFlowSheet, ReactorType
from .process_templates import InfluentConfig
from .solver import SteadyStateResult
from .analysis import (
    ComplianceResult,
    OptimizationSuggestion,
    SludgeProductionResult,
    EnergyConsumptionResult,
)
from .visualization import (
    plot_reactor_stack,
    plot_process_diagram,
    plot_compliance_radar,
    plot_energy_pie,
)


@dataclass
class ReportData:
    """报告数据集合"""
    pfs: ProcessFlowSheet
    influent: InfluentConfig
    asm1_params: ASM1Parameters
    steady_result: SteadyStateResult
    compliance_result: Optional[ComplianceResult] = None
    optimization_suggestions: Optional[List[OptimizationSuggestion]] = None
    sludge_result: Optional[SludgeProductionResult] = None
    energy_result: Optional[EnergyConsumptionResult] = None
    process_name: str = "A2O工艺"
    standard_name: str = "一级A"


def fig_to_base64(fig: go.Figure) -> str:
    """将Plotly图表转换为base64编码的PNG图片"""
    try:
        img_bytes = fig.to_image(format='png', width=1000, height=500, scale=2)
        return base64.b64encode(img_bytes).decode('utf-8')
    except:
        return ""


def generate_timestamp_filename(prefix: str = "simulation_report", extension: str = "html") -> str:
    """生成带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def _get_process_summary_html(pfs: ProcessFlowSheet, process_name: str) -> str:
    """生成工艺流程概要HTML"""
    reactor_rows = ""
    for i, reactor in enumerate(pfs.reactors):
        icon = reactor.get_icon()
        name = reactor.name
        type_name = reactor.get_type_name()
        volume = reactor.geometry.volume
        hrt = reactor.operation.HRT
        srt = reactor.operation.SRT
        do = reactor.operation.DO_setpoint if hasattr(reactor.operation, 'DO_setpoint') else 0
        rr = reactor.operation.return_sludge_ratio * 100
        irr = reactor.operation.internal_return_ratio * 100 if hasattr(reactor.operation, 'internal_return_ratio') else 0
        
        extra_info = ""
        if reactor.reactor_type == ReactorType.AEROBIC:
            extra_info = f"DO: {do:.1f} mg/L"
        elif reactor.reactor_type == ReactorType.ANOXIC or reactor.reactor_type == ReactorType.ANAEROBIC:
            extra_info = "缺氧/厌氧条件"
        if irr > 0:
            extra_info += f" | 内回流: {irr:.0f}%"
        
        reactor_rows += f"""
        <tr>
            <td style="padding: 8px; text-align: center; font-size: 20px;">{icon}</td>
            <td style="padding: 8px;"><strong>{name}</strong></td>
            <td style="padding: 8px;">{type_name}</td>
            <td style="padding: 8px; text-align: right;">{volume:.0f} m³</td>
            <td style="padding: 8px; text-align: right;">{hrt:.1f} h</td>
            <td style="padding: 8px; text-align: right;">{srt:.1f} 天</td>
            <td style="padding: 8px; text-align: right;">{rr:.0f}%</td>
            <td style="padding: 8px;">{extra_info}</td>
        </tr>
        """
    
    total_volume = pfs.get_volume()
    
    return f"""
    <div class="section">
        <h2>1. 工艺流程配置概要</h2>
        <p><strong>工艺类型:</strong> {process_name}</p>
        <p><strong>总容积:</strong> {total_volume:.0f} m³</p>
        <p><strong>处理单元数:</strong> {len(pfs.reactors)} 个</p>
        <table border="1" style="border-collapse: collapse; width: 100%; margin-top: 10px;">
            <thead>
                <tr style="background-color: #1f77b4; color: white;">
                    <th style="padding: 10px; width: 60px;">图标</th>
                    <th style="padding: 10px;">单元名称</th>
                    <th style="padding: 10px;">单元类型</th>
                    <th style="padding: 10px;">容积</th>
                    <th style="padding: 10px;">HRT</th>
                    <th style="padding: 10px;">SRT</th>
                    <th style="padding: 10px;">回流比</th>
                    <th style="padding: 10px;">备注</th>
                </tr>
            </thead>
            <tbody>
                {reactor_rows}
            </tbody>
        </table>
    </div>
    """


def _get_influent_html(influent: InfluentConfig) -> str:
    """生成进水水质参数HTML"""
    Q = influent.Q_base
    C = influent.get_C(0)
    wq = aggregate_to_wq_indices(C)
    
    quality_mode = "典型进水" if influent.quality_mode == 'typical' else "自定义进水"
    flow_mode = "恒定流量" if influent.flow_mode == 'constant' else "日变化流量"
    
    return f"""
    <div class="section">
        <h2>2. 进水水质参数</h2>
        <div style="display: flex; gap: 40px; margin-bottom: 15px;">
            <p><strong>流量模式:</strong> {flow_mode}</p>
            <p><strong>水质模式:</strong> {quality_mode}</p>
            <p><strong>日均流量:</strong> {Q:.0f} m³/day</p>
        </div>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background-color: #2ca02c; color: white;">
                    <th style="padding: 10px;">指标</th>
                    <th style="padding: 10px;">COD</th>
                    <th style="padding: 10px;">BOD5</th>
                    <th style="padding: 10px;">NH3-N</th>
                    <th style="padding: 10px;">TN</th>
                    <th style="padding: 10px;">TP</th>
                    <th style="padding: 10px;">SS</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; text-align: center;"><strong>浓度 (mg/L)</strong></td>
                    <td style="padding: 10px; text-align: right;">{wq['COD']:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['BOD5']:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['NH3_N']:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['TN']:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['TP']:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['SS']:.1f}</td>
                </tr>
                <tr style="background-color: #f5f5f5;">
                    <td style="padding: 10px; text-align: center;"><strong>日负荷 (kg/day)</strong></td>
                    <td style="padding: 10px; text-align: right;">{wq['COD'] * Q / 1000:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['BOD5'] * Q / 1000:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['NH3_N'] * Q / 1000:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['TN'] * Q / 1000:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['TP'] * Q / 1000:.1f}</td>
                    <td style="padding: 10px; text-align: right;">{wq['SS'] * Q / 1000:.1f}</td>
                </tr>
            </tbody>
        </table>
    </div>
    """


def _get_asm1_params_html(params: ASM1Parameters) -> str:
    """生成ASM1模型参数表HTML"""
    temp_corrected = params.get_temperature_corrected_params(params.temperature)
    
    stoich_params = [
        ('Y_H', '异养菌产率系数', 'mg COD/mg COD'),
        ('Y_A', '自养菌产率系数', 'mg COD/mg N'),
        ('f_P', '惰性颗粒产物比例', '无量纲'),
        ('i_XB', '菌体氮含量', 'mg N/mg COD'),
        ('i_XP', '惰性产物氮含量', 'mg N/mg COD'),
    ]
    
    kinetic_params = [
        ('mu_H', '异养菌最大比生长速率', '1/day'),
        ('K_S', '异养菌半饱和系数', 'mg COD/L'),
        ('K_O_H', '异养菌氧半饱和系数', 'mg O2/L'),
        ('K_NO', '硝酸盐半饱和系数', 'mg N/L'),
        ('b_H', '异养菌衰减系数', '1/day'),
        ('eta_g', '缺氧生长修正系数', '无量纲'),
        ('k_h', '颗粒有机物水解速率', '1/day'),
        ('mu_A', '自养菌最大比生长速率', '1/day'),
        ('K_O_A', '自养菌氧半饱和系数', 'mg O2/L'),
        ('K_NH', '氨氮半饱和系数', 'mg N/L'),
        ('b_A', '自养菌衰减系数', '1/day'),
        ('k_a', '氨化速率', 'L/(mg COD·day)'),
    ]
    
    stoich_rows = ""
    for key, desc, unit in stoich_params:
        val = getattr(params, key)
        stoich_rows += f"""
        <tr>
            <td style="padding: 8px;"><strong>{key}</strong></td>
            <td style="padding: 8px;">{desc}</td>
            <td style="padding: 8px;">{unit}</td>
            <td style="padding: 8px; text-align: right;">{val:.4f}</td>
        </tr>
        """
    
    kinetic_rows = ""
    for key, desc, unit in kinetic_params:
        default_val = getattr(params, key)
        corrected_val = getattr(temp_corrected, key)
        kinetic_rows += f"""
        <tr>
            <td style="padding: 8px;"><strong>{key}</strong></td>
            <td style="padding: 8px;">{desc}</td>
            <td style="padding: 8px;">{unit}</td>
            <td style="padding: 8px; text-align: right;">{default_val:.4f}</td>
            <td style="padding: 8px; text-align: right; background-color: #fff3cd;">{corrected_val:.4f}</td>
        </tr>
        """
    
    return f"""
    <div class="section">
        <h2>3. ASM1模型参数表</h2>
        <p><strong>水温:</strong> {params.temperature:.1f} °C (温度修正已自动应用)</p>
        
        <h3 style="color: #1f77b4;">3.1 化学计量系数</h3>
        <table border="1" style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">
            <thead>
                <tr style="background-color: #1f77b4; color: white;">
                    <th style="padding: 8px;">参数符号</th>
                    <th style="padding: 8px;">描述</th>
                    <th style="padding: 8px;">单位</th>
                    <th style="padding: 8px;">20°C值</th>
                </tr>
            </thead>
            <tbody>
                {stoich_rows}
            </tbody>
        </table>
        
        <h3 style="color: #2ca02c;">3.2 动力学参数</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background-color: #2ca02c; color: white;">
                    <th style="padding: 8px;">参数符号</th>
                    <th style="padding: 8px;">描述</th>
                    <th style="padding: 8px;">单位</th>
                    <th style="padding: 8px;">20°C值</th>
                    <th style="padding: 8px;">{params.temperature:.0f}°C修正值</th>
                </tr>
            </thead>
            <tbody>
                {kinetic_rows}
            </tbody>
        </table>
    </div>
    """


def _get_steady_result_html(
    steady_result: SteadyStateResult,
    compliance_result: Optional[ComplianceResult],
    standard_name: str,
) -> str:
    """生成稳态求解结果HTML"""
    effluent = steady_result.effluent_quality
    converged = steady_result.converged
    iterations = steady_result.iterations
    residual = steady_result.final_residual
    
    status_style = "color: green;" if converged else "color: red;"
    status_text = "✅ 收敛" if converged else "❌ 未收敛"
    
    effluent_rows = ""
    compliance_rows = ""
    
    indicators = [('COD', 'COD'), ('BOD5', 'BOD5'), ('NH3_N', 'NH3-N'), 
                  ('TN', 'TN'), ('TP', 'TP'), ('SS', 'SS')]
    
    for ind_key, ind_name in indicators:
        val = effluent.get(ind_key, 0)
        limit = None
        compliant = True
        ratio = 0
        suggestion = ""
        
        if compliance_result:
            item = compliance_result[ind_name]
            if item:
                limit = item.limit
                compliant = item.compliant
                ratio = item.ratio * 100
                suggestion = item.suggestion
        
        status_cell = "✅" if compliant else "❌"
        row_style = "" if compliant else "background-color: #ffe6e6;"
        
        effluent_rows += f"""
        <tr style="{row_style}">
            <td style="padding: 8px;"><strong>{ind_name}</strong></td>
            <td style="padding: 8px; text-align: right;">{val:.2f}</td>
            <td style="padding: 8px; text-align: right;">{limit if limit else '-'}</td>
            <td style="padding: 8px; text-align: right;">{ratio:.1f}%</td>
            <td style="padding: 8px; text-align: center;">{status_cell}</td>
            <td style="padding: 8px;">{suggestion}</td>
        </tr>
        """
    
    overall_status = ""
    if compliance_result:
        if compliance_result.overall_compliant:
            overall_status = '<p style="color: green; font-size: 18px;"><strong>🎉 出水水质全面达标！</strong></p>'
        else:
            overall_status = '<p style="color: red; font-size: 18px;"><strong>⚠️ 存在超标指标</strong></p>'
    
    return f"""
    <div class="section">
        <h2>4. 稳态求解结果</h2>
        <div style="display: flex; gap: 40px; margin-bottom: 15px;">
            <p><strong>求解状态:</strong> <span style="{status_style}">{status_text}</span></p>
            <p><strong>迭代次数:</strong> {iterations}</p>
            <p><strong>最终残差:</strong> {residual:.2e}</p>
            <p><strong>执行标准:</strong> {standard_name}</p>
        </div>
        
        {overall_status}
        
        <h3 style="color: #d62728;">4.1 出水水质与达标判定</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                <tr style="background-color: #d62728; color: white;">
                    <th style="padding: 8px;">指标</th>
                    <th style="padding: 8px;">出水浓度 (mg/L)</th>
                    <th style="padding: 8px;">标准限值 (mg/L)</th>
                    <th style="padding: 8px;">占标率</th>
                    <th style="padding: 8px;">达标情况</th>
                    <th style="padding: 8px;">建议</th>
                </tr>
            </thead>
            <tbody>
                {effluent_rows}
            </tbody>
        </table>
    </div>
    """


def _get_reactor_concentration_html(
    pfs: ProcessFlowSheet,
    reactor_states: List[np.ndarray],
) -> str:
    """生成各池浓度分布HTML"""
    indicator_keys = ['COD', 'BOD5', 'NH3_N', 'TN', 'TP', 'SS']
    indicator_names = ['COD', 'BOD5', 'NH3-N', 'TN', 'TP', 'SS']
    
    reactor_data = []
    for reactor, state in zip(pfs.reactors, reactor_states):
        wq = aggregate_to_wq_indices(state)
        row = {'单元': reactor.name}
        for key, name in zip(indicator_keys, indicator_names):
            row[name] = wq.get(key, 0)
        reactor_data.append(row)
    
    df = pd.DataFrame(reactor_data)
    
    header_row = "<tr style='background-color: #9467bd; color: white;'>"
    header_row += "<th style='padding: 8px;'>处理单元</th>"
    for name in indicator_names:
        header_row += f"<th style='padding: 8px;'>{name}</th>"
    header_row += "</tr>"
    
    data_rows = ""
    for _, row in df.iterrows():
        data_rows += "<tr>"
        data_rows += f"<td style='padding: 8px;'><strong>{row['单元']}</strong></td>"
        for name in indicator_names:
            data_rows += f"<td style='padding: 8px; text-align: right;'>{row[name]:.2f}</td>"
        data_rows += "</tr>"
    
    return f"""
    <div class="section">
        <h3 style="color: #9467bd;">4.2 各池浓度分布</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <thead>
                {header_row}
            </thead>
            <tbody>
                {data_rows}
            </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 5px;">单位: mg/L</p>
    </div>
    """


def _get_sludge_energy_html(
    sludge_result: Optional[SludgeProductionResult],
    energy_result: Optional[EnergyConsumptionResult],
) -> str:
    """生成污泥产量和能耗分析HTML"""
    sludge_html = ""
    if sludge_result:
        sludge_html = f"""
        <h3 style="color: #27ae60;">4.3 污泥产量分析</h3>
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div class="metric-box">
                <div class="metric-value">{sludge_result.daily_sludge_kg:.1f}</div>
                <div class="metric-label">日剩余污泥产量 (kg DS/d)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{sludge_result.MLSS_gL:.2f}</div>
                <div class="metric-label">平均MLSS (g/L)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{sludge_result.total_biomass_kg:.0f}</div>
                <div class="metric-label">系统总生物量 (kg)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{sludge_result.XBH_kg:.0f} / {sludge_result.XBA_kg:.0f}</div>
                <div class="metric-label">异养菌/自养菌 (kg)</div>
            </div>
        </div>
        """
    
    energy_html = ""
    if energy_result:
        energy_html = f"""
        <h3 style="color: #e67e22;">4.4 能耗分析</h3>
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div class="metric-box" style="background-color: #fff3e0;">
                <div class="metric-value" style="color: #e67e22;">{energy_result.total_kwh_d:.1f}</div>
                <div class="metric-label">日均总电耗 (kWh/d)</div>
            </div>
            <div class="metric-box" style="background-color: #fff3e0;">
                <div class="metric-value" style="color: #e67e22;">{energy_result.unit_kwh_m3:.4f}</div>
                <div class="metric-label">单位水量电耗 (kWh/m³)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{energy_result.aeration_kwh_d:.1f}</div>
                <div class="metric-label">曝气系统 (kWh/d)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{energy_result.return_pump_kwh_d + energy_result.internal_pump_kwh_d:.1f}</div>
                <div class="metric-label">泵类能耗 (kWh/d)</div>
            </div>
        </div>
        """
    
    return sludge_html + energy_html


def _get_optimization_html(
    suggestions: Optional[List[OptimizationSuggestion]],
) -> str:
    """生成优化建议HTML"""
    if not suggestions:
        return """
        <div class="section">
            <h2>5. 优化建议</h2>
            <p>当前工艺运行良好，暂无优化建议。</p>
        </div>
        """
    
    priority_labels = {1: '高', 2: '中', 3: '低', 4: '建议'}
    priority_colors = {1: '#e74c3c', 2: '#e67e22', 3: '#f39c12', 4: '#3498db'}
    
    suggestion_rows = ""
    for i, s in enumerate(suggestions):
        priority = priority_labels.get(s.priority, '中')
        color = priority_colors.get(s.priority, '#3498db')
        
        improvement_text = ""
        if s.expected_improvement:
            improvements = []
            for key, val in s.expected_improvement.items():
                name_map = {'COD': 'COD', 'NH3_N': 'NH3-N', 'TN': 'TN', 'TP': 'TP', 'SS': 'SS'}
                improvements.append(f"{name_map.get(key, key)}: ↓{val:.1f} mg/L")
            improvement_text = " | ".join(improvements)
        
        suggestion_rows += f"""
        <div class="suggestion-card">
            <div class="suggestion-header">
                <span class="priority-badge" style="background-color: {color};">{priority}优先级</span>
                <strong style="font-size: 16px;">{s.title}</strong>
            </div>
            <p style="margin: 8px 0;">{s.description}</p>
            <div style="display: flex; gap: 30px; margin-top: 10px;">
                <p><strong>当前值:</strong> {s.current_value}</p>
                <p><strong>建议值:</strong> {s.suggested_value}</p>
            </div>
            <p style="color: #27ae60;"><strong>预期效果:</strong> {s.expected_effect}</p>
            {f'<p style="color: #2980b9;"><strong>预期改善:</strong> {improvement_text}</p>' if improvement_text else ''}
        </div>
        """
    
    return f"""
    <div class="section">
        <h2>5. 优化建议</h2>
        <p>基于当前仿真结果，系统生成以下优化建议（按优先级排序）：</p>
        {suggestion_rows}
    </div>
    """


def generate_html_report(
    report_data: ReportData,
    include_charts: bool = True,
) -> Tuple[str, str]:
    """
    生成HTML格式的仿真报告
    
    参数:
        report_data: 报告数据集合
        include_charts: 是否包含图表
    
    返回:
        (html_content, filename)
    """
    process_diagram_img = ""
    reactor_stack_img = ""
    compliance_radar_img = ""
    energy_pie_img = ""
    
    if include_charts:
        try:
            fig_diagram = plot_process_diagram(report_data.pfs)
            process_diagram_img = fig_to_base64(fig_diagram)
        except:
            pass
        
        try:
            fig_stack = plot_reactor_stack(report_data.pfs, report_data.steady_result.reactor_states)
            reactor_stack_img = fig_to_base64(fig_stack)
        except:
            pass
        
        try:
            if report_data.compliance_result:
                fig_radar = plot_compliance_radar(report_data.compliance_result)
                compliance_radar_img = fig_to_base64(fig_radar)
        except:
            pass
        
        try:
            if report_data.energy_result:
                fig_energy = plot_energy_pie(report_data.energy_result)
                energy_pie_img = fig_to_base64(fig_energy)
        except:
            pass
    
    now = datetime.now()
    report_time = now.strftime("%Y年%m月%d日 %H:%M:%S")
    
    css_styles = """
    <style>
        body {
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
            color: #333;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1f77b4, #2ca02c);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            font-size: 28px;
        }
        .header p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        .section {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .section h2 {
            color: #1f77b4;
            border-bottom: 3px solid #1f77b4;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .section h3 {
            color: #555;
            margin-top: 20px;
        }
        table {
            margin-top: 10px;
        }
        th {
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .chart-container {
            text-align: center;
            margin: 20px 0;
        }
        .chart-container img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .metric-box {
            background-color: #f0f8ff;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
            min-width: 150px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #1f77b4;
        }
        .metric-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .suggestion-card {
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 0 5px 5px 0;
        }
        .suggestion-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 5px;
        }
        .priority-badge {
            color: white;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #ddd;
            margin-top: 30px;
        }
    </style>
    """
    
    diagram_html = ""
    if process_diagram_img:
        diagram_html = f"""
        <div class="chart-container">
            <h3>工艺流程示意图</h3>
            <img src="data:image/png;base64,{process_diagram_img}" alt="工艺流程示意图">
        </div>
        """
    
    stack_html = ""
    if reactor_stack_img:
        stack_html = f"""
        <div class="chart-container">
            <h3>各池组分浓度分布图</h3>
            <img src="data:image/png;base64,{reactor_stack_img}" alt="各池浓度分布图">
        </div>
        """
    
    radar_html = ""
    if compliance_radar_img:
        radar_html = f"""
        <div class="chart-container">
            <h3>出水达标雷达图</h3>
            <img src="data:image/png;base64,{compliance_radar_img}" alt="达标雷达图">
        </div>
        """
    
    energy_html = ""
    if energy_pie_img:
        energy_html = f"""
        <div class="chart-container">
            <h3>能耗分项饼图</h3>
            <img src="data:image/png;base64,{energy_pie_img}" alt="能耗饼图">
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>污水处理工艺仿真报告</title>
        {css_styles}
    </head>
    <body>
        <div class="header">
            <h1>💧 污水处理工艺仿真报告</h1>
            <p>基于ASM1活性污泥模型的仿真分析报告</p>
            <p style="margin-top: 10px;">生成时间: {report_time}</p>
        </div>
        
        {_get_process_summary_html(report_data.pfs, report_data.process_name)}
        {diagram_html}
        
        {_get_influent_html(report_data.influent)}
        
        {_get_asm1_params_html(report_data.asm1_params)}
        
        {_get_steady_result_html(
            report_data.steady_result,
            report_data.compliance_result,
            report_data.standard_name,
        )}
        
        {_get_reactor_concentration_html(
            report_data.pfs,
            report_data.steady_result.reactor_states,
        )}
        
        {stack_html}
        {radar_html}
        
        {_get_sludge_energy_html(
            report_data.sludge_result,
            report_data.energy_result,
        )}
        
        {energy_html}
        
        {_get_optimization_html(report_data.optimization_suggestions)}
        
        <div class="footer">
            <p>本报告由污水处理工艺仿真系统自动生成</p>
            <p>基于IWA ASM1活性污泥模型 | © 2024</p>
        </div>
    </body>
    </html>
    """
    
    filename = generate_timestamp_filename("simulation_report", "html")
    
    return html_content, filename


def generate_pdf_report(
    report_data: ReportData,
    include_charts: bool = True,
) -> Tuple[bytes, str]:
    """
    生成PDF格式的仿真报告（通过HTML转换）
    
    参数:
        report_data: 报告数据集合
        include_charts: 是否包含图表
    
    返回:
        (pdf_bytes, filename)
    """
    html_content, _ = generate_html_report(report_data, include_charts)
    
    try:
        import weasyprint
        pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
        filename = generate_timestamp_filename("simulation_report", "pdf")
        return pdf_bytes, filename
    except ImportError:
        raise ImportError("需要安装weasyprint库以生成PDF报告: pip install weasyprint")
    except Exception as e:
        raise RuntimeError(f"PDF生成失败: {str(e)}")


def save_report_to_file(content: str, filepath: str) -> None:
    """保存报告到文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def save_pdf_to_file(pdf_bytes: bytes, filepath: str) -> None:
    """保存PDF到文件"""
    with open(filepath, 'wb') as f:
        f.write(pdf_bytes)


def get_download_link_html(content: str, filename: str, filetype: str = 'html') -> str:
    """生成HTML下载链接（用于Streamlit）"""
    if filetype == 'html':
        b64 = base64.b64encode(content.encode('utf-8')).decode()
        href = f'data:text/html;charset=utf-8;base64,{b64}'
    elif filetype == 'pdf':
        b64 = base64.b64encode(content).decode() if isinstance(content, bytes) else base64.b64encode(content.encode()).decode()
        href = f'data:application/pdf;base64,{b64}'
    else:
        raise ValueError(f"不支持的文件类型: {filetype}")
    
    return f'<a href="{href}" download="{filename}">📥 下载{filetype.upper()}报告</a>'
