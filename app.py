"""
污水处理工艺仿真与出水水质预测系统
基于ASM1活性污泥模型的Streamlit应用
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import io
import base64
from typing import Dict, List, Optional
import copy

from src.asm1_model import (
    ASM1Parameters,
    NUM_COMPONENTS,
    COMPONENT_NAMES,
    COMPONENT_DESCRIPTIONS,
    COMPONENT_UNITS,
    get_stoichiometric_matrix,
    aggregate_to_wq_indices,
    get_typical_influent,
    create_influent_from_quality,
)
from src.reactor_units import (
    ReactorType,
    ReactorGeometry,
    ReactorOperation,
    ProcessFlowSheet,
    create_reactor_by_type,
    REACTOR_TYPE_NAMES,
    REACTOR_TYPE_ICONS,
)
from src.process_templates import (
    PROCESS_TEMPLATES,
    get_template_names,
    create_process_by_name,
    InfluentConfig,
)
from src.solver import (
    SolverConfig,
    SteadyStateResult,
    DynamicResult,
    DynamicSimulator,
    solve_steady_state,
    run_dynamic_simulation,
)
from src.analysis import (
    STANDARDS,
    STANDARD_NAMES,
    check_compliance,
    ComplianceResult,
    SENSITIVITY_PARAMETERS,
    run_sensitivity_analysis,
    run_two_factor_sensitivity,
    generate_optimization_suggestions,
    OptimizationSuggestion,
)
from src.visualization import (
    plot_reactor_stack,
    plot_process_diagram,
    plot_effluent_timeseries,
    plot_influent_diurnal,
    plot_compliance_radar,
    plot_sensitivity_curves,
    plot_two_factor_heatmap,
    plot_residual_convergence,
    plot_process_comparison,
    plot_srt_vs_sludge,
    plot_srt_vs_energy,
    plot_energy_pie,
    plot_pareto_front,
    plot_convergence_curve,
    plot_objective_parallel_coordinates,
)
from src.nsga2_optimizer import (
    NSGA2Optimizer,
    OptimizationConfig,
    OptimizationVariable,
    OptimizationObjective,
    Individual,
    OptimizationResult,
    DEFAULT_VARIABLES,
    DEFAULT_OBJECTIVES,
    get_default_config,
    calculate_composite_score,
)
from src.analysis import (
    calculate_sludge_production,
    calculate_energy_consumption,
    generate_srt_vs_sludge_curve,
    SludgeProductionResult,
    EnergyConsumptionResult,
    ProcessComparisonResult,
)
from src.report_generator import (
    ReportData,
    generate_html_report,
    generate_pdf_report,
    get_download_link_html,
    generate_timestamp_filename,
)


st.set_page_config(
    page_title="污水处理工艺仿真系统",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """初始化会话状态"""
    if 'pfs' not in st.session_state:
        st.session_state.pfs = create_process_by_name('A2O')
    
    if 'influent' not in st.session_state:
        st.session_state.influent = InfluentConfig()
        st.session_state.influent.set_diurnal_pattern('morning_evening_peak')
    
    if 'asm1_params' not in st.session_state:
        st.session_state.asm1_params = ASM1Parameters()
    
    if 'solver_config' not in st.session_state:
        st.session_state.solver_config = SolverConfig()
    
    if 'steady_result' not in st.session_state:
        st.session_state.steady_result = None
    
    if 'dynamic_result' not in st.session_state:
        st.session_state.dynamic_result = None
    
    if 'compliance_result' not in st.session_state:
        st.session_state.compliance_result = None
    
    if 'selected_standard' not in st.session_state:
        st.session_state.selected_standard = '一级A'
    
    if 'sensitivity_result' not in st.session_state:
        st.session_state.sensitivity_result = None
    
    if 'selected_hour' not in st.session_state:
        st.session_state.selected_hour = 8
    
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    
    if 'simulation_paused' not in st.session_state:
        st.session_state.simulation_paused = False
    
    if 'simulation_aborted' not in st.session_state:
        st.session_state.simulation_aborted = False
    
    if 'simulation_progress' not in st.session_state:
        st.session_state.simulation_progress = 0.0
    
    if 'two_factor_result' not in st.session_state:
        st.session_state.two_factor_result = None
    
    if 'optimization_suggestions' not in st.session_state:
        st.session_state.optimization_suggestions = None
    
    if 'current_page' not in st.session_state:
        st.session_state.current_page = '首页'
    
    if 'custom_reactors' not in st.session_state:
        st.session_state.custom_reactors = []
    
    if 'diurnal_curve' not in st.session_state:
        st.session_state.diurnal_curve = np.ones(24)
    
    if 'diurnal_flow_curve' not in st.session_state:
        st.session_state.diurnal_flow_curve = np.ones(24)
    
    if 'sludge_result' not in st.session_state:
        st.session_state.sludge_result = None
    
    if 'energy_result' not in st.session_state:
        st.session_state.energy_result = None
    
    if 'srt_sludge_curve' not in st.session_state:
        st.session_state.srt_sludge_curve = None
    
    if 'comparison_result' not in st.session_state:
        st.session_state.comparison_result = None
    
    if 'comparison_pfs1' not in st.session_state:
        st.session_state.comparison_pfs1 = None
    
    if 'comparison_pfs2' not in st.session_state:
        st.session_state.comparison_pfs2 = None
    
    if 'comparison_influent1' not in st.session_state:
        st.session_state.comparison_influent1 = None
    
    if 'comparison_influent2' not in st.session_state:
        st.session_state.comparison_influent2 = None
    
    if 'comparison_params1' not in st.session_state:
        st.session_state.comparison_params1 = None
    
    if 'comparison_params2' not in st.session_state:
        st.session_state.comparison_params2 = None
    
    if 'comparison_name1' not in st.session_state:
        st.session_state.comparison_name1 = "方案1"
    
    if 'comparison_name2' not in st.session_state:
        st.session_state.comparison_name2 = "方案2"
    
    if 'optimization_config' not in st.session_state:
        st.session_state.optimization_config = get_default_config()
    
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    
    if 'optimization_running' not in st.session_state:
        st.session_state.optimization_running = False
    
    if 'optimization_aborted' not in st.session_state:
        st.session_state.optimization_aborted = False
    
    if 'optimization_progress' not in st.session_state:
        st.session_state.optimization_progress = 0.0
    
    if 'optimization_optimizer' not in st.session_state:
        st.session_state.optimization_optimizer = None
    
    if 'optimization_selected_solution' not in st.session_state:
        st.session_state.optimization_selected_solution = None
    
    if 'optimization_color_by' not in st.session_state:
        st.session_state.optimization_color_by = 'sludge'
    
    if 'optimization_current_gen' not in st.session_state:
        st.session_state.optimization_current_gen = 0
    
    if 'optimization_max_gen' not in st.session_state:
        st.session_state.optimization_max_gen = 100
    
    if 'optimization_best_fitness' not in st.session_state:
        st.session_state.optimization_best_fitness = 0.0
    
    if 'optimization_avg_fitness' not in st.session_state:
        st.session_state.optimization_avg_fitness = 0.0
    
    if 'optimization_status' not in st.session_state:
        st.session_state.optimization_status = ""
    
    if 'optimization_config_copy' not in st.session_state:
        st.session_state.optimization_config_copy = None
    
    if 'optimization_error' not in st.session_state:
        st.session_state.optimization_error = None


def page_home():
    """首页"""
    st.title("💧 污水处理工艺仿真与出水水质预测系统")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("系统概述")
        st.write("""
        本系统基于**IWA ASM1活性污泥模型**，对典型污水处理流程进行数学建模和动态仿真，
        可预测不同操作条件下的出水水质。系统支持多种典型工艺模板和自定义流程配置，
        提供稳态求解、动态仿真、参数敏感性分析和工艺优化建议等功能。
        """)
        
        st.subheader("核心功能")
        features = [
            ("🔧 **工艺流程配置**", "支持A2O、SBR、MBR三种典型工艺模板，也可自定义流程"),
            ("🌊 **进水水质配置**", "支持典型进水一键加载和自定义浓度配置，含日变化模式"),
            ("⚗️ **ASM1模型**", "完整的13组分、8反应过程活性污泥模型"),
            ("⚙️ **参数管理**", "动力学参数可编辑，支持温度修正和JSON导入导出"),
            ("🎯 **稳态求解**", "Newton-Raphson迭代求解，显示收敛过程"),
            ("📈 **动态仿真**", "求解ODE刚性系统，支持暂停/继续/终止"),
            ("✅ **达标判定**", "对比中国国标一级A/B标准"),
            ("📊 **敏感性分析**", "单因素/双因素参数扫描"),
            ("💡 **优化建议**", "智能生成操作优化建议"),
        ]
        
        for title, desc in features:
            st.markdown(f"- {title}: {desc}")
    
    with col2:
        st.subheader("快速开始")
        if st.button("🚀 加载A2O工艺示例", use_container_width=True):
            st.session_state.pfs = create_process_by_name('A2O')
            st.success("已加载A2O工艺")
        
        if st.button("⚡ 运行稳态求解", use_container_width=True):
            with st.spinner("正在求解..."):
                result = solve_steady_state(
                    st.session_state.pfs,
                    st.session_state.influent,
                    st.session_state.asm1_params,
                    st.session_state.solver_config,
                )
                st.session_state.steady_result = result
                if result.converged:
                    st.session_state.compliance_result = check_compliance(
                        result.effluent_quality,
                        st.session_state.selected_standard,
                    )
                    st.success("求解完成！")
                else:
                    st.warning("求解未收敛")
        
        st.markdown("---")
        st.subheader("当前配置")
        pfs = st.session_state.pfs
        st.info(f"""
        **工艺**: {len(pfs.reactors)} 个处理单元
        
        **总容积**: {pfs.get_volume():.0f} m³
        
        **进水流量**: {st.session_state.influent.Q_base:.0f} m³/day
        
        **水温**: {st.session_state.asm1_params.temperature:.0f} °C
        """)


def page_process_config():
    """工艺流程配置页面"""
    st.title("🔧 工艺流程配置")
    st.markdown("---")
    
    st.subheader("工艺模板")
    template_cols = st.columns(3)
    templates = list(PROCESS_TEMPLATES.items())
    
    for i, (key, template) in enumerate(templates):
        with template_cols[i]:
            st.markdown(f"### {template.name}")
            st.write(template.description)
            if st.button(f"加载 {template.name}", key=f"load_{key}", use_container_width=True):
                st.session_state.pfs = template.create()
                st.success(f"已加载{template.name}")
    
    st.markdown("---")
    st.subheader("自定义流程")
    
    st.info("💡 **操作说明**: 从下方单元库中点击按钮逐个添加，或使用多选框批量添加多个单元。添加后可在下方调整单元顺序和参数。")
    
    unit_lib = {
        'bar_screen': ('格栅', ReactorType.BAR_SCREEN),
        'grit': ('沉砂池', ReactorType.GRIT),
        'primary': ('初沉池', ReactorType.PRIMARY),
        'anaerobic': ('厌氧池', ReactorType.ANAEROBIC),
        'anoxic': ('缺氧池', ReactorType.ANOXIC),
        'aerobic': ('好氧池', ReactorType.AEROBIC),
        'secondary': ('二沉池', ReactorType.SECONDARY),
        'disinfection': ('消毒池', ReactorType.DISINFECTION),
        'membrane': ('膜组件', ReactorType.MEMBRANE),
    }
    
    add_mode = st.radio(
        "添加方式",
        ["逐个添加", "批量添加"],
        horizontal=True,
        label_visibility="collapsed",
    )
    
    if add_mode == "逐个添加":
        st.markdown("**🏭 单元库** (点击按钮添加)")
        lib_cols = st.columns(3)
        for i, (key, (name, rtype)) in enumerate(unit_lib.items()):
            with lib_cols[i % 3]:
                icon = REACTOR_TYPE_ICONS.get(rtype, '⬜')
                if st.button(f"{icon} 添加{name}", key=f"add_{key}", use_container_width=True):
                    new_reactor = create_reactor_by_type(rtype, f"{name}{len(st.session_state.pfs.reactors)+1}")
                    st.session_state.pfs.add_reactor(new_reactor)
                    if len(st.session_state.pfs.reactors) > 1:
                        st.session_state.pfs.connect(len(st.session_state.pfs.reactors)-2, 
                                                      len(st.session_state.pfs.reactors)-1)
                    st.success(f"✅ 已添加 {icon} {name}")
    else:
        st.markdown("**🏭 单元库** (勾选后点击批量添加)")
        selected_units = []
        lib_cols = st.columns(3)
        for i, (key, (name, rtype)) in enumerate(unit_lib.items()):
            with lib_cols[i % 3]:
                icon = REACTOR_TYPE_ICONS.get(rtype, '⬜')
                if st.checkbox(f"{icon} {name}", key=f"sel_{key}", value=False):
                    selected_units.append((key, name, rtype))
        
        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button(f"➕ 批量添加选中的 {len(selected_units)} 个单元", 
                        type="primary", use_container_width=True, 
                        disabled=len(selected_units) == 0):
                for key, name, rtype in selected_units:
                    new_reactor = create_reactor_by_type(rtype, f"{name}{len(st.session_state.pfs.reactors)+1}")
                    st.session_state.pfs.add_reactor(new_reactor)
                    if len(st.session_state.pfs.reactors) > 1:
                        st.session_state.pfs.connect(len(st.session_state.pfs.reactors)-2, 
                                                      len(st.session_state.pfs.reactors)-1)
                st.success(f"✅ 已批量添加 {len(selected_units)} 个单元")
                st.rerun()
        with col_clear:
            if st.button("🔄 清空选择", use_container_width=True, disabled=len(selected_units) == 0):
                st.rerun()
    
    if len(st.session_state.pfs.reactors) > 0:
        st.markdown("---")
        col_actions = st.columns(3)
        with col_actions[0]:
            if st.button("🗑️ 清空流程", use_container_width=True):
                st.session_state.pfs = ProcessFlowSheet()
                st.warning("已清空流程")
                st.rerun()
        with col_actions[1]:
            if st.button("↩️ 撤销最后一个", use_container_width=True, 
                        disabled=len(st.session_state.pfs.reactors) == 0):
                removed = st.session_state.pfs.reactors.pop()
                st.warning(f"已撤销: {removed.name}")
                st.rerun()
        with col_actions[2]:
            st.metric("当前单元数", len(st.session_state.pfs.reactors))
    
    st.markdown("---")
    st.subheader("当前流程")
    
    pfs = st.session_state.pfs
    
    if len(pfs.reactors) == 0:
        st.info("请选择工艺模板或从单元库添加反应器")
    else:
        fig = plot_process_diagram(pfs)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("单元参数配置")
        
        for idx, reactor in enumerate(pfs.reactors):
            with st.expander(f"{reactor.get_icon()} {reactor.name} - {reactor.get_type_name()}", expanded=(idx == 0)):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**几何参数**")
                    new_vol = st.number_input(
                        "容积 (m³)",
                        min_value=10.0, max_value=20000.0,
                        value=float(reactor.geometry.volume),
                        step=100.0,
                        key=f"vol_{idx}",
                    )
                    new_area = st.number_input(
                        "面积 (m²)",
                        min_value=1.0, max_value=2000.0,
                        value=float(reactor.geometry.area),
                        step=10.0,
                        key=f"area_{idx}",
                    )
                    new_height = st.number_input(
                        "高度 (m)",
                        min_value=1.0, max_value=20.0,
                        value=float(reactor.geometry.height),
                        step=0.5,
                        key=f"height_{idx}",
                    )
                    
                    reactor.geometry.volume = new_vol
                    reactor.geometry.area = new_area
                    reactor.geometry.height = new_height
                
                with col2:
                    st.markdown("**运行参数**")
                    new_hrt = st.number_input(
                        "HRT (小时)",
                        min_value=0.1, max_value=72.0,
                        value=float(reactor.operation.HRT),
                        step=0.5,
                        key=f"hrt_{idx}",
                    )
                    new_srt = st.number_input(
                        "SRT (天)",
                        min_value=0.0, max_value=60.0,
                        value=float(reactor.operation.SRT),
                        step=1.0,
                        key=f"srt_{idx}",
                    )
                    
                    if reactor.reactor_type == ReactorType.AEROBIC:
                        new_do = st.number_input(
                            "DO设定值 (mg/L)",
                            min_value=0.0, max_value=6.0,
                            value=float(reactor.operation.DO_setpoint),
                            step=0.1,
                            key=f"do_{idx}",
                        )
                        reactor.operation.DO_setpoint = new_do
                    
                    new_rr = st.number_input(
                        "回流比 (%)",
                        min_value=0, max_value=300,
                        value=int(reactor.operation.return_sludge_ratio * 100),
                        step=10,
                        key=f"rr_{idx}",
                    )
                    reactor.operation.return_sludge_ratio = new_rr / 100.0
                    
                    if hasattr(reactor.operation, 'internal_return_ratio'):
                        new_irr = st.number_input(
                            "内回流比 (%)",
                            min_value=0, max_value=500,
                            value=int(reactor.operation.internal_return_ratio * 100),
                            step=50,
                            key=f"irr_{idx}",
                        )
                        reactor.operation.internal_return_ratio = new_irr / 100.0
                    
                    reactor.operation.HRT = new_hrt
                    reactor.operation.SRT = new_srt
                
                col_act1, col_act2, col_act3 = st.columns(3)
                with col_act1:
                    if st.button("⬆️ 上移", key=f"up_{idx}", disabled=(idx == 0)):
                        if idx > 0:
                            pfs.reactors[idx-1], pfs.reactors[idx] = pfs.reactors[idx], pfs.reactors[idx-1]
                            st.rerun()
                with col_act2:
                    if st.button("⬇️ 下移", key=f"down_{idx}", disabled=(idx == len(pfs.reactors)-1)):
                        if idx < len(pfs.reactors) - 1:
                            pfs.reactors[idx+1], pfs.reactors[idx] = pfs.reactors[idx], pfs.reactors[idx+1]
                            st.rerun()
                with col_act3:
                    if st.button("❌ 删除", key=f"del_{idx}"):
                        pfs.reactors.pop(idx)
                        pfs.connections = []
                        for i in range(len(pfs.reactors) - 1):
                            pfs.connect(i, i + 1)
                        st.rerun()
                
                if len(pfs.reactors) > 2:
                    col_move, col_pos, col_go = st.columns([2, 1, 1])
                    with col_move:
                        target_pos = st.selectbox(
                            "移动到",
                            [f"第{i+1}位" for i in range(len(pfs.reactors))],
                            index=idx,
                            key=f"pos_{idx}",
                            label_visibility="collapsed",
                        )
                    with col_pos:
                        pass
                    with col_go:
                        if st.button("→ 移动", key=f"move_{idx}"):
                            target_idx = int(target_pos.replace("第", "").replace("位", "")) - 1
                            if target_idx != idx:
                                reactor = pfs.reactors.pop(idx)
                                pfs.reactors.insert(target_idx, reactor)
                                pfs.connections = []
                                for i in range(len(pfs.reactors) - 1):
                                    pfs.connect(i, i + 1)
                                st.success(f"✅ 已将「{reactor.name}」从第{idx+1}位移到第{target_idx+1}位")
                                st.rerun()
        
        total_volume = pfs.get_volume()
        Q = st.session_state.influent.Q_base
        total_hrt = total_volume / Q * 24 if Q > 0 else 0
        st.info(f"**总容积**: {total_volume:.0f} m³ | **总HRT**: {total_hrt:.1f} 小时 (流量 {Q:.0f} m³/d)")


def page_influent_config():
    """进水水质配置页面"""
    st.title("🌊 进水水质配置")
    st.markdown("---")
    
    influent = st.session_state.influent
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("进水流量")
        flow_mode = st.radio("流量模式", ["恒定", "日变化"], index=0 if influent.flow_mode == 'constant' else 1)
        influent.flow_mode = 'constant' if flow_mode == "恒定" else 'diurnal'
        
        Q_base = st.number_input(
            "日均流量 (m³/day)",
            min_value=100.0, max_value=100000.0,
            value=float(influent.Q_base),
            step=500.0,
        )
        influent.Q_base = Q_base
    
    with col2:
        st.subheader("进水水质类型")
        quality_mode = st.radio("水质模式", ["典型进水", "自定义"], index=0 if influent.quality_mode == 'typical' else 1)
        influent.quality_mode = 'typical' if quality_mode == "典型进水" else 'custom'
        
        if quality_mode == "典型进水":
            influent_type = st.selectbox(
                "选择典型进水",
                ['生活污水', '工业混合', '高浓度有机废水'],
                index=['domestic', 'industrial', 'high_strength'].index(influent.influent_type),
            )
            type_map = {'生活污水': 'domestic', '工业混合': 'industrial', '高浓度有机废水': 'high_strength'}
            influent.influent_type = type_map[influent_type]
            
            if st.button("加载典型进水", use_container_width=True):
                C = get_typical_influent(influent.influent_type)
                wq = aggregate_to_wq_indices(C)
                for key in wq:
                    influent.custom_quality[key] = wq[key]
                st.success(f"已加载{influent_type}水质")
    
    st.markdown("---")
    st.subheader("进水组分浓度")
    
    if quality_mode == "自定义":
        wq_cols = st.columns(3)
        params = [
            ('COD', 'COD (mg/L)', 50, 5000, 10),
            ('BOD5', 'BOD5 (mg/L)', 20, 3000, 10),
            ('NH3_N', 'NH3-N (mg/L)', 5, 500, 5),
            ('TN', 'TN (mg/L)', 10, 800, 10),
            ('TP', 'TP (mg/L)', 1, 100, 1),
            ('SS', 'SS (mg/L)', 50, 2000, 50),
        ]
        
        for i, (key, label, min_val, max_val, step) in enumerate(params):
            with wq_cols[i % 3]:
                val = st.number_input(
                    label,
                    min_value=min_val, max_value=max_val,
                    value=float(influent.custom_quality.get(key, 0)),
                    step=step,
                    key=f"wq_{key}",
                )
                influent.custom_quality[key] = val
    
    C_current = influent.get_C(0)
    wq_current = aggregate_to_wq_indices(C_current)
    
    wq_df = pd.DataFrame({
        '指标': ['COD', 'BOD5', 'NH3-N', 'TN', 'TP', 'SS'],
        '浓度 (mg/L)': [wq_current['COD'], wq_current['BOD5'], wq_current['NH3_N'], 
                       wq_current['TN'], wq_current['TP'], wq_current['SS']],
        '单位': ['mg/L'] * 6,
    })
    st.table(wq_df)
    
    st.markdown("---")
    st.subheader("日变化模式")
    
    if flow_mode == "日变化":
        st.info("💡 **操作说明**: 选择预设模式后，可通过下方的「关键节点调整」拖拽修改各时段的变化系数，也可展开「逐小时精细调整」修改每个小时的值。")
        
        col_pattern, col_apply = st.columns([2, 1])
        with col_pattern:
            pattern_type = st.selectbox(
                "选择预设日变化模式",
                ['早晚高峰模式', '均匀模式', '单峰模式', '工业模式'],
                index=0,
            )
        with col_apply:
            st.markdown("")
            st.markdown("")
            if st.button("🔄 应用预设模式", use_container_width=True):
                if pattern_type == '早晚高峰模式':
                    influent.set_diurnal_pattern('morning_evening_peak')
                elif pattern_type == '单峰模式':
                    influent.set_diurnal_pattern('morning_evening_peak')
                    for h in range(24):
                        if 8 <= h < 20:
                            influent.diurnal_flow_curve[h] = 1.2 + 0.3 * np.sin(np.pi * (h - 8) / 12)
                            influent.diurnal_curve[h] = 1.1 + 0.2 * np.sin(np.pi * (h - 8) / 12)
                elif pattern_type == '工业模式':
                    influent.set_diurnal_pattern('uniform')
                    for h in range(24):
                        if 8 <= h < 18:
                            influent.diurnal_flow_curve[h] = 1.3
                            influent.diurnal_curve[h] = 1.2
                else:
                    influent.set_diurnal_pattern('uniform')
                st.rerun()
        
        adjust_mode = st.radio(
            "调整方式",
            ["🎯 可视化节点编辑", "🔧 关键节点调整", "⏰ 逐小时精细调整"],
            horizontal=True,
            label_visibility="collapsed",
        )
        
        if adjust_mode == "🎯 可视化节点编辑":
            st.markdown("**🎯 可视化节点编辑** (点击小时按钮选择节点，拖拽滑块实时调整曲线)")
            
            st.markdown("**选择要编辑的时间点**:")
            hour_cols = st.columns(12)
            for h in range(24):
                with hour_cols[h % 12]:
                    btn_label = f"{'●' if st.session_state.selected_hour == h else '○'} {h:02d}"
                    if st.button(btn_label, key=f"sel_hour_{h}", 
                                type="primary" if st.session_state.selected_hour == h else "secondary",
                                use_container_width=True):
                        st.session_state.selected_hour = h
                        st.rerun()
            
            st.markdown("---")
            
            selected_h = st.session_state.selected_hour
            st.info(f"📌 当前选中: **{selected_h:02d}:00** - 拖拽下方滑块调整该时间点的流量和浓度，曲线实时更新")
            
            col_flow, col_conc = st.columns(2)
            
            with col_flow:
                st.markdown(f"**🌊 流量系数 - {selected_h:02d}:00**")
                current_flow = float(influent.diurnal_flow_curve[selected_h])
                new_flow = st.slider(
                    f"调整 {selected_h:02d}:00 流量系数",
                    min_value=0.5, max_value=2.0,
                    value=current_flow,
                    step=0.01,
                    key=f"vis_flow_{selected_h}",
                    label_visibility="collapsed",
                )
                influent.diurnal_flow_curve[selected_h] = new_flow
                
                smooth_nearby = st.checkbox(
                    "平滑相邻时段",
                    value=True,
                    key=f"smooth_flow_{selected_h}",
                    help="自动平滑调整点前后1小时的系数，使曲线更自然"
                )
                if smooth_nearby:
                    prev_h = (selected_h - 1) % 24
                    next_h = (selected_h + 1) % 24
                    influent.diurnal_flow_curve[prev_h] = 0.7 * influent.diurnal_flow_curve[prev_h] + 0.3 * new_flow
                    influent.diurnal_flow_curve[next_h] = 0.7 * influent.diurnal_flow_curve[next_h] + 0.3 * new_flow
            
            with col_conc:
                st.markdown(f"**🔬 浓度系数 - {selected_h:02d}:00**")
                current_conc = float(influent.diurnal_curve[selected_h])
                new_conc = st.slider(
                    f"调整 {selected_h:02d}:00 浓度系数",
                    min_value=0.5, max_value=2.0,
                    value=current_conc,
                    step=0.01,
                    key=f"vis_conc_{selected_h}",
                    label_visibility="collapsed",
                )
                influent.diurnal_curve[selected_h] = new_conc
                
                smooth_nearby_c = st.checkbox(
                    "平滑相邻时段",
                    value=True,
                    key=f"smooth_conc_{selected_h}",
                    help="自动平滑调整点前后1小时的系数，使曲线更自然"
                )
                if smooth_nearby_c:
                    prev_h = (selected_h - 1) % 24
                    next_h = (selected_h + 1) % 24
                    influent.diurnal_curve[prev_h] = 0.7 * influent.diurnal_curve[prev_h] + 0.3 * new_conc
                    influent.diurnal_curve[next_h] = 0.7 * influent.diurnal_curve[next_h] + 0.3 * new_conc
            
            st.markdown("**📈 实时曲线预览** (红色标记为当前选中节点)")
            fig = plot_influent_diurnal(influent, selected_hour=st.session_state.selected_hour, 
                                       show_editable_hint=False)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            **当前 {selected_h:02d}:00 数值:**
            - 流量系数: {new_flow:.2f} (流量: {new_flow * influent.Q_base:.0f} m³/d)
            - 浓度系数: {new_conc:.2f}
            {f'- 相邻时段已平滑' if smooth_nearby or smooth_nearby_c else ''}
            """)
        
        elif adjust_mode == "🔧 关键节点调整":
            st.markdown("**🔧 关键节点调整** (拖拽调整主要时段，中间时段自动插值)")
            
            key_hours = [0, 6, 9, 12, 15, 18, 21]
            key_hours_labels = ["00:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
            
            tab_flow, tab_conc = st.tabs(["🌊 流量调整", "🔬 浓度调整"])
            
            with tab_flow:
                col_nodes = st.columns(len(key_hours))
                flow_key_values = []
                for i, h in enumerate(key_hours):
                    with col_nodes[i]:
                        val = st.slider(
                            key_hours_labels[i],
                            min_value=0.5, max_value=2.0,
                            value=float(influent.diurnal_flow_curve[h]),
                            step=0.05,
                            key=f"flow_key_{h}",
                        )
                        flow_key_values.append(val)
                
                for h in range(24):
                    for i in range(len(key_hours) - 1):
                        if key_hours[i] <= h < key_hours[i + 1]:
                            t = (h - key_hours[i]) / (key_hours[i + 1] - key_hours[i])
                            influent.diurnal_flow_curve[h] = flow_key_values[i] * (1 - t) + flow_key_values[i + 1] * t
                            break
                
                st.success(f"已通过 {len(key_hours)} 个关键节点插值生成24小时流量曲线")
            
            with tab_conc:
                col_nodes = st.columns(len(key_hours))
                conc_key_values = []
                for i, h in enumerate(key_hours):
                    with col_nodes[i]:
                        val = st.slider(
                            key_hours_labels[i],
                            min_value=0.5, max_value=2.0,
                            value=float(influent.diurnal_curve[h]),
                            step=0.05,
                            key=f"conc_key_{h}",
                        )
                        conc_key_values.append(val)
                
                for h in range(24):
                    for i in range(len(key_hours) - 1):
                        if key_hours[i] <= h < key_hours[i + 1]:
                            t = (h - key_hours[i]) / (key_hours[i + 1] - key_hours[i])
                            influent.diurnal_curve[h] = conc_key_values[i] * (1 - t) + conc_key_values[i + 1] * t
                            break
                
                st.success(f"已通过 {len(key_hours)} 个关键节点插值生成24小时浓度曲线")
            
            st.markdown("**📊 快速时段调整** (批量设置时段系数)")
            col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
            with col_preset1:
                night_factor = st.slider("夜间(0-6)", 0.5, 2.0, 0.7, 0.05)
            with col_preset2:
                morning_factor = st.slider("早高峰(6-10)", 0.5, 2.0, 1.5, 0.05)
            with col_preset3:
                day_factor = st.slider("白天(10-18)", 0.5, 2.0, 1.0, 0.05)
            with col_preset4:
                evening_factor = st.slider("晚高峰(18-24)", 0.5, 2.0, 1.3, 0.05)
            
            if st.button("📋 应用时段设置到流量和浓度", use_container_width=True):
                for h in range(24):
                    if h < 6:
                        influent.diurnal_flow_curve[h] = night_factor
                        influent.diurnal_curve[h] = night_factor * 0.95
                    elif h < 10:
                        t = (h - 6) / 4
                        influent.diurnal_flow_curve[h] = night_factor * (1 - t) + morning_factor * t
                        influent.diurnal_curve[h] = (night_factor * 0.95) * (1 - t) + (morning_factor * 0.95) * t
                    elif h < 18:
                        t = (h - 10) / 8
                        influent.diurnal_flow_curve[h] = morning_factor * (1 - t) + day_factor * t
                        influent.diurnal_curve[h] = (morning_factor * 0.95) * (1 - t) + (day_factor * 0.95) * t
                    else:
                        t = (h - 18) / 6
                        influent.diurnal_flow_curve[h] = day_factor * (1 - t) + evening_factor * t
                        influent.diurnal_curve[h] = (day_factor * 0.95) * (1 - t) + (evening_factor * 0.95) * t
                        if h >= 22:
                            t2 = (h - 22) / 2
                            influent.diurnal_flow_curve[h] = evening_factor * (1 - t2) + night_factor * t2
                            influent.diurnal_curve[h] = (evening_factor * 0.95) * (1 - t2) + (night_factor * 0.95) * t2
                st.rerun()
        
        else:
            st.markdown("**⏰ 逐小时精细调整**")
            
            col_flow, col_conc = st.columns(2)
            
            with col_flow:
                st.markdown("**🌊 流量日变化曲线**")
                flow_values = []
                for h in range(24):
                    val = st.slider(
                        f"{h:02d}:00",
                        min_value=0.5, max_value=2.0,
                        value=float(influent.diurnal_flow_curve[h]),
                        step=0.05,
                        key=f"flow_{h}",
                    )
                    flow_values.append(val)
                influent.diurnal_flow_curve = np.array(flow_values)
            
            with col_conc:
                st.markdown("**🔬 浓度日变化曲线**")
                conc_values = []
                for h in range(24):
                    val = st.slider(
                        f"{h:02d}:00",
                        min_value=0.5, max_value=2.0,
                        value=float(influent.diurnal_curve[h]),
                        step=0.05,
                        key=f"conc_{h}",
                    )
                    conc_values.append(val)
                influent.diurnal_curve = np.array(conc_values)
    
    fig = plot_influent_diurnal(influent)
    st.plotly_chart(fig, use_container_width=True)


def page_parameters():
    """ASM1参数管理页面"""
    st.title("⚗️ ASM1动力学参数管理")
    st.markdown("---")
    
    params = st.session_state.asm1_params
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("温度设置")
        temperature = st.slider(
            "水温 (°C)",
            min_value=10.0, max_value=35.0,
            value=float(params.temperature),
            step=0.5,
        )
        params.temperature = temperature
        
        st.info(f"当前温度: {temperature:.1f} °C\n"
                f"参数已按Arrhenius公式修正")
        
        st.markdown("---")
        
        if st.button("↺ 恢复默认参数", use_container_width=True):
            params.reset_to_default()
            st.success("已恢复默认参数")
        
        st.markdown("---")
        
        col_exp, col_imp = st.columns(2)
        
        with col_exp:
            if st.button("📤 导出参数", use_container_width=True):
                json_str = params.to_json()
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="asm1_parameters.json">下载参数文件</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        with col_imp:
            uploaded_file = st.file_uploader("📥 导入参数", type=['json'], label_visibility="collapsed")
            if uploaded_file is not None:
                try:
                    data = json.load(uploaded_file)
                    params.from_dict(data)
                    st.success("参数导入成功")
                except Exception as e:
                    st.error(f"导入失败: {str(e)}")
    
    with col2:
        st.subheader("动力学参数表")
        st.caption("参数基于20°C标准值，温度修正已自动应用")
        
        temp_corrected = params.get_temperature_corrected_params(temperature)
        
        stoich_params = [
            ('Y_H', '异养菌产率系数'),
            ('Y_A', '自养菌产率系数'),
            ('f_P', '惰性颗粒产物比例'),
            ('i_XB', '菌体氮含量'),
            ('i_XP', '惰性产物氮含量'),
        ]
        
        kinetic_params = [
            ('mu_H', '异养菌最大比生长速率', '1/day'),
            ('K_S', '异养菌半饱和系数', 'mg COD/L'),
            ('K_O_H', '异养菌氧半饱和系数', 'mg O2/L'),
            ('K_NO', '硝酸盐半饱和系数', 'mg N/L'),
            ('b_H', '异养菌衰减系数', '1/day'),
            ('eta_g', '缺氧生长修正系数', '无量纲'),
            ('eta_h', '缺氧水解修正系数', '无量纲'),
            ('k_h', '颗粒有机物水解速率', '1/day'),
            ('K_X', '水解半饱和系数', 'mg COD/mg COD'),
            ('mu_A', '自养菌最大比生长速率', '1/day'),
            ('K_O_A', '自养菌氧半饱和系数', 'mg O2/L'),
            ('K_NH', '氨氮半饱和系数', 'mg N/L'),
            ('b_A', '自养菌衰减系数', '1/day'),
            ('k_a', '氨化速率', 'L/(mg COD·day)'),
        ]
        
        st.markdown("**化学计量系数**")
        stoich_data = []
        for key, desc in stoich_params:
            desc_info = params.description.get(key, (desc, ''))
            unit = desc_info[1] if len(desc_info) > 1 else ''
            val = getattr(params, key)
            stoich_data.append({
                '参数': key,
                '描述': desc_info[0] if desc_info else desc,
                '单位': unit,
                '默认值': val,
                '当前值': val,
            })
        
        st.dataframe(
            pd.DataFrame(stoich_data),
            hide_index=True,
            use_container_width=True,
        )
        
        st.markdown("**动力学参数**")
        kinetic_data = []
        for key, desc, unit in kinetic_params:
            desc_info = params.description.get(key, (desc, ''))
            default_val = getattr(params, key)
            corrected_val = getattr(temp_corrected, key)
            kinetic_data.append({
                '参数': key,
                '描述': desc_info[0] if desc_info else desc,
                '单位': unit,
                '20°C值': default_val,
                f'{temperature:.0f}°C值': corrected_val,
            })
        
        k_df = pd.DataFrame(kinetic_data)
        edited_df = st.data_editor(
            k_df,
            column_config={
                "20°C值": st.column_config.NumberColumn("20°C值", format="%.4f"),
                f"{temperature:.0f}°C值": st.column_config.NumberColumn(
                    f"{temperature:.0f}°C值", format="%.4f", disabled=True
                ),
            },
            hide_index=True,
            use_container_width=True,
        )
        
        for i, (key, _, _) in enumerate(kinetic_params):
            new_val = edited_df.iloc[i]['20°C值']
            setattr(params, key, float(new_val))
    
    st.markdown("---")
    st.subheader("化学计量矩阵")
    
    stoich_matrix = get_stoichiometric_matrix()
    
    process_names = [
        '好氧异养生长', '缺氧异养生长', '好氧自养生长',
        '异养菌衰减', '自养菌衰减', '氨化', '水解', '有机氮水解'
    ]
    
    matrix_df = pd.DataFrame(
        stoich_matrix,
        index=COMPONENT_NAMES,
        columns=process_names,
    )
    
    st.dataframe(
        matrix_df.style.background_gradient(cmap='RdBu_r', vmin=-3, vmax=3),
        use_container_width=True,
        height=400,
    )


def page_steady_state():
    """稳态求解页面"""
    st.title("🎯 稳态求解")
    st.markdown("---")
    
    config = st.session_state.solver_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("求解器配置")
        max_iter = st.number_input(
            "最大迭代次数",
            min_value=10, max_value=200,
            value=int(config.max_iterations),
            step=10,
        )
        tolerance = st.number_input(
            "收敛阈值",
            min_value=1e-10, max_value=1e-2,
            value=float(config.tolerance),
            step=1e-7,
            format="%.1e",
            help="当残差范数小于此值时判定为收敛",
        )
        relaxation = st.number_input(
            "松弛因子",
            min_value=0.1, max_value=1.0,
            value=float(config.relaxation),
            step=0.1,
            help="Newton迭代的松弛因子，较小值更稳定但收敛更慢",
        )
        
        use_engineering = st.checkbox(
            "启用工程收敛标准",
            value=config.use_engineering_tolerance,
            help="允许在残差较大但工程可接受的情况下判定为收敛",
        )
        
        eng_factor = 50000.0
        if use_engineering:
            eng_factor = st.slider(
                "工程容差倍数",
                min_value=1000.0, max_value=200000.0,
                value=float(config.engineering_tolerance_factor),
                step=1000.0,
                help=f"工程收敛阈值 = 收敛阈值 × 倍数，当前: {tolerance * eng_factor:.2e}",
            )
        
        config.max_iterations = max_iter
        config.tolerance = tolerance
        config.relaxation = relaxation
        config.use_engineering_tolerance = use_engineering
        config.engineering_tolerance_factor = eng_factor
        
        st.info(f"""
        **收敛标准说明:**
        - 严格收敛: 残差 < {tolerance:.2e}
        {'- 工程收敛: 残差 < ' + f'{tolerance * eng_factor:.2e}' if use_engineering else ''}
        """)
    
    with col2:
        st.subheader("排放标准")
        standard = st.selectbox(
            "选择排放标准",
            STANDARD_NAMES,
            index=STANDARD_NAMES.index(st.session_state.selected_standard),
        )
        st.session_state.selected_standard = standard
        
        std = STANDARDS[standard]
        st.info(f"""
        **{std.name}**
        
        COD ≤ {std.COD} mg/L | BOD5 ≤ {std.BOD5} mg/L
        
        NH3-N ≤ {std.NH3_N} mg/L | TN ≤ {std.TN} mg/L
        
        TP ≤ {std.TP} mg/L | SS ≤ {std.SS} mg/L
        """)
    
    st.markdown("---")
    
    if st.button("▶️ 运行稳态求解", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(iter_num, residual):
            progress = min(iter_num / max_iter, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"迭代 {iter_num}/{max_iter} | 残差: {residual:.2e}")
        
        with st.spinner("正在求解..."):
            result = solve_steady_state(
                st.session_state.pfs,
                st.session_state.influent,
                st.session_state.asm1_params,
                config,
                progress_callback=progress_callback,
            )
            st.session_state.steady_result = result
            
            if result.converged:
                st.session_state.compliance_result = check_compliance(
                    result.effluent_quality,
                    standard,
                )
                
                st.session_state.optimization_suggestions = generate_optimization_suggestions(
                    st.session_state.pfs,
                    st.session_state.influent,
                    st.session_state.asm1_params,
                    st.session_state.compliance_result,
                )
                
                Q = st.session_state.influent.Q_base
                st.session_state.sludge_result = calculate_sludge_production(
                    st.session_state.pfs,
                    result.reactor_states,
                    Q,
                    st.session_state.asm1_params,
                )
                
                st.session_state.energy_result = calculate_energy_consumption(
                    st.session_state.pfs,
                    result.reactor_states,
                    st.session_state.influent,
                    st.session_state.asm1_params,
                )
                
                progress_bar.progress(1.0)
                status_text.text("求解完成！")
            else:
                status_text.text("求解未收敛")
    
    if st.session_state.steady_result is not None:
        result = st.session_state.steady_result
        
        st.markdown("---")
        st.subheader("求解结果")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if result.converged:
                st.success(result.message)
            else:
                st.error(result.message)
                st.warning("建议检查：DO设定是否合理、SRT是否过短、进水负荷是否过高")
        
        with col_res2:
            fig = plot_residual_convergence(result)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("各池浓度分布")
        
        fig = plot_reactor_stack(st.session_state.pfs, result.reactor_states)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("出水水质")
        
        effluent = result.effluent_quality
        col_e1, col_e2, col_e3, col_e4, col_e5, col_e6 = st.columns(6)
        
        metrics = [
            ('COD', 'mg/L', effluent['COD']),
            ('BOD5', 'mg/L', effluent['BOD5']),
            ('NH3-N', 'mg/L', effluent['NH3_N']),
            ('TN', 'mg/L', effluent['TN']),
            ('TP', 'mg/L', effluent['TP']),
            ('SS', 'mg/L', effluent['SS']),
        ]
        
        cols = [col_e1, col_e2, col_e3, col_e4, col_e5, col_e6]
        for i, (name, unit, val) in enumerate(metrics):
            with cols[i]:
                st.metric(name, f"{val:.2f}", unit)
        
        if st.session_state.compliance_result is not None:
            st.markdown("---")
            st.subheader("达标判定")
            
            comp = st.session_state.compliance_result
            
            comp_data = []
            for item in comp.items:
                status = "✅ 达标" if item.compliant else "❌ 超标"
                comp_data.append({
                    '指标': item.name,
                    '出水值': f"{item.value:.2f} {item.unit}",
                    '标准限值': f"{item.limit} {item.unit}",
                    '占标率': f"{item.ratio*100:.1f}%",
                    '达标情况': status,
                    '建议': item.suggestion,
                })
            
            comp_df = pd.DataFrame(comp_data)
            
            def highlight_compliant(row):
                color = 'background-color: #d4edda' if '达标' in row['达标情况'] else 'background-color: #f8d7da'
                return [color] * len(row)
            
            st.dataframe(
                comp_df.style.apply(highlight_compliant, axis=1),
                hide_index=True,
                use_container_width=True,
            )
            
            if comp.overall_compliant:
                st.success("🎉 出水水质全面达标！")
            else:
                st.error("⚠️ 存在超标指标，请参考建议进行工艺调整")
            
            col_radar, col_table = st.columns(2)
            with col_radar:
                fig_radar = plot_compliance_radar(comp)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with col_table:
                st.markdown("**各池详细出水**")
                reactor_data = []
                for i, (reactor, state) in enumerate(zip(st.session_state.pfs.reactors, result.reactor_states)):
                    wq = aggregate_to_wq_indices(state)
                    reactor_data.append({
                        '单元': reactor.name,
                        'COD': wq['COD'],
                        'BOD5': wq['BOD5'],
                        'NH3-N': wq['NH3_N'],
                        'TN': wq['TN'],
                        'TP': wq['TP'],
                        'SS': wq['SS'],
                    })
                st.dataframe(pd.DataFrame(reactor_data), hide_index=True, use_container_width=True)
        
        if result.converged:
            st.markdown("---")
            st.subheader("📦 污泥产量分析")
            
            if st.session_state.sludge_result is not None:
                sludge = st.session_state.sludge_result
                
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                with col_s1:
                    st.metric("日剩余污泥产量", f"{sludge.daily_sludge_kg:.1f}", "kg DS/d")
                with col_s2:
                    st.metric("平均MLSS", f"{sludge.MLSS_gL:.2f}", "g/L")
                with col_s3:
                    st.metric("系统总生物量", f"{sludge.total_biomass_kg:.0f}", "kg")
                with col_s4:
                    st.metric("异养菌/自养菌", f"{sludge.XBH_kg:.0f} / {sludge.XBA_kg:.0f}", "kg")
                
                with st.expander("📊 查看污泥组成详情", expanded=False):
                    sludge_detail_data = [
                        {'组分': '异养菌 (XBH)', '质量 (kg)': sludge.XBH_kg, '占比 (%)': round(sludge.XBH_kg / sludge.total_biomass_kg * 100, 1) if sludge.total_biomass_kg > 0 else 0},
                        {'组分': '自养菌 (XBA)', '质量 (kg)': sludge.XBA_kg, '占比 (%)': round(sludge.XBA_kg / sludge.total_biomass_kg * 100, 1) if sludge.total_biomass_kg > 0 else 0},
                        {'组分': '代谢产物 (XP)', '质量 (kg)': sludge.XP_kg, '占比 (%)': round(sludge.XP_kg / sludge.total_biomass_kg * 100, 1) if sludge.total_biomass_kg > 0 else 0},
                        {'组分': '惰性颗粒 (XI)', '质量 (kg)': sludge.XI_kg, '占比 (%)': round(sludge.XI_kg / sludge.total_biomass_kg * 100, 1) if sludge.total_biomass_kg > 0 else 0},
                        {'组分': '缓慢降解 (XS)', '质量 (kg)': sludge.XS_kg, '占比 (%)': round(sludge.XS_kg / sludge.total_biomass_kg * 100, 1) if sludge.total_biomass_kg > 0 else 0},
                    ]
                    st.dataframe(pd.DataFrame(sludge_detail_data), hide_index=True, use_container_width=True)
                    
                    st.info(f"""
                    💡 **污泥处理提示**:
                    - 日产量: {sludge.daily_sludge_kg:.1f} kg DS/d ≈ {sludge.daily_sludge_kg/1000:.3f} 吨/天
                    - 年产量: {sludge.daily_sludge_kg * 365 / 1000:.1f} 吨/年
                    - 污泥浓度: {sludge.sludge_concentration_mgL:.0f} mg/L = {sludge.sludge_concentration_mgL/1000:.1f} g/L
                    """)
                
                if st.button("📈 生成SRT-污泥产量关系曲线", use_container_width=True):
                    with st.spinner("正在计算SRT-污泥产量曲线..."):
                        srt_range, sludge_values, converged_list = generate_srt_vs_sludge_curve(
                            st.session_state.pfs,
                            st.session_state.influent,
                            st.session_state.asm1_params,
                            config=st.session_state.solver_config,
                        )
                        st.session_state.srt_sludge_curve = (srt_range, sludge_values, converged_list)
                        st.success("曲线生成完成！")
                
                if st.session_state.srt_sludge_curve is not None:
                    srt_range, sludge_values, converged_list = st.session_state.srt_sludge_curve
                    
                    current_srt = 0.0
                    srt_count = 0
                    for reactor in st.session_state.pfs.reactors:
                        if reactor.is_biological() and hasattr(reactor.operation, 'SRT') and reactor.operation.SRT > 0:
                            current_srt += reactor.operation.SRT
                            srt_count += 1
                    current_srt = current_srt / srt_count if srt_count > 0 else 10.0
                    
                    fig_srt = plot_srt_vs_sludge(
                        srt_range, sludge_values,
                        current_srt=current_srt,
                        current_sludge=sludge.daily_sludge_kg,
                        converged_list=converged_list,
                    )
                    st.plotly_chart(fig_srt, use_container_width=True)
            
            st.markdown("---")
            st.subheader("⚡ 能耗估算")
            
            if st.session_state.energy_result is not None:
                energy = st.session_state.energy_result
                
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    st.metric("日均总电耗", f"{energy.total_kwh_d:.1f}", "kWh/d")
                with col_e2:
                    st.metric("单位水量电耗", f"{energy.unit_kwh_m3:.4f}", "kWh/m³")
                with col_e3:
                    annual_cost = energy.total_kwh_d * 365 * 0.8
                    st.metric("估算年电费", f"{annual_cost/10000:.1f}", "万元/年 (0.8元/kWh)")
                
                col_pie, col_table = st.columns(2)
                
                with col_pie:
                    fig_pie = plot_energy_pie(energy)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_table:
                    st.markdown("**能耗分项明细**")
                    energy_detail_data = [
                        {'分项': '曝气系统', '能耗 (kWh/d)': energy.aeration_kwh_d, '占比 (%)': round(energy.aeration_kwh_d / energy.total_kwh_d * 100, 1) if energy.total_kwh_d > 0 else 0},
                        {'分项': '回流泵', '能耗 (kWh/d)': energy.return_pump_kwh_d, '占比 (%)': round(energy.return_pump_kwh_d / energy.total_kwh_d * 100, 1) if energy.total_kwh_d > 0 else 0},
                        {'分项': '内回流泵', '能耗 (kWh/d)': energy.internal_pump_kwh_d, '占比 (%)': round(energy.internal_pump_kwh_d / energy.total_kwh_d * 100, 1) if energy.total_kwh_d > 0 else 0},
                        {'分项': '搅拌系统', '能耗 (kWh/d)': energy.mixing_kwh_d, '占比 (%)': round(energy.mixing_kwh_d / energy.total_kwh_d * 100, 1) if energy.total_kwh_d > 0 else 0},
                        {'分项': '其他', '能耗 (kWh/d)': energy.other_kwh_d, '占比 (%)': round(energy.other_kwh_d / energy.total_kwh_d * 100, 1) if energy.total_kwh_d > 0 else 0},
                    ]
                    st.dataframe(pd.DataFrame(energy_detail_data), hide_index=True, use_container_width=True)
                    
                    st.info(f"""
                    💡 **节能提示**:
                    - 曝气系统占比最大 ({energy.aeration_kwh_d / energy.total_kwh_d * 100:.0f}%)，可通过优化DO设定节能
                    - 当前单位水量电耗 {energy.unit_kwh_m3:.4f} kWh/m³，处于{ '较低' if energy.unit_kwh_m3 < 0.3 else '中等' if energy.unit_kwh_m3 < 0.6 else '较高' }水平
                    """)
            
            st.markdown("---")
            st.subheader("📄 报告导出")
            
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                export_format = st.radio("导出格式", ["HTML", "PDF"], horizontal=True)
                include_charts = st.checkbox("包含图表", value=True)
            
            with col_export2:
                st.markdown("")
                st.markdown("")
                if st.button("📥 生成并下载报告", type="primary", use_container_width=True):
                    with st.spinner(f"正在生成{export_format}报告..."):
                        try:
                            process_name = st.session_state.current_page
                            if process_name == "🎯 稳态求解":
                                process_name = "A2O工艺"
                            
                            report_data = ReportData(
                                pfs=st.session_state.pfs,
                                influent=st.session_state.influent,
                                asm1_params=st.session_state.asm1_params,
                                steady_result=result,
                                compliance_result=st.session_state.compliance_result,
                                optimization_suggestions=st.session_state.optimization_suggestions,
                                sludge_result=st.session_state.sludge_result,
                                energy_result=st.session_state.energy_result,
                                process_name=process_name,
                                standard_name=st.session_state.selected_standard,
                            )
                            
                            if export_format == "HTML":
                                html_content, filename = generate_html_report(report_data, include_charts=include_charts)
                                download_link = get_download_link_html(html_content, filename, 'html')
                                st.success(f"✅ HTML报告生成成功！")
                                st.markdown(download_link, unsafe_allow_html=True)
                                
                                st.download_button(
                                    label="💾 直接下载HTML报告",
                                    data=html_content,
                                    file_name=filename,
                                    mime="text/html",
                                    use_container_width=True,
                                )
                            else:
                                try:
                                    pdf_bytes, filename = generate_pdf_report(report_data, include_charts=include_charts)
                                    st.success(f"✅ PDF报告生成成功！")
                                    
                                    st.download_button(
                                        label="💾 直接下载PDF报告",
                                        data=pdf_bytes,
                                        file_name=filename,
                                        mime="application/pdf",
                                        use_container_width=True,
                                    )
                                except ImportError as e:
                                    st.warning(f"⚠️ PDF导出需要安装额外依赖: {str(e)}")
                                    st.info("将自动生成HTML格式报告")
                                    html_content, filename = generate_html_report(report_data, include_charts=include_charts)
                                    st.download_button(
                                        label="💾 下载HTML报告",
                                        data=html_content,
                                        file_name=filename,
                                        mime="text/html",
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.error(f"PDF生成失败: {str(e)}，将使用HTML格式")
                                    html_content, filename = generate_html_report(report_data, include_charts=include_charts)
                                    st.download_button(
                                        label="💾 下载HTML报告",
                                        data=html_content,
                                        file_name=filename,
                                        mime="text/html",
                                        use_container_width=True,
                                    )
                        except Exception as e:
                            st.error(f"报告生成失败: {str(e)}")
            
            st.info("📋 报告包含: 工艺流程配置、进水水质、ASM1参数、稳态求解结果、各池浓度分布、污泥产量、能耗分析、优化建议")


def page_process_comparison():
    """工艺对比页面"""
    st.title("🔍 工艺方案对比")
    st.markdown("---")
    
    st.info("💡 **操作说明**: 分别配置两套工艺方案，点击运行求解后，系统将自动对比两套方案的出水水质、污泥产量和能耗。")
    
    col_scheme1, col_scheme2 = st.columns(2)
    
    with col_scheme1:
        st.markdown("### 🟦 方案 1")
        scheme1_name = st.text_input("方案1名称", value=st.session_state.comparison_name1, key="s1_name")
        st.session_state.comparison_name1 = scheme1_name
        
        st.markdown("**快速配置**")
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            if st.button("📋 复制当前工艺", key="copy_s1", use_container_width=True):
                st.session_state.comparison_pfs1 = copy.deepcopy(st.session_state.pfs)
                st.session_state.comparison_influent1 = copy.deepcopy(st.session_state.influent)
                st.session_state.comparison_params1 = copy.deepcopy(st.session_state.asm1_params)
                st.success("已复制当前工艺配置到方案1")
        
        with col1_2:
            template1 = st.selectbox(
                "工艺模板",
                ["A2O", "SBR", "MBR"],
                key="template_s1",
                label_visibility="collapsed",
            )
            if st.button("🔄 加载模板", key="load_s1", use_container_width=True):
                from src.process_templates import create_process_by_name
                st.session_state.comparison_pfs1 = create_process_by_name(template1)
                st.session_state.comparison_influent1 = copy.deepcopy(st.session_state.influent)
                st.session_state.comparison_params1 = copy.deepcopy(st.session_state.asm1_params)
                st.success(f"已加载{template1}工艺到方案1")
        
        if st.session_state.comparison_pfs1 is not None:
            with st.expander("⚙️ 方案1参数调整", expanded=False):
                st.markdown("**好氧池DO设定**")
                do1 = st.slider("DO (mg/L)", 0.5, 5.0, 2.0, 0.1, key="do_s1")
                
                st.markdown("**污泥停留时间SRT**")
                srt1 = st.slider("SRT (天)", 3, 40, 15, 1, key="srt_s1")
                
                st.markdown("**内回流比**")
                irr1 = st.slider("内回流比 (%)", 0, 500, 200, 50, key="irr_s1")
                
                st.markdown("**好氧池HRT**")
                hrt1 = st.slider("好氧池HRT (小时)", 2, 24, 8, 1, key="hrt_s1")
                
                if st.button("应用参数", key="apply_s1", use_container_width=True):
                    for reactor in st.session_state.comparison_pfs1.reactors:
                        if reactor.reactor_type == ReactorType.AEROBIC:
                            reactor.operation.DO_setpoint = do1
                        if hasattr(reactor.operation, 'SRT') and reactor.operation.SRT > 0:
                            reactor.operation.SRT = srt1
                        if hasattr(reactor.operation, 'internal_return_ratio'):
                            reactor.operation.internal_return_ratio = irr1 / 100.0
                        if reactor.reactor_type == ReactorType.AEROBIC:
                            V = st.session_state.comparison_influent1.Q_base * hrt1 / 24
                            reactor.geometry.volume = V
                            reactor.operation.HRT = hrt1
                    st.success("参数已应用到方案1")
            
            if st.session_state.comparison_pfs1 is not None:
                pfs1 = st.session_state.comparison_pfs1
                st.info(f"""
                **方案1配置摘要**:
                - 处理单元: {len(pfs1.reactors)} 个
                - 总容积: {pfs1.get_volume():.0f} m³
                - 工艺类型: {template1 if 'template1' in locals() else '自定义'}
                """)
    
    with col_scheme2:
        st.markdown("### 🟧 方案 2")
        scheme2_name = st.text_input("方案2名称", value=st.session_state.comparison_name2, key="s2_name")
        st.session_state.comparison_name2 = scheme2_name
        
        st.markdown("**快速配置**")
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            if st.button("📋 复制当前工艺", key="copy_s2", use_container_width=True):
                st.session_state.comparison_pfs2 = copy.deepcopy(st.session_state.pfs)
                st.session_state.comparison_influent2 = copy.deepcopy(st.session_state.influent)
                st.session_state.comparison_params2 = copy.deepcopy(st.session_state.asm1_params)
                st.success("已复制当前工艺配置到方案2")
        
        with col2_2:
            template2 = st.selectbox(
                "工艺模板",
                ["A2O", "SBR", "MBR"],
                index=0,
                key="template_s2",
                label_visibility="collapsed",
            )
            if st.button("🔄 加载模板", key="load_s2", use_container_width=True):
                from src.process_templates import create_process_by_name
                st.session_state.comparison_pfs2 = create_process_by_name(template2)
                st.session_state.comparison_influent2 = copy.deepcopy(st.session_state.influent)
                st.session_state.comparison_params2 = copy.deepcopy(st.session_state.asm1_params)
                st.success(f"已加载{template2}工艺到方案2")
        
        if st.session_state.comparison_pfs2 is not None:
            with st.expander("⚙️ 方案2参数调整", expanded=False):
                st.markdown("**好氧池DO设定**")
                do2 = st.slider("DO (mg/L)", 0.5, 5.0, 3.0, 0.1, key="do_s2")
                
                st.markdown("**污泥停留时间SRT**")
                srt2 = st.slider("SRT (天)", 3, 40, 20, 1, key="srt_s2")
                
                st.markdown("**内回流比**")
                irr2 = st.slider("内回流比 (%)", 0, 500, 250, 50, key="irr_s2")
                
                st.markdown("**好氧池HRT**")
                hrt2 = st.slider("好氧池HRT (小时)", 2, 24, 10, 1, key="hrt_s2")
                
                if st.button("应用参数", key="apply_s2", use_container_width=True):
                    for reactor in st.session_state.comparison_pfs2.reactors:
                        if reactor.reactor_type == ReactorType.AEROBIC:
                            reactor.operation.DO_setpoint = do2
                        if hasattr(reactor.operation, 'SRT') and reactor.operation.SRT > 0:
                            reactor.operation.SRT = srt2
                        if hasattr(reactor.operation, 'internal_return_ratio'):
                            reactor.operation.internal_return_ratio = irr2 / 100.0
                        if reactor.reactor_type == ReactorType.AEROBIC:
                            V = st.session_state.comparison_influent2.Q_base * hrt2 / 24
                            reactor.geometry.volume = V
                            reactor.operation.HRT = hrt2
                    st.success("参数已应用到方案2")
            
            if st.session_state.comparison_pfs2 is not None:
                pfs2 = st.session_state.comparison_pfs2
                st.info(f"""
                **方案2配置摘要**:
                - 处理单元: {len(pfs2.reactors)} 个
                - 总容积: {pfs2.get_volume():.0f} m³
                - 工艺类型: {template2 if 'template2' in locals() else '自定义'}
                """)
    
    st.markdown("---")
    
    col_ready1, col_ready2, col_run = st.columns([1, 1, 1])
    
    ready1 = st.session_state.comparison_pfs1 is not None and st.session_state.comparison_influent1 is not None
    ready2 = st.session_state.comparison_pfs2 is not None and st.session_state.comparison_influent2 is not None
    
    with col_ready1:
        if ready1:
            st.success(f"✅ {scheme1_name} 已就绪")
        else:
            st.warning(f"⚠️ {scheme1_name} 未配置")
    
    with col_ready2:
        if ready2:
            st.success(f"✅ {scheme2_name} 已就绪")
        else:
            st.warning(f"⚠️ {scheme2_name} 未配置")
    
    with col_run:
        if st.button("🚀 运行双方案对比求解", 
                    type="primary", 
                    use_container_width=True,
                    disabled=not (ready1 and ready2)):
            with st.spinner("正在并行求解两套工艺方案..."):
                result1 = solve_steady_state(
                    st.session_state.comparison_pfs1,
                    st.session_state.comparison_influent1,
                    st.session_state.comparison_params1,
                    st.session_state.solver_config,
                )
                
                result2 = solve_steady_state(
                    st.session_state.comparison_pfs2,
                    st.session_state.comparison_influent2,
                    st.session_state.comparison_params2,
                    st.session_state.solver_config,
                )
                
                compliance1 = check_compliance(result1.effluent_quality, st.session_state.selected_standard) if result1.converged else None
                compliance2 = check_compliance(result2.effluent_quality, st.session_state.selected_standard) if result2.converged else None
                
                sludge1 = calculate_sludge_production(
                    st.session_state.comparison_pfs1,
                    result1.reactor_states,
                    st.session_state.comparison_influent1.Q_base,
                    st.session_state.comparison_params1,
                ) if result1.converged else None
                
                sludge2 = calculate_sludge_production(
                    st.session_state.comparison_pfs2,
                    result2.reactor_states,
                    st.session_state.comparison_influent2.Q_base,
                    st.session_state.comparison_params2,
                ) if result2.converged else None
                
                energy1 = calculate_energy_consumption(
                    st.session_state.comparison_pfs1,
                    result1.reactor_states,
                    st.session_state.comparison_influent1,
                    st.session_state.comparison_params1,
                ) if result1.converged else None
                
                energy2 = calculate_energy_consumption(
                    st.session_state.comparison_pfs2,
                    result2.reactor_states,
                    st.session_state.comparison_influent2,
                    st.session_state.comparison_params2,
                ) if result2.converged else None
                
                st.session_state.comparison_result = ProcessComparisonResult(
                    name1=scheme1_name,
                    name2=scheme2_name,
                    result1=result1,
                    result2=result2,
                    compliance1=compliance1,
                    compliance2=compliance2,
                    sludge1=sludge1,
                    sludge2=sludge2,
                    energy1=energy1,
                    energy2=energy2,
                )
                
                st.success("对比求解完成！")
    
    if st.session_state.comparison_result is not None:
        comp = st.session_state.comparison_result
        
        st.markdown("---")
        st.subheader("📊 对比结果")
        
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            if comp.result1 and comp.result1.converged:
                st.success(f"✅ {comp.name1}: 收敛")
            else:
                st.error(f"❌ {comp.name1}: 未收敛")
        
        with col_status2:
            if comp.result2 and comp.result2.converged:
                st.success(f"✅ {comp.name2}: 收敛")
            else:
                st.error(f"❌ {comp.name2}: 未收敛")
        
        if comp.result1.converged and comp.result2.converged:
            st.markdown("### 📈 出水水质对比")
            
            fig_comp = plot_process_comparison(comp)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown("### 📋 详细对比表")
            comp_df = comp.get_comparison_table()
            
            def highlight_better(row):
                styles = [''] * len(row)
                try:
                    val1 = float(str(row['方案1 (mg/L)']))
                    val2 = float(str(row['方案2 (mg/L)']))
                    if val2 < val1:
                        styles[2] = 'background-color: #d4edda'
                        styles[3] = 'background-color: #d4edda'
                        styles[4] = 'background-color: #d4edda'
                    elif val1 < val2:
                        styles[1] = 'background-color: #d4edda'
                except:
                    pass
                return styles
            
            st.dataframe(
                comp_df.style.apply(highlight_better, axis=1),
                hide_index=True,
                use_container_width=True,
            )
            
            st.markdown("### 📦 污泥产量对比")
            if comp.sludge1 is not None and comp.sludge2 is not None:
                col_sl1, col_sl2, col_sl3 = st.columns(3)
                with col_sl1:
                    st.metric(f"{comp.name1} 产泥量", 
                              f"{comp.sludge1.daily_sludge_kg:.1f} kg/d",
                              f"{comp.sludge1.MLSS_gL:.2f} g/L MLSS")
                with col_sl2:
                    st.metric(f"{comp.name2} 产泥量", 
                              f"{comp.sludge2.daily_sludge_kg:.1f} kg/d",
                              f"{comp.sludge2.MLSS_gL:.2f} g/L MLSS")
                with col_sl3:
                    diff = comp.sludge2.daily_sludge_kg - comp.sludge1.daily_sludge_kg
                    pct = diff / comp.sludge1.daily_sludge_kg * 100 if comp.sludge1.daily_sludge_kg > 0 else 0
                    st.metric("产泥量差异", 
                              f"{diff:+.1f} kg/d",
                              f"{pct:+.1f}%")
            
            st.markdown("### ⚡ 能耗对比")
            if comp.energy1 is not None and comp.energy2 is not None:
                col_en1, col_en2, col_en3 = st.columns(3)
                with col_en1:
                    st.metric(f"{comp.name1} 电耗", 
                              f"{comp.energy1.total_kwh_d:.1f} kWh/d",
                              f"{comp.energy1.unit_kwh_m3:.4f} kWh/m³")
                with col_en2:
                    st.metric(f"{comp.name2} 电耗", 
                              f"{comp.energy2.total_kwh_d:.1f} kWh/d",
                              f"{comp.energy2.unit_kwh_m3:.4f} kWh/m³")
                with col_en3:
                    diff = comp.energy2.total_kwh_d - comp.energy1.total_kwh_d
                    pct = diff / comp.energy1.total_kwh_d * 100 if comp.energy1.total_kwh_d > 0 else 0
                    st.metric("电耗差异", 
                              f"{diff:+.1f} kWh/d",
                              f"{pct:+.1f}%")
                
                col_pie1, col_pie2 = st.columns(2)
                with col_pie1:
                    st.markdown(f"**{comp.name1} 能耗分项**")
                    fig1 = plot_energy_pie(comp.energy1)
                    st.plotly_chart(fig1, use_container_width=True)
                with col_pie2:
                    st.markdown(f"**{comp.name2} 能耗分项**")
                    fig2 = plot_energy_pie(comp.energy2)
                    st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("### ✅ 达标情况对比")
            if comp.compliance1 is not None and comp.compliance2 is not None:
                col_comp1, col_comp2 = st.columns(2)
                with col_comp1:
                    st.markdown(f"**{comp.name1}**")
                    if comp.compliance1.overall_compliant:
                        st.success("🎉 全面达标")
                    else:
                        fail_items = [item.name for item in comp.compliance1.items if not item.compliant]
                        st.error(f"⚠️ 超标指标: {', '.join(fail_items)}")
                
                with col_comp2:
                    st.markdown(f"**{comp.name2}**")
                    if comp.compliance2.overall_compliant:
                        st.success("🎉 全面达标")
                    else:
                        fail_items = [item.name for item in comp.compliance2.items if not item.compliant]
                        st.error(f"⚠️ 超标指标: {', '.join(fail_items)}")
            
            st.info("💡 **建议**: 综合考虑出水水质、污泥产量和能耗，选择最优方案。通常提高DO和延长SRT可改善出水水质，但会增加能耗和降低污泥产量。")


def page_dynamic():
    """动态仿真页面"""
    st.title("📈 动态仿真")
    st.markdown("---")
    
    config = st.session_state.solver_config
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("仿真配置")
        sim_days = st.slider(
            "仿真时长 (天)",
            min_value=1, max_value=30,
            value=int(config.simulation_days),
            step=1,
        )
        output_interval = st.number_input(
            "输出间隔 (天)",
            min_value=0.001, max_value=0.1,
            value=float(config.output_interval_days),
            step=0.005,
            format="%.3f",
        )
        
        config.simulation_days = sim_days
        config.output_interval_days = output_interval
    
    with col2:
        st.subheader("数值方法")
        method = st.selectbox(
            "ODE求解方法",
            ['BDF', 'RK45', 'Radau'],
            index=0,
        )
        config.method = method
        
        rtol = st.number_input(
            "相对误差容限",
            min_value=1e-8, max_value=1e-4,
            value=float(config.rtol),
            step=1e-7,
            format="%.1e",
        )
        atol = st.number_input(
            "绝对误差容限",
            min_value=1e-10, max_value=1e-6,
            value=float(config.atol),
            step=1e-9,
            format="%.1e",
        )
        config.rtol = rtol
        config.atol = atol
    
    st.markdown("---")
    
    if st.session_state.influent.flow_mode == 'constant':
        st.info("💡 当前进水为恒定模式，建议切换到日变化模式以观察动态响应")
    
    col_run, col_pause, col_stop = st.columns(3)
    
    with col_run:
        if st.button("▶️ 开始仿真", type="primary", use_container_width=True,
                    disabled=st.session_state.simulation_running and not st.session_state.simulation_paused):
            if not st.session_state.simulation_running:
                st.session_state.simulation_running = True
                st.session_state.simulation_paused = False
                st.session_state.simulation_aborted = False
                st.session_state.simulation_progress = 0.0
                
                initial_states = None
                if st.session_state.steady_result is not None and st.session_state.steady_result.converged:
                    initial_states = st.session_state.steady_result.reactor_states
                    st.info("使用稳态解作为初始条件")
                
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                chart_placeholder = st.empty()
                
                def stop_check():
                    return st.session_state.simulation_aborted
                
                def pause_check():
                    return st.session_state.simulation_paused
                
                def progress_callback(progress):
                    st.session_state.simulation_progress = progress
                    progress_bar.progress(progress)
                    status_text.text(f"正在仿真... {progress*100:.1f}%")
                
                status_text.text("正在仿真...")
                with st.spinner("动态仿真进行中... (可暂停或终止)"):
                    simulator = DynamicSimulator(
                        st.session_state.pfs,
                        st.session_state.influent,
                        st.session_state.asm1_params,
                        config
                    )
                    result = simulator.run(
                        initial_states=initial_states,
                        progress_callback=progress_callback,
                        stop_check_callback=stop_check,
                        pause_check_callback=pause_check,
                        segment_days=0.5
                    )
                    st.session_state.dynamic_result = result
                    progress_bar.progress(1.0)
                
                st.session_state.simulation_running = False
                st.session_state.simulation_paused = False
                
                if result.success:
                    if result.was_aborted:
                        st.warning(f"仿真已终止: {result.message}")
                    else:
                        st.success("仿真完成！")
                else:
                    st.error(f"仿真失败: {result.message}")
    
    with col_pause:
        if st.session_state.simulation_running:
            if st.session_state.simulation_paused:
                if st.button("▶️ 继续", type="secondary", use_container_width=True):
                    st.session_state.simulation_paused = False
                    st.rerun()
            else:
                if st.button("⏸️ 暂停", type="secondary", use_container_width=True):
                    st.session_state.simulation_paused = True
                    st.rerun()
        else:
            st.button("⏸️ 暂停", disabled=True, use_container_width=True)
    
    with col_stop:
        if st.button("⏹️ 终止", type="secondary", use_container_width=True,
                    disabled=not st.session_state.simulation_running):
            st.session_state.simulation_aborted = True
            st.session_state.simulation_running = False
            st.session_state.simulation_paused = False
            if st.session_state.dynamic_result is not None:
                if len(st.session_state.dynamic_result.time_days) > 0:
                    st.warning("已发送终止信号，当前仿真段完成后将停止")
            st.rerun()
    
    if st.session_state.simulation_running:
        if st.session_state.simulation_paused:
            st.info(f"⏸️ 仿真已暂停 ({st.session_state.simulation_progress*100:.1f}%) - 点击继续按钮恢复")
        else:
            st.info(f"🔄 仿真进行中 ({st.session_state.simulation_progress*100:.1f}%) - 可点击暂停或终止按钮")
    
    if st.session_state.simulation_aborted and st.session_state.dynamic_result is not None:
        if hasattr(st.session_state.dynamic_result, 'was_aborted') and st.session_state.dynamic_result.was_aborted:
            st.warning(f"仿真已被用户终止，已完成 {len(st.session_state.dynamic_result.time_days)} 个时间步")
    
    if st.session_state.dynamic_result is not None:
        result = st.session_state.dynamic_result
        
        st.markdown("---")
        st.subheader("仿真结果")
        
        if not result.success:
            st.error(f"仿真失败: {result.message}")
        else:
            st.info(f"仿真时长: {result.time_days[-1]:.1f} 天 | "
                    f"时间点数: {len(result.time_days)}")
            
            fig = plot_effluent_timeseries(result)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("时间轴查看")
            
            t_idx = st.slider(
                "选择时间点",
                min_value=0, max_value=len(result.time_days)-1,
                value=len(result.time_days)-1,
                step=1,
                format="%d",
                label_visibility="collapsed",
            )
            
            t_day = result.time_days[t_idx]
            t_hour = (t_day % 1) * 24
            
            st.info(f"**第 {t_day:.1f} 天 ({int(t_hour):02d}:{int((t_hour%1)*60):02d})**")
            
            effluent = result.effluent_quality_history[t_idx]
            
            col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
            metrics = [
                ('COD', 'mg/L', effluent['COD']),
                ('BOD5', 'mg/L', effluent['BOD5']),
                ('NH3-N', 'mg/L', effluent['NH3_N']),
                ('TN', 'mg/L', effluent['TN']),
                ('TP', 'mg/L', effluent['TP']),
                ('SS', 'mg/L', effluent['SS']),
            ]
            cols = [col_m1, col_m2, col_m3, col_m4, col_m5, col_m6]
            for i, (name, unit, val) in enumerate(metrics):
                with cols[i]:
                    st.metric(name, f"{val:.2f}", unit)
            
            st.markdown("**各池浓度分布**")
            reactor_states_at_t = []
            for rs in result.reactor_states:
                reactor_states_at_t.append(rs[t_idx, :])
            
            fig_stack = plot_reactor_stack(st.session_state.pfs, reactor_states_at_t)
            st.plotly_chart(fig_stack, use_container_width=True)
            
            st.markdown("---")
            st.subheader("统计数据")
            
            effluent_history = result.effluent_quality_history
            stats_data = []
            for key in ['COD', 'BOD5', 'NH3_N', 'TN', 'TP', 'SS']:
                values = [h[key] for h in effluent_history]
                name_map = {'NH3_N': 'NH3-N'}
                display_name = name_map.get(key, key)
                stats_data.append({
                    '指标': display_name,
                    '最小值': min(values),
                    '最大值': max(values),
                    '平均值': np.mean(values),
                    '标准差': np.std(values),
                })
            
            st.dataframe(pd.DataFrame(stats_data), hide_index=True, use_container_width=True)


def page_sensitivity():
    """敏感性分析页面"""
    st.title("📊 参数敏感性分析")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("分析类型")
        analysis_type = st.radio(
            "选择分析类型",
            ["单因素分析", "双因素分析"],
            index=0,
        )
    
    with col2:
        st.subheader("参数选择")
        param_options = [(v.display_name, k) for k, v in SENSITIVITY_PARAMETERS.items()]
        
        if analysis_type == "单因素分析":
            param1_display = st.selectbox(
                "扫描参数",
                [p[0] for p in param_options],
                index=0,
            )
            param1_key = [p[1] for p in param_options if p[0] == param1_display][0]
            
            param_info = SENSITIVITY_PARAMETERS[param1_key]
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                min_val = st.number_input(
                    "最小值",
                    value=float(param_info.min_value),
                    step=1.0,
                )
            with col_p2:
                max_val = st.number_input(
                    "最大值",
                    value=float(param_info.max_value),
                    step=1.0,
                )
            with col_p3:
                steps = st.number_input(
                    "扫描步数",
                    min_value=3, max_value=20,
                    value=10,
                    step=1,
                )
            
            param_values = np.linspace(min_val, max_val, int(steps))
            
            st.info(f"**{param_info.display_name}**\n\n"
                    f"范围: {min_val} - {max_val} {param_info.unit}\n"
                    f"步数: {steps}")
            
            if st.button("▶️ 运行单因素分析", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_results = []
                all_converged = []
                
                for i, val in enumerate(param_values):
                    progress_bar.progress((i + 1) / len(param_values))
                    status_text.text(f"计算 {i+1}/{len(param_values)}: {param_info.display_name} = {val:.2f}")
                    
                    pfs_copy = copy.deepcopy(st.session_state.pfs)
                    influent_copy = copy.deepcopy(st.session_state.influent)
                    params_copy = copy.deepcopy(st.session_state.asm1_params)
                    
                    from src.analysis import run_sensitivity_analysis
                    single_result = run_sensitivity_analysis(
                        pfs_copy,
                        influent_copy,
                        params_copy,
                        param1_key,
                        [val],
                        st.session_state.solver_config,
                    )
                    
                    all_results.append(single_result.effluent_results[0])
                    all_converged.append(single_result.converged_list[0])
                
                from src.analysis import SensitivityResult
                st.session_state.sensitivity_result = SensitivityResult(
                    parameter_name=param1_key,
                    parameter_values=list(param_values),
                    effluent_results=all_results,
                    converged_list=all_converged,
                )
                
                status_text.text("分析完成！")
                progress_bar.progress(1.0)
        
        else:
            param1_display = st.selectbox(
                "参数1 (X轴)",
                [p[0] for p in param_options],
                index=0,
            )
            param1_key = [p[1] for p in param_options if p[0] == param1_display][0]
            
            param2_display = st.selectbox(
                "参数2 (Y轴)",
                [p[0] for p in param_options],
                index=2,
            )
            param2_key = [p[1] for p in param_options if p[0] == param2_display][0]
            
            p1_info = SENSITIVITY_PARAMETERS[param1_key]
            p2_info = SENSITIVITY_PARAMETERS[param2_key]
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                p1_min = st.number_input(f"{p1_info.display_name} 最小值", value=float(p1_info.min_value), key="p1_min")
                p1_max = st.number_input(f"{p1_info.display_name} 最大值", value=float(p1_info.max_value), key="p1_max")
                p1_steps = st.number_input(f"{p1_info.display_name} 步数", min_value=3, max_value=10, value=5, key="p1_steps")
            with col_s2:
                p2_min = st.number_input(f"{p2_info.display_name} 最小值", value=float(p2_info.min_value), key="p2_min")
                p2_max = st.number_input(f"{p2_info.display_name} 最大值", value=float(p2_info.max_value), key="p2_max")
                p2_steps = st.number_input(f"{p2_info.display_name} 步数", min_value=3, max_value=10, value=5, key="p2_steps")
            
            p1_values = np.linspace(p1_min, p1_max, int(p1_steps))
            p2_values = np.linspace(p2_min, p2_max, int(p2_steps))
            
            heatmap_indicator = st.selectbox(
                "热力图指标",
                ['NH3_N', 'COD', 'TN', 'TP', 'SS'],
                index=0,
                format_func=lambda x: {'NH3_N': 'NH3-N'}.get(x, x),
            )
            
            if st.button("▶️ 运行双因素分析", type="primary", use_container_width=True):
                total_calcs = len(p1_values) * len(p2_values)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                result_matrix = []
                converged_matrix = []
                count = 0
                
                for val1 in p1_values:
                    row_results = []
                    row_converged = []
                    
                    for val2 in p2_values:
                        count += 1
                        progress_bar.progress(count / total_calcs)
                        status_text.text(f"计算 {count}/{total_calcs}: "
                                         f"{p1_info.display_name}={val1:.2f}, "
                                         f"{p2_info.display_name}={val2:.2f}")
                        
                        pfs_copy = copy.deepcopy(st.session_state.pfs)
                        influent_copy = copy.deepcopy(st.session_state.influent)
                        params_copy = copy.deepcopy(st.session_state.asm1_params)
                        
                        from src.analysis import run_sensitivity_analysis
                        result = run_sensitivity_analysis(
                            pfs_copy, influent_copy, params_copy,
                            param1_key, [val1],
                            st.session_state.solver_config,
                        )
                        
                        if result.converged_list[0]:
                            row_results.append(result.effluent_results[0])
                        else:
                            row_results.append({k: np.nan for k in result.effluent_results[0]})
                        row_converged.append(result.converged_list[0])
                    
                    result_matrix.append(row_results)
                    converged_matrix.append(row_converged)
                
                from src.analysis import TwoFactorSensitivityResult
                st.session_state.two_factor_result = TwoFactorSensitivityResult(
                    param1_name=param1_key,
                    param2_name=param2_key,
                    param1_values=list(p1_values),
                    param2_values=list(p2_values),
                    results_matrix=result_matrix,
                    converged_matrix=converged_matrix,
                )
                
                status_text.text("分析完成！")
                progress_bar.progress(1.0)
    
    st.markdown("---")
    
    if analysis_type == "单因素分析" and st.session_state.sensitivity_result is not None:
        result = st.session_state.sensitivity_result
        
        st.subheader("单因素敏感性分析结果")
        fig = plot_sensitivity_curves(result)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("详细数据")
        df = result.to_dataframe()
        st.dataframe(df, hide_index=True, use_container_width=True)
    
    if analysis_type == "双因素分析" and st.session_state.two_factor_result is not None:
        result = st.session_state.two_factor_result
        
        st.subheader("双因素敏感性分析结果")
        
        heatmap_indicator = st.selectbox(
            "显示指标",
            ['NH3_N', 'COD', 'TN', 'TP', 'SS'],
            index=0,
            key="heatmap_indicator_select",
            format_func=lambda x: {'NH3_N': 'NH3-N'}.get(x, x),
        )
        
        fig = plot_two_factor_heatmap(result, heatmap_indicator)
        st.plotly_chart(fig, use_container_width=True)


def page_optimization():
    """优化建议页面"""
    st.title("💡 工艺优化建议")
    st.markdown("---")
    
    if st.session_state.steady_result is None or not st.session_state.steady_result.converged:
        st.warning("请先运行稳态求解并获得收敛结果")
        return
    
    if st.session_state.compliance_result is None:
        st.warning("请先运行稳态求解")
        return
    
    if st.session_state.optimization_suggestions is None:
        st.session_state.optimization_suggestions = generate_optimization_suggestions(
            st.session_state.pfs,
            st.session_state.influent,
            st.session_state.asm1_params,
            st.session_state.compliance_result,
        )
    
    suggestions = st.session_state.optimization_suggestions
    
    if st.session_state.compliance_result.overall_compliant:
        st.success("🎉 当前运行条件下出水水质全面达标！")
    else:
        st.error("⚠️ 当前运行条件下存在超标指标，建议参考以下优化方案")
    
    st.markdown("---")
    
    if len(suggestions) == 0:
        st.info("当前无需优化")
    else:
        priority_colors = {
            1: ("🔴", "#fef0f0"),
            2: ("🟠", "#fff7e6"),
            3: ("🟡", "#ffffe0"),
            4: ("🟢", "#f0fff4"),
        }
        
        for i, sug in enumerate(suggestions):
            icon, bg_color = priority_colors.get(sug.priority, ("⚪", "#f5f5f5"))
            
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 10px;
                        border-left: 5px solid {'#ff4d4f' if sug.priority==1 else '#faad14' if sug.priority==2 else '#fadb14' if sug.priority==3 else '#52c41a'};
                    ">
                        <h4 style="margin-top: 0;">{icon} 优先级 {sug.priority}: {sug.title}</h4>
                        <p style="margin: 5px 0;"><strong>问题描述:</strong> {sug.description}</p>
                        <p style="margin: 5px 0;"><strong>当前值:</strong> {sug.current_value}</p>
                        <p style="margin: 5px 0;"><strong>建议值:</strong> {sug.suggested_value}</p>
                        <p style="margin: 5px 0;"><strong>预期效果:</strong> {sug.expected_effect}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
    st.markdown("---")
    st.subheader("当前工况总结")
    
    pfs = st.session_state.pfs
    comp = st.session_state.compliance_result
    
    col_sum1, col_sum2 = st.columns(2)
    
    with col_sum1:
        st.markdown("**工艺流程**")
        for reactor in pfs.reactors:
            st.write(f"- {reactor.get_icon()} {reactor.name}: "
                     f"V={reactor.geometry.volume:.0f}m³, "
                     f"HRT={reactor.operation.HRT:.1f}h, "
                     f"SRT={reactor.operation.SRT:.0f}d"
                     f"{', DO=' + str(reactor.operation.DO_setpoint) if reactor.reactor_type == ReactorType.AEROBIC else ''}")
    
    with col_sum2:
        st.markdown("**运行参数**")
        st.write(f"进水流量: {st.session_state.influent.Q_base:.0f} m³/d")
        st.write(f"水温: {st.session_state.asm1_params.temperature:.1f} °C")
        
        for reactor in pfs.reactors:
            if reactor.operation.return_sludge_ratio > 0:
                st.write(f"{reactor.name} 回流比: {reactor.operation.return_sludge_ratio*100:.0f}%")
            if hasattr(reactor.operation, 'internal_return_ratio') and reactor.operation.internal_return_ratio > 0:
                st.write(f"{reactor.name} 内回流比: {reactor.operation.internal_return_ratio*100:.0f}%")
    
    st.markdown("---")
    st.subheader("出水水质")
    
    effluent = st.session_state.steady_result.effluent_quality
    std = STANDARDS[st.session_state.selected_standard]
    
    summary_data = []
    for item in comp.items:
        summary_data.append({
            '指标': item.name,
            '出水值': item.value,
            '标准限值': item.limit,
            '超标量': max(0, item.value - item.limit),
            '达标裕度': item.limit - item.value,
            '占标率(%)': item.ratio * 100,
        })
    
    st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)


def page_multiobjective_optimization():
    """多目标工艺优化页面"""
    st.title("🎯 多目标工艺优化 (NSGA-II)")
    st.markdown("---")
    
    opt_config = st.session_state.optimization_config
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ 决策变量配置")
        st.info("选择需要优化的决策变量及其取值范围")
        
        for i, var in enumerate(DEFAULT_VARIABLES):
            with st.expander(f"📌 {var.display_name}", expanded=True):
                col_min, col_max, col_enable = st.columns([1, 1, 1])
                
                with col_min:
                    min_val = st.number_input(
                        "最小值",
                        min_value=var.min_value * 0.5,
                        max_value=var.max_value,
                        value=var.min_value,
                        step=0.1 if var.unit == 'mg/L' else 1.0,
                        key=f"opt_min_{i}",
                    )
                
                with col_max:
                    max_val = st.number_input(
                        "最大值",
                        min_value=var.min_value,
                        max_value=var.max_value * 1.5,
                        value=var.max_value,
                        step=0.1 if var.unit == 'mg/L' else 1.0,
                        key=f"opt_max_{i}",
                    )
                
                with col_enable:
                    enabled = st.checkbox(
                        "启用优化",
                        value=True,
                        key=f"opt_enable_{i}",
                    )
                
                if enabled:
                    var.min_value = float(min_val)
                    var.max_value = float(max_val)
                    if var not in opt_config.variables:
                        opt_config.variables.append(var)
                else:
                    if var in opt_config.variables:
                        opt_config.variables.remove(var)
                
                st.caption(f"单位: {var.unit} | {var.description}")
    
    with col2:
        st.subheader("🎯 优化目标配置")
        
        opt_mode = st.radio(
            "优化模式",
            ["Pareto多目标优化", "加权单目标优化"],
            index=0 if opt_config.use_pareto else 1,
            horizontal=True,
        )
        opt_config.use_pareto = opt_mode == "Pareto多目标优化"
        
        st.markdown("**选择优化目标及权重**")
        st.caption("权重仅在加权模式下生效，用于计算综合评分")
        
        weights_total = 0.0
        for i, obj in enumerate(DEFAULT_OBJECTIVES):
            col_name, col_dir, col_weight = st.columns([2, 1, 1])
            
            with col_name:
                selected = st.checkbox(
                    f"{obj.display_name} ({obj.unit})",
                    value=True,
                    key=f"opt_obj_{i}",
                )
            
            with col_dir:
                direction = st.selectbox(
                    "方向",
                    ["最小化", "最大化"],
                    index=0,
                    key=f"opt_dir_{i}",
                    label_visibility="collapsed",
                )
                obj.direction = 'minimize' if direction == "最小化" else 'maximize'
            
            with col_weight:
                weight = st.slider(
                    "权重",
                    min_value=0.0,
                    max_value=1.0,
                    value=opt_config.objective_weights.get(obj.name, 0.25),
                    step=0.05,
                    key=f"opt_weight_{i}",
                    label_visibility="collapsed",
                )
                opt_config.objective_weights[obj.name] = weight
                weights_total += weight
            
            if selected:
                if obj not in opt_config.objectives:
                    opt_config.objectives.append(obj)
            else:
                if obj in opt_config.objectives:
                    opt_config.objectives.remove(obj)
        
        if not opt_config.use_pareto:
            st.info(f"当前权重合计: {weights_total:.2f}" + 
                   (" (建议归一化到1.0)" if abs(weights_total - 1.0) > 0.01 else " ✓"))
    
    st.markdown("---")
    
    col_params1, col_params2 = st.columns([1, 1])
    
    with col_params1:
        st.subheader("🔬 算法参数")
        
        pop_size = st.number_input(
            "种群规模",
            min_value=10,
            max_value=200,
            value=opt_config.population_size,
            step=10,
            help="每代中的个体数量，越大搜索越充分但计算越慢",
        )
        
        max_gen = st.number_input(
            "迭代代数",
            min_value=10,
            max_value=500,
            value=opt_config.max_generations,
            step=10,
            help="遗传算法迭代次数",
        )
        
        opt_config.population_size = int(pop_size)
        opt_config.max_generations = int(max_gen)
    
    with col_params2:
        st.subheader("📋 约束条件 (一级A标准)")
        
        standard = STANDARDS['一级A']
        st.info(f"""
        **强制约束:**
        - 出水 COD ≤ {standard.COD} mg/L
        - 出水 NH3-N ≤ {standard.NH3_N} mg/L
        - 出水 TN ≤ {standard.TN} mg/L
        - 出水 TP ≤ {standard.TP} mg/L
        
        不满足约束的个体将在排序时给予惩罚
        """)
    
    st.markdown("---")
    
    col_run, col_stop, col_status = st.columns([1, 1, 2])
    
    with col_run:
        if st.button("🚀 开始优化", 
                    type="primary", 
                    use_container_width=True,
                    disabled=st.session_state.optimization_running):
            
            if len(opt_config.variables) == 0:
                st.error("请至少选择一个决策变量！")
            elif len(opt_config.objectives) == 0:
                st.error("请至少选择一个优化目标！")
            else:
                opt_config_copy = copy.deepcopy(opt_config)
                opt_config_copy.population_size = int(pop_size)
                opt_config_copy.max_generations = int(max_gen)
                
                st.session_state.optimization_running = True
                st.session_state.optimization_aborted = False
                st.session_state.optimization_progress = 0.0
                st.session_state.optimization_result = None
                st.session_state.optimization_current_gen = 0
                st.session_state.optimization_max_gen = opt_config_copy.max_generations
                st.session_state.optimization_best_fitness = 0.0
                st.session_state.optimization_avg_fitness = 0.0
                st.session_state.optimization_status = "初始化中..."
                
                optimizer = NSGA2Optimizer(
                    config=opt_config_copy,
                    pfs=st.session_state.pfs,
                    influent=st.session_state.influent,
                    asm1_params=st.session_state.asm1_params,
                    solver_config=st.session_state.solver_config,
                )
                
                st.session_state.optimization_optimizer = optimizer
                st.session_state.optimization_config_copy = opt_config_copy
                
                import threading
                
                def run_optimization():
                    def progress_callback(current_gen, max_gen, best_fitness, avg_fitness):
                        progress = current_gen / max_gen if max_gen > 0 else 0
                        st.session_state.optimization_progress = progress
                        st.session_state.optimization_current_gen = current_gen
                        st.session_state.optimization_max_gen = max_gen
                        st.session_state.optimization_best_fitness = best_fitness
                        st.session_state.optimization_avg_fitness = avg_fitness
                        st.session_state.optimization_status = f"迭代 {current_gen}/{max_gen} | 最优: {best_fitness:.2f} | 平均: {avg_fitness:.2f}"
                    
                    def stop_check():
                        return st.session_state.optimization_aborted
                    
                    try:
                        result = optimizer.optimize(
                            progress_callback=progress_callback,
                            stop_check_callback=stop_check,
                        )
                        st.session_state.optimization_result = result
                        st.session_state.optimization_status = "优化完成！"
                    except Exception as e:
                        st.session_state.optimization_error = str(e)
                        st.session_state.optimization_status = f"出错: {str(e)}"
                    finally:
                        st.session_state.optimization_running = False
                
                optimization_thread = threading.Thread(target=run_optimization, daemon=True)
                optimization_thread.start()
                st.rerun()
    
    with col_stop:
        if st.button("⏹️ 终止优化", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=not st.session_state.optimization_running):
            st.session_state.optimization_aborted = True
            if st.session_state.optimization_optimizer is not None:
                st.session_state.optimization_optimizer.stop()
            st.rerun()
    
    with col_status:
        if st.session_state.optimization_running:
            progress = st.session_state.optimization_progress
            current_gen = st.session_state.get('optimization_current_gen', 0)
            max_gen = st.session_state.get('optimization_max_gen', opt_config.max_generations)
            best_fitness = st.session_state.get('optimization_best_fitness', 0.0)
            avg_fitness = st.session_state.get('optimization_avg_fitness', 0.0)
            status = st.session_state.get('optimization_status', '初始化中...')
            
            st.progress(min(progress, 1.0))
            st.info(f"🔄 {status}")
            
            import time
            time.sleep(0.5)
            st.rerun()
        elif st.session_state.optimization_result is not None:
            result = st.session_state.optimization_result
            if result.was_aborted:
                st.warning(f"⚠️ 优化已提前终止，完成 {len(result.all_populations)-1}/{st.session_state.optimization_max_gen} 代")
            else:
                st.success(f"✅ 优化完成！共评估 {result.total_evaluations} 个方案，获得 {len(result.pareto_front)} 个Pareto最优解")
        elif 'optimization_error' in st.session_state and st.session_state.optimization_error is not None:
            st.error(f"❌ 优化出错: {st.session_state.optimization_error}")
    
    if st.session_state.optimization_running:
        st.info("💡 优化过程中每代需要对种群中每个个体调用稳态求解器进行仿真计算，"
                "种群50、迭代100代约需要5-10分钟，请耐心等待。可随时点击终止按钮停止优化。")
    
    if st.session_state.optimization_result is not None:
        result = st.session_state.optimization_result
        
        st.markdown("---")
        st.subheader("📊 优化结果")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Pareto前沿", 
            "📋 方案推荐", 
            "📉 收敛曲线", 
            "🎯 平行坐标"
        ])
        
        with tab1:
            st.markdown("### Pareto前沿散点图")
            st.caption("每个点代表一个非支配最优方案，横轴为能耗，纵轴为出水TN，颜色表示第三个目标值")
            
            col_color, _ = st.columns([1, 3])
            with col_color:
                color_by = st.selectbox(
                    "颜色映射",
                    options=[
                        ("产泥量", "sludge"),
                        ("出水NH3-N", "NH3_N"),
                        ("能耗", "energy"),
                        ("出水TN", "TN"),
                    ],
                    format_func=lambda x: x[0],
                    index=0,
                    key="opt_color_by",
                )
                st.session_state.optimization_color_by = color_by[1]
            
            fig_pareto = plot_pareto_front(
                result.pareto_front, 
                color_by=st.session_state.optimization_color_by
            )
            st.plotly_chart(fig_pareto, use_container_width=True)
            
            st.info("💡 **Pareto最优解**: 这些方案在多个目标之间达到了最优平衡，"
                   "改进其中一个目标必然会导致至少一个其他目标变差。"
                   "鼠标悬停可查看该方案的详细参数和出水指标。")
        
        with tab2:
            st.markdown("### 最优方案推荐")
            st.caption("按综合评分排序，评分越低越好。可选择方案一键回填到工艺配置。")
            
            all_individuals = []
            for pop in result.all_populations:
                all_individuals.extend(pop)
            
            unique_solutions = {}
            for ind in all_individuals:
                if ind.converged:
                    key = tuple(round(v, 4) for v in ind.variables)
                    if key not in unique_solutions:
                        unique_solutions[key] = ind
            
            scored_solutions = []
            for ind in unique_solutions.values():
                score = calculate_composite_score(ind, opt_config.objective_weights)
                scored_solutions.append((score, ind))
            
            scored_solutions.sort(key=lambda x: x[0])
            top_solutions = scored_solutions[:10]
            
            if len(top_solutions) == 0:
                st.warning("未找到有效方案")
            else:
                table_data = []
                for rank, (score, ind) in enumerate(top_solutions, 1):
                    do = ind.variables[0] if len(ind.variables) > 0 else 0
                    irr = ind.variables[1] if len(ind.variables) > 1 else 0
                    srt = ind.variables[2] if len(ind.variables) > 2 else 0
                    rr = ind.variables[3] if len(ind.variables) > 3 else 0
                    
                    cod = ind.effluent_quality.get('COD', 0)
                    nh3 = ind.effluent_quality.get('NH3_N', 0)
                    tn = ind.effluent_quality.get('TN', 0)
                    tp = ind.effluent_quality.get('TP', 0)
                    
                    energy = ind.energy_result.total_kwh_d if ind.energy_result else 0
                    sludge = ind.sludge_result.daily_sludge_kg if ind.sludge_result else 0
                    
                    compliant = "✅ 达标" if ind.is_feasible else "❌ 超标"
                    
                    table_data.append({
                        '排名': rank,
                        '综合评分': f"{score:.3f}",
                        'DO (mg/L)': f"{do:.2f}",
                        '内回流比 (%)': f"{irr:.0f}",
                        'SRT (天)': f"{srt:.1f}",
                        '回流比 (%)': f"{rr:.0f}",
                        'COD (mg/L)': f"{cod:.2f}",
                        'NH3-N (mg/L)': f"{nh3:.2f}",
                        'TN (mg/L)': f"{tn:.2f}",
                        'TP (mg/L)': f"{tp:.2f}",
                        '能耗 (kWh/d)': f"{energy:.1f}",
                        '产泥量 (kg/d)': f"{sludge:.1f}",
                        '达标情况': compliant,
                    })
                
                df = pd.DataFrame(table_data)
                
                selected_rank = st.selectbox(
                    "选择方案进行查看",
                    options=[row['排名'] for row in table_data],
                    format_func=lambda x: f"方案 {x} (评分: {table_data[x-1]['综合评分']})",
                    key="opt_selected_rank",
                )
                
                st.dataframe(
                    df,
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                )
                
                selected_idx = selected_rank - 1
                _, selected_ind = top_solutions[selected_idx]
                st.session_state.optimization_selected_solution = selected_ind
                
                st.markdown("---")
                st.markdown(f"#### 🎯 已选方案详情 (方案 {selected_rank})")
                
                col_det1, col_det2 = st.columns([1, 1])
                
                with col_det1:
                    st.markdown("**🔧 决策变量**")
                    do = selected_ind.variables[0] if len(selected_ind.variables) > 0 else 0
                    irr = selected_ind.variables[1] if len(selected_ind.variables) > 1 else 0
                    srt = selected_ind.variables[2] if len(selected_ind.variables) > 2 else 0
                    rr = selected_ind.variables[3] if len(selected_ind.variables) > 3 else 0
                    
                    st.metric("好氧池DO", f"{do:.2f} mg/L")
                    st.metric("内回流比", f"{irr:.0f} %")
                    st.metric("好氧池SRT", f"{srt:.1f} 天")
                    st.metric("回流污泥比", f"{rr:.0f} %")
                
                with col_det2:
                    st.markdown("**🎯 出水指标**")
                    cod = selected_ind.effluent_quality.get('COD', 0)
                    nh3 = selected_ind.effluent_quality.get('NH3_N', 0)
                    tn = selected_ind.effluent_quality.get('TN', 0)
                    tp = selected_ind.effluent_quality.get('TP', 0)
                    energy = selected_ind.energy_result.total_kwh_d if selected_ind.energy_result else 0
                    sludge = selected_ind.sludge_result.daily_sludge_kg if selected_ind.sludge_result else 0
                    
                    st.metric("出水COD", f"{cod:.2f} mg/L", 
                              delta=f"{cod - standard.COD:.2f}",
                              delta_color="inverse")
                    st.metric("出水NH3-N", f"{nh3:.2f} mg/L",
                              delta=f"{nh3 - standard.NH3_N:.2f}",
                              delta_color="inverse")
                    st.metric("出水TN", f"{tn:.2f} mg/L",
                              delta=f"{tn - standard.TN:.2f}",
                              delta_color="inverse")
                    st.metric("出水TP", f"{tp:.2f} mg/L",
                              delta=f"{tp - standard.TP:.2f}",
                              delta_color="inverse")
                
                col_e1, col_e2, col_e3 = st.columns(3)
                with col_e1:
                    st.metric("日均能耗", f"{energy:.1f} kWh/d")
                with col_e2:
                    st.metric("日产泥量", f"{sludge:.1f} kg DS/d")
                with col_e3:
                    if selected_ind.is_feasible:
                        st.success("✅ 全面达标")
                    else:
                        st.error("❌ 存在超标指标")
                
                st.markdown("---")
                
                if st.button("📋 一键回填到工艺配置", 
                            type="primary", 
                            use_container_width=True,
                            help="将此方案的参数应用到主系统的工艺配置中"):
                    
                    do = selected_ind.variables[0] if len(selected_ind.variables) > 0 else 2.0
                    irr = selected_ind.variables[1] if len(selected_ind.variables) > 1 else 200.0
                    srt = selected_ind.variables[2] if len(selected_ind.variables) > 2 else 15.0
                    rr = selected_ind.variables[3] if len(selected_ind.variables) > 3 else 50.0
                    
                    for reactor in st.session_state.pfs.reactors:
                        if reactor.reactor_type == ReactorType.AEROBIC:
                            reactor.operation.DO_setpoint = float(do)
                            reactor.operation.SRT = float(srt)
                        
                        if hasattr(reactor.operation, 'internal_return_ratio'):
                            reactor.operation.internal_return_ratio = float(irr) / 100.0
                        
                        if hasattr(reactor.operation, 'return_sludge_ratio'):
                            if reactor.reactor_type == ReactorType.SECONDARY or reactor.is_biological():
                                reactor.operation.return_sludge_ratio = float(rr) / 100.0
                    
                    for reactor in st.session_state.pfs.reactors:
                        if reactor.is_biological() and hasattr(reactor.operation, 'SRT'):
                            reactor.operation.SRT = float(srt)
                    
                    st.success(f"""
                    ✅ 参数已成功回填！
                    
                    **应用的参数:**
                    - 好氧池 DO: {do:.2f} mg/L
                    - 内回流比: {irr:.0f}%
                    - 好氧池 SRT: {srt:.1f} 天
                    - 回流污泥比: {rr:.0f}%
                    
                    请前往「稳态求解」页面重新运行仿真以验证优化效果。
                    """)
                    
                    st.session_state.current_page = "🎯 稳态求解"
                    st.rerun()
        
        with tab3:
            st.markdown("### 收敛曲线图")
            st.caption("展示每代种群的平均适应度和最优适应度变化趋势")
            
            if len(result.best_fitness_history) > 0:
                fig_conv = plot_convergence_curve(
                    result.best_fitness_history,
                    result.avg_fitness_history,
                )
                st.plotly_chart(fig_conv, use_container_width=True)
                
                st.info(f"""
                **优化统计:**
                - 总迭代代数: {len(result.all_populations)-1} 代
                - 总评估次数: {result.total_evaluations} 次
                - Pareto最优解数量: {len(result.pareto_front)} 个
                - 初始最优适应度: {result.best_fitness_history[0]:.2f}
                - 最终最优适应度: {result.best_fitness_history[-1]:.2f}
                - 改进幅度: {((result.best_fitness_history[0] - result.best_fitness_history[-1]) / result.best_fitness_history[0] * 100):.1f}%
                """)
            else:
                st.info("暂无收敛数据")
        
        with tab4:
            st.markdown("### 多目标平行坐标图")
            st.caption("展示各目标之间的权衡关系，每条线代表一个Pareto最优解")
            
            fig_parallel = plot_objective_parallel_coordinates(result.pareto_front)
            st.plotly_chart(fig_parallel, use_container_width=True)
            
            st.info("💡 **平行坐标图解读**: 从左到右依次为四个优化目标。"
                   "每条线代表一个方案，颜色深浅表示TN浓度。"
                   "通过观察线条走势可以发现目标之间的权衡关系。")
        
        st.markdown("---")
        st.subheader("💡 优化建议")
        
        if len(result.pareto_front) > 0:
            feasible_count = sum(1 for ind in result.pareto_front if ind.is_feasible)
            total_count = len(result.pareto_front)
            
            st.success(f"""
            **优化总结:**
            
            🎯 共找到 **{total_count}** 个Pareto最优解，其中 **{feasible_count}** 个满足一级A排放标准。
            
            📊 **目标范围:**
            - 能耗范围: {min(ind.energy_result.total_kwh_d for ind in result.pareto_front if ind.energy_result):.1f} - {max(ind.energy_result.total_kwh_d for ind in result.pareto_front if ind.energy_result):.1f} kWh/d
            - 出水TN范围: {min(ind.effluent_quality.get('TN', 0) for ind in result.pareto_front):.2f} - {max(ind.effluent_quality.get('TN', 0) for ind in result.pareto_front):.2f} mg/L
            
            💡 **决策建议**:
            - 如果最看重节能降耗，可选择能耗最低的方案
            - 如果最看重出水水质，可选择TN/氨氮最低的方案
            - 如果需要平衡多个目标，可选择综合评分最优的方案
            """)


def main():
    """主函数"""
    init_session_state()
    
    with st.sidebar:
        st.title("💧 污水仿真系统")
        st.markdown("---")
        
        pages = {
            "🏠 首页": page_home,
            "🔧 工艺流程": page_process_config,
            "🌊 进水配置": page_influent_config,
            "⚗️ 参数管理": page_parameters,
            "🎯 稳态求解": page_steady_state,
            "🔍 工艺对比": page_process_comparison,
            "📈 动态仿真": page_dynamic,
            "📊 敏感性分析": page_sensitivity,
            "💡 优化建议": page_optimization,
            "🧬 多目标优化": page_multiobjective_optimization,
        }
        
        for page_name, page_func in pages.items():
            if st.button(page_name, use_container_width=True):
                st.session_state.current_page = page_name
        
        st.markdown("---")
        st.caption("基于ASM1活性污泥模型")
        st.caption("版本 1.0.0")
    
    current_page = st.session_state.current_page
    
    for page_name, page_func in pages.items():
        if current_page == page_name:
            page_func()
            break


if __name__ == "__main__":
    main()
