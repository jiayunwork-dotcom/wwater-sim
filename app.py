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
    solve_steady_state,
    run_dynamic_simulation,
    DynamicSimulator,
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
    
    unit_lib = {
        'grit': ('沉砂池', ReactorType.GRIT),
        'primary': ('初沉池', ReactorType.PRIMARY),
        'anaerobic': ('厌氧池', ReactorType.ANAEROBIC),
        'anoxic': ('缺氧池', ReactorType.ANOXIC),
        'aerobic': ('好氧池', ReactorType.AEROBIC),
        'secondary': ('二沉池', ReactorType.SECONDARY),
        'disinfection': ('消毒池', ReactorType.DISINFECTION),
        'membrane': ('膜组件', ReactorType.MEMBRANE),
    }
    
    lib_cols = st.columns(4)
    for i, (key, (name, rtype)) in enumerate(unit_lib.items()):
        with lib_cols[i % 4]:
            icon = REACTOR_TYPE_ICONS.get(rtype, '⬜')
            if st.button(f"{icon} 添加{name}", key=f"add_{key}", use_container_width=True):
                new_reactor = create_reactor_by_type(rtype, f"{name}{len(st.session_state.pfs.reactors)+1}")
                st.session_state.pfs.add_reactor(new_reactor)
                if len(st.session_state.pfs.reactors) > 1:
                    st.session_state.pfs.connect(len(st.session_state.pfs.reactors)-2, 
                                                  len(st.session_state.pfs.reactors)-1)
                st.success(f"已添加{name}")
    
    if len(st.session_state.pfs.reactors) > 0:
        if st.button("🗑️ 清空流程", use_container_width=True):
            st.session_state.pfs = ProcessFlowSheet()
            st.warning("已清空流程")
    
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
                
                col_act1, col_act2 = st.columns(2)
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
                
                if st.button("❌ 删除", key=f"del_{idx}"):
                    pfs.reactors.pop(idx)
                    pfs.connections = []
                    for i in range(len(pfs.reactors) - 1):
                        pfs.connect(i, i + 1)
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
        st.write("设置24小时内流量和浓度的变化系数")
        
        pattern_type = st.selectbox(
            "选择日变化模式",
            ['早晚高峰模式', '均匀模式'],
            index=0,
        )
        
        if pattern_type == '早晚高峰模式':
            influent.set_diurnal_pattern('morning_evening_peak')
        else:
            influent.set_diurnal_pattern('uniform')
        
        hours = list(range(24))
        
        col_flow, col_conc = st.columns(2)
        
        with col_flow:
            st.markdown("**流量日变化曲线**")
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
            st.markdown("**浓度日变化曲线**")
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
            min_value=1e-10, max_value=1e-4,
            value=float(config.tolerance),
            step=1e-8,
            format="%.1e",
        )
        relaxation = st.number_input(
            "松弛因子",
            min_value=0.1, max_value=1.0,
            value=float(config.relaxation),
            step=0.1,
        )
        
        config.max_iterations = max_iter
        config.tolerance = tolerance
        config.relaxation = relaxation
    
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
    
    col_run, col_stop = st.columns(2)
    
    with col_run:
        if st.button("▶️ 开始动态仿真", type="primary", use_container_width=True):
            initial_states = None
            if st.session_state.steady_result is not None and st.session_state.steady_result.converged:
                initial_states = st.session_state.steady_result.reactor_states
                st.info("使用稳态解作为初始条件")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("正在仿真...")
            with st.spinner("动态仿真进行中..."):
                result = run_dynamic_simulation(
                    st.session_state.pfs,
                    st.session_state.influent,
                    st.session_state.asm1_params,
                    initial_states=initial_states,
                    config=config,
                )
                st.session_state.dynamic_result = result
                progress_bar.progress(1.0)
            
            if result.success:
                st.success("仿真完成！")
            else:
                st.error(f"仿真失败: {result.message}")
    
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
            "📈 动态仿真": page_dynamic,
            "📊 敏感性分析": page_sensitivity,
            "💡 优化建议": page_optimization,
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
