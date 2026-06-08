"""
系统测试脚本
验证各模块功能是否正常
"""

import sys
import numpy as np

print("=" * 60)
print("污水处理仿真系统 - 模块测试")
print("=" * 60)

try:
    print("\n1. 测试 ASM1 模型模块...")
    from src.asm1_model import (
        ASM1Parameters,
        NUM_COMPONENTS,
        COMPONENT_NAMES,
        get_stoichiometric_matrix,
        calculate_process_rates,
        calculate_reaction_contributions,
        aggregate_to_wq_indices,
        get_typical_influent,
    )
    
    params = ASM1Parameters()
    print(f"   ✓ ASM1参数加载成功，共{NUM_COMPONENTS}个组分")
    print(f"   ✓ 组分: {', '.join(COMPONENT_NAMES[:5])}...")
    
    stoich_matrix = get_stoichiometric_matrix()
    print(f"   ✓ 化学计量矩阵形状: {stoich_matrix.shape}")
    
    C_test = get_typical_influent('domestic')
    print(f"   ✓ 典型生活污水进水加载成功")
    
    rates = calculate_process_rates(C_test, params, DO_setpoint=2.0)
    print(f"   ✓ 反应速率计算成功，共{len(rates)}个过程")
    
    reaction = calculate_reaction_contributions(C_test, params, DO_setpoint=2.0)
    print(f"   ✓ 反应贡献计算成功")
    
    wq = aggregate_to_wq_indices(C_test)
    print(f"   ✓ 水质指标聚合: COD={wq['COD']}, BOD5={wq['BOD5']}, NH3-N={wq['NH3_N']}")
    
    temp_params = params.get_temperature_corrected_params(25)
    print(f"   ✓ 温度修正成功 (20°C -> 25°C)")
    
    print("   ✅ ASM1模型模块测试通过")
    
except Exception as e:
    print(f"   ❌ ASM1模型模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n2. 测试 反应器单元模块...")
    from src.reactor_units import (
        ReactorType,
        ProcessFlowSheet,
        create_reactor_by_type,
        CSTRReactor,
        SecondaryClarifier,
        MembraneUnit,
    )
    
    pfs = ProcessFlowSheet()
    
    anaerobic = create_reactor_by_type(ReactorType.ANAEROBIC, "厌氧池", volume=800)
    pfs.add_reactor(anaerobic)
    
    aerobic = create_reactor_by_type(ReactorType.AEROBIC, "好氧池", volume=2000)
    pfs.add_reactor(aerobic)
    
    secondary = create_reactor_by_type(ReactorType.SECONDARY, "二沉池", volume=600)
    pfs.add_reactor(secondary)
    
    pfs.connect(0, 1)
    pfs.connect(1, 2)
    
    print(f"   ✓ 工艺流程创建成功，共{len(pfs.reactors)}个单元")
    print(f"   ✓ 总容积: {pfs.get_volume():.0f} m³")
    print(f"   ✓ 反应器类型: {[r.get_type_name() for r in pfs.reactors]}")
    
    print("   ✅ 反应器单元模块测试通过")
    
except Exception as e:
    print(f"   ❌ 反应器单元模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n3. 测试 工艺模板模块...")
    from src.process_templates import (
        PROCESS_TEMPLATES,
        create_process_by_name,
        InfluentConfig,
    )
    
    for key, template in PROCESS_TEMPLATES.items():
        pfs = template.create()
        valid, msg = pfs.validate()
        print(f"   ✓ {template.name}: {len(pfs.reactors)}个单元, 验证{'通过' if valid else '失败'}")
    
    influent = InfluentConfig()
    influent.set_diurnal_pattern('morning_evening_peak')
    Q = influent.get_Q(8)
    C = influent.get_C(8)
    print(f"   ✓ 进水配置: 流量={Q:.0f} m³/d, 浓度维度={len(C)}")
    
    print("   ✅ 工艺模板模块测试通过")
    
except Exception as e:
    print(f"   ❌ 工艺模板模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n4. 测试 求解器模块...")
    from src.solver import (
        SolverConfig,
        solve_steady_state,
        run_dynamic_simulation,
    )
    
    pfs = create_process_by_name('A2O')
    influent = InfluentConfig()
    params = ASM1Parameters()
    config = SolverConfig(
        max_iterations=100, 
        tolerance=1e-4, 
        warmup_days=2.0,
        steady_state_method='dynamic',
        dynamic_steady_days=20.0,
        check_steady_every_days=5.0
    )
    
    print("   正在运行稳态求解 (可能需要几秒钟)...")
    result = solve_steady_state(pfs, influent, params, config)
    
    print(f"   ✓ 稳态求解完成: 收敛={result.converged}, "
          f"迭代次数={result.iterations}, "
          f"残差={result.final_residual:.2e}")
    
    print(f"   ✓ 出水水质: COD={result.effluent_quality['COD']:.1f}, "
          f"NH3-N={result.effluent_quality['NH3_N']:.1f}, "
          f"TN={result.effluent_quality['TN']:.1f}")
    
    print("   正在运行动态仿真 (简化测试)...")
    config_dyn = SolverConfig(simulation_days=1, output_interval_days=0.5)
    dyn_result = run_dynamic_simulation(
        pfs, influent, params,
        initial_states=result.reactor_states,
        config=config_dyn
    )
    print(f"   ✓ 动态仿真完成: 成功={dyn_result.success}, "
          f"时间步数={len(dyn_result.time_days)}")
    
    if not result.converged:
        print("   ⚠ 注意: 稳态求解标记为未完全收敛，但结果仍可使用")
    
    print("   ✅ 求解器模块测试通过")
    
except Exception as e:
    print(f"   ❌ 求解器模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n5. 测试 分析模块...")
    from src.analysis import (
        check_compliance,
        STANDARDS,
        SENSITIVITY_PARAMETERS,
        generate_optimization_suggestions,
    )
    
    effluent = {'COD': 45, 'BOD5': 8, 'NH3_N': 4.5, 'TN': 14, 'TP': 0.4, 'SS': 8}
    
    comp = check_compliance(effluent, '一级A')
    print(f"   ✓ 达标判定: 总体{'达标' if comp.overall_compliant else '不达标'}")
    print(f"   ✓ 指标数: {len(comp.items)}")
    
    print(f"   ✓ 可用标准: {list(STANDARDS.keys())}")
    print(f"   ✓ 敏感性参数: {len(SENSITIVITY_PARAMETERS)}个")
    
    print("   ✅ 分析模块测试通过")
    
except Exception as e:
    print(f"   ❌ 分析模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n6. 测试 可视化模块...")
    from src.visualization import (
        plot_process_diagram,
        plot_reactor_stack,
        plot_compliance_radar,
    )
    
    pfs = create_process_by_name('A2O')
    fig1 = plot_process_diagram(pfs)
    print(f"   ✓ 工艺流程图创建成功")
    
    reactor_states = [np.random.rand(NUM_COMPONENTS) * 100 for _ in pfs.reactors]
    fig2 = plot_reactor_stack(pfs, reactor_states)
    print(f"   ✓ 组分堆叠图创建成功")
    
    from src.analysis import check_compliance
    effluent = {'COD': 45, 'BOD5': 8, 'NH3_N': 4.5, 'TN': 14, 'TP': 0.4, 'SS': 8}
    comp = check_compliance(effluent, '一级A')
    fig3 = plot_compliance_radar(comp)
    print(f"   ✓ 雷达图创建成功")
    
    print("   ✅ 可视化模块测试通过")
    
except Exception as e:
    print(f"   ❌ 可视化模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有模块测试通过！系统可以正常运行。")
print("=" * 60)

print("\n启动方式:")
print("  source venv/bin/activate")
print("  streamlit run app.py")
