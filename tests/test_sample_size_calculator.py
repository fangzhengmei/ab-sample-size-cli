import pytest
from sample_size_cli.sample_size_calculator import (
    SampleSizeCalculator,
    SampleSizeResult,
    MultiVariantResult,
)


class TestSampleSizeCalculator:
    
    def test_basic_ab_test_sample_size(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        assert isinstance(result, SampleSizeResult)
        assert result.baseline_conversion_rate == pytest.approx(0.05)
        assert result.expected_conversion_rate == pytest.approx(0.055)
        assert result.absolute_effect == pytest.approx(0.005)
        assert result.relative_effect == pytest.approx(0.1)
        assert result.group_size > 0
        assert result.total_size == result.group_size * 2
    
    def test_absolute_effect_ab_test(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.005,
            significance_level=0.05,
            power=0.8,
            relative_effect=False,
            test_type="two-sided",
        )
        
        assert result.absolute_effect == 0.005
        assert result.expected_conversion_rate == 0.055
    
    def test_one_sided_test(self):
        two_sided_result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        one_sided_result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="one-sided",
        )
        
        assert one_sided_result.group_size < two_sided_result.group_size
    
    def test_different_significance_level(self):
        strict_result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.01,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        relaxed_result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.10,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        assert strict_result.group_size > relaxed_result.group_size
    
    def test_different_power(self):
        high_power_result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.9,
            relative_effect=True,
            test_type="two-sided",
        )
        
        low_power_result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.7,
            relative_effect=True,
            test_type="two-sided",
        )
        
        assert high_power_result.group_size > low_power_result.group_size
    
    def test_larger_effect_size_requires_smaller_sample(self):
        small_effect = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.05,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        large_effect = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.20,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        assert small_effect.group_size > large_effect.group_size
    
    def test_invalid_baseline_cr_zero(self):
        with pytest.raises(ValueError, match="转化率必须在.*范围内"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=0.0,
                mde=0.1,
                significance_level=0.05,
                power=0.8,
            )
    
    def test_invalid_baseline_cr_one(self):
        with pytest.raises(ValueError, match="转化率必须在.*范围内"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=1.0,
                mde=0.1,
                significance_level=0.05,
                power=0.8,
            )
    
    def test_invalid_significance_level(self):
        with pytest.raises(ValueError, match="显著性水平必须在.*范围内"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=0.05,
                mde=0.1,
                significance_level=0.0,
                power=0.8,
            )
    
    def test_invalid_power(self):
        with pytest.raises(ValueError, match="统计功效必须在.*范围内"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=0.05,
                mde=0.1,
                significance_level=0.05,
                power=1.0,
            )
    
    def test_invalid_test_type(self):
        with pytest.raises(ValueError, match="检验类型必须是"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=0.05,
                mde=0.1,
                significance_level=0.05,
                power=0.8,
                test_type="invalid",
            )
    
    def test_negative_effect_size(self):
        with pytest.raises(ValueError, match="最小可检测效应必须为正值"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=0.05,
                mde=-0.1,
                significance_level=0.05,
                power=0.8,
            )
    
    def test_effect_size_too_large(self):
        with pytest.raises(ValueError, match="期望转化率不能超过 1.0"):
            SampleSizeCalculator.calculate_sample_size(
                baseline_cr=0.95,
                mde=0.1,
                significance_level=0.05,
                power=0.8,
                relative_effect=True,
            )


class TestMultiVariantCalculator:
    
    def test_basic_multi_variant(self):
        result = SampleSizeCalculator.calculate_multi_variant(
            baseline_cr=0.05,
            variant_crs=[0.055, 0.06],
            significance_level=0.05,
            power=0.8,
            correction_method="bonferroni",
        )
        
        assert isinstance(result, MultiVariantResult)
        assert len(result.groups) == 3
        assert result.groups[0] == "对照组"
        assert result.groups[1] == "变体1"
        assert result.groups[2] == "变体2"
        assert result.conversions[0] == 0.05
        assert result.required_per_group > 0
        assert result.required_total == result.required_per_group * 3
    
    def test_multi_variant_without_correction(self):
        with_correction = SampleSizeCalculator.calculate_multi_variant(
            baseline_cr=0.05,
            variant_crs=[0.055, 0.06],
            significance_level=0.05,
            power=0.8,
            correction_method="bonferroni",
        )
        
        without_correction = SampleSizeCalculator.calculate_multi_variant(
            baseline_cr=0.05,
            variant_crs=[0.055, 0.06],
            significance_level=0.05,
            power=0.8,
            correction_method="none",
        )
        
        assert with_correction.required_per_group >= without_correction.required_per_group
    
    def test_single_variant(self):
        result = SampleSizeCalculator.calculate_multi_variant(
            baseline_cr=0.05,
            variant_crs=[0.055],
            significance_level=0.05,
            power=0.8,
            correction_method="bonferroni",
        )
        
        assert len(result.groups) == 2
        assert result.required_total == result.required_per_group * 2


class TestReportGeneration:
    
    def test_ab_test_report_generation(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        report = SampleSizeCalculator.generate_report(result, experiment_name="按钮颜色测试")
        
        assert "按钮颜色测试" in report
        assert "5.00%" in report
        assert "5.50%" in report
        assert "10.00%" in report
        assert "5.00%" in report
        assert "80.00%" in report
        assert "样本量" in report
    
    def test_multi_variant_report_generation(self):
        result = SampleSizeCalculator.calculate_multi_variant(
            baseline_cr=0.05,
            variant_crs=[0.055, 0.06],
            significance_level=0.05,
            power=0.8,
            correction_method="bonferroni",
        )
        
        report = SampleSizeCalculator.generate_multi_variant_report(result, experiment_name="登录页面测试")
        
        assert "登录页面测试" in report
        assert "多变量" in report
        assert "对照组" in report
        assert "变体1" in report
        assert "变体2" in report
        assert "5.00%" in report
        assert "样本量" in report
    
    def test_report_without_name(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
        )
        
        report = SampleSizeCalculator.generate_report(result)
        
        assert "A/B 实验样本量估算报告" in report
        assert "=" * 60 in report


class TestSampleSizeAccuracy:
    
    def test_sample_size_order_of_magnitude(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
        )
        
        assert 10000 < result.group_size < 50000
    
    def test_very_small_effect_size(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.01,
            significance_level=0.05,
            power=0.8,
        )
        
        assert result.group_size > 100000
    
    def test_high_conversion_rate(self):
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.5,
            mde=0.05,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
        )
        
        assert isinstance(result, SampleSizeResult)
        assert result.group_size > 0
        assert result.expected_conversion_rate == 0.525


class TestSampleSizeFormulaRegression:
    """
    样本量计算公式回归测试
    防止再次出现重复开方导致样本量偏大的问题
    """
    
    def test_formula_no_double_sqrt_regression_basic_case(self):
        """
        回归测试：验证标准场景下的样本量计算
        基准值：5% 转化率，10% 相对提升，α=0.05，功效=0.8
        修复前（错误）：每组约 47,600
        修复后（正确）：每组约 31,000
        此测试防止公式回归到重复开方的错误实现
        """
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05,
            mde=0.1,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        assert 30000 < result.group_size < 33000
        assert 60000 < result.total_size < 66000
        
        assert not (45000 < result.group_size < 50000), \
            "样本量异常偏大，可能存在重复开方的公式回归问题"
    
    def test_formula_accurate_vs_polled_se(self):
        """
        回归测试：验证公式使用正确的 pooled SE 计算
        正确公式：pooled_se = sqrt(2 * p̂ * (1-p̂))
        错误公式（重复开方）：对已经开方的值再次开方
        """
        from scipy import stats
        import math
        
        baseline_cr = 0.05
        mde = 0.1
        absolute_effect = baseline_cr * mde
        expected_cr = baseline_cr + absolute_effect
        
        z_alpha = stats.norm.ppf(1 - 0.05 / 2)
        z_beta = stats.norm.ppf(0.8)
        
        p_pooled = (baseline_cr + expected_cr) / 2
        pooled_se = math.sqrt(2 * p_pooled * (1 - p_pooled))
        effect_se = math.sqrt(
            baseline_cr * (1 - baseline_cr) + 
            expected_cr * (1 - expected_cr)
        )
        
        expected_numerator = (z_alpha * pooled_se + z_beta * effect_se) ** 2
        expected_sample_size = expected_numerator / (absolute_effect ** 2)
        
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=baseline_cr,
            mde=mde,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
            test_type="two-sided",
        )
        
        assert result.group_size == math.ceil(expected_sample_size), \
            "公式计算结果与手动推导不一致，可能存在回归问题"
    
    def test_formula_parameter_sensitivity_regression(self):
        """
        回归测试：验证公式对参数变化的敏感度方向正确
        防止逻辑错误导致参数影响方向反转
        """
        base = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05, mde=0.1, significance_level=0.05, power=0.8
        )
        
        stricter_alpha = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05, mde=0.1, significance_level=0.01, power=0.8
        )
        assert stricter_alpha.group_size > base.group_size, \
            "更严格的显著性水平应该需要更大样本量"
        
        higher_power = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05, mde=0.1, significance_level=0.05, power=0.95
        )
        assert higher_power.group_size > base.group_size, \
            "更高的功效应该需要更大样本量"
        
        larger_effect = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.05, mde=0.2, significance_level=0.05, power=0.8
        )
        assert larger_effect.group_size < base.group_size, \
            "更大的效应量应该需要更小样本量"
        
        higher_cr = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.2, mde=0.1, significance_level=0.05, power=0.8
        )
        higher_cr_base = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=0.1, mde=0.1, significance_level=0.05, power=0.8
        )
        assert higher_cr.group_size < higher_cr_base.group_size, \
            "更高的基线转化率应该需要更小样本量（相同相对效应下）"
    
    def test_formula_absolute_vs_relative_consistency(self):
        """
        回归测试：验证绝对效应和相对效应计算的一致性
        当 MDE 数值相同时，两种模式应该产生相同的绝对效应
        """
        baseline_cr = 0.05
        absolute_mde = 0.005
        relative_mde = 0.1
        
        result_absolute = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=baseline_cr,
            mde=absolute_mde,
            significance_level=0.05,
            power=0.8,
            relative_effect=False,
        )
        
        result_relative = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=baseline_cr,
            mde=relative_mde,
            significance_level=0.05,
            power=0.8,
            relative_effect=True,
        )
        
        assert result_absolute.group_size == result_relative.group_size, \
            "相同的绝对效应值应该产生相同的样本量计算结果"
        assert result_absolute.absolute_effect == pytest.approx(result_relative.absolute_effect)


class TestCLIArgumentRegression:
    """
    CLI 短参数解析回归测试
    防止再次出现 -a 参数冲突的问题（显著性水平 vs 效应类型）
    """
    
    def test_cli_short_arg_a_is_significance_level(self):
        """
        回归测试：验证 -a 短参数用于显著性水平
        历史问题：-a 曾同时被用于 --significance-level 和 --relative
        """
        from click.testing import CliRunner
        from sample_size_cli.main import main
        
        runner = CliRunner()
        
        result = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.1", "-a", "0.01"]
        )
        
        assert result.exit_code == 0
        assert "1.00%" in result.output or "0.01" in result.output
        
        result_default = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.1"]
        )
        
        assert result.exit_code == 0
        assert "5.00%" in result_default.output
    
    def test_cli_short_arg_e_is_absolute_effect(self):
        """
        回归测试：验证 -e 短参数用于绝对效应
        修复后：-e = --absolute，-R = --relative
        """
        from click.testing import CliRunner
        from sample_size_cli.main import main
        
        runner = CliRunner()
        
        result_with_e = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.005", "-e"]
        )
        
        assert result_with_e.exit_code == 0
        assert "10.00%" not in result_with_e.output
        
        result_without_e = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.1"]
        )
        
        assert result_without_e.exit_code == 0
        assert "10.00%" in result_without_e.output
    
    def test_cli_short_arg_R_is_relative_effect(self):
        """
        回归测试：验证 -R 短参数用于相对效应（默认值）
        """
        from click.testing import CliRunner
        from sample_size_cli.main import main
        
        runner = CliRunner()
        
        result_with_R = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.1", "-R"]
        )
        
        assert result_with_R.exit_code == 0
        assert "10.00%" in result_with_R.output
        
        result_default = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.1"]
        )
        
        assert result_default.exit_code == 0
        assert result_with_R.output == result_default.output
    
    def test_cli_no_argument_conflict_regression(self):
        """
        回归测试：验证 -a 和 -e/-R 可以同时使用且不冲突
        历史问题：-a 曾冲突，导致参数解析混乱
        """
        from click.testing import CliRunner
        from sample_size_cli.main import main
        
        runner = CliRunner()
        
        result = runner.invoke(
            main,
            ["ab-test", "-b", "0.05", "-m", "0.005", "-a", "0.01", "-e"]
        )
        
        assert result.exit_code == 0
        
        assert "1.00%" in result.output or "显著性水平.*1.00" in result.output.lower() or "0.01" in result.output
    
    def test_cli_help_shows_correct_short_args(self):
        """
        回归测试：验证帮助信息显示正确的短参数映射
        """
        from click.testing import CliRunner
        from sample_size_cli.main import main
        
        runner = CliRunner()
        
        result = runner.invoke(main, ["ab-test", "--help"])
        
        assert result.exit_code == 0
        
        assert "-a, --significance-level" in result.output
        assert "-e, --absolute" in result.output
        assert "-R, --relative" in result.output
        
        assert "-r/-a" not in result.output, \
            "帮助信息不应再显示旧的 -r/-a 参数冲突映射"
