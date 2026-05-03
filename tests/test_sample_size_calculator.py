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
