import math
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ExperimentConfig:
    baseline_conversion_rate: float
    minimum_detectable_effect: float
    significance_level: float = 0.05
    statistical_power: float = 0.8
    relative_effect: bool = True
    test_type: str = "two-sided"


@dataclass
class SampleSizeResult:
    group_size: int
    total_size: int
    baseline_conversion_rate: float
    expected_conversion_rate: float
    absolute_effect: float
    relative_effect: float
    significance_level: float
    statistical_power: float
    test_type: str


@dataclass
class MultiVariantResult:
    groups: List[str]
    conversions: List[float]
    sample_sizes: List[int]
    required_per_group: int
    required_total: int
    baseline_group: str
    significance_level: float
    statistical_power: float


class SampleSizeCalculator:
    @staticmethod
    def calculate_sample_size(
        baseline_cr: float,
        mde: float,
        significance_level: float = 0.05,
        power: float = 0.8,
        relative_effect: bool = True,
        test_type: str = "two-sided"
    ) -> SampleSizeResult:
        if not (0 < baseline_cr < 1):
            raise ValueError("转化率必须在 (0, 1) 范围内")
        if not (0 < significance_level < 1):
            raise ValueError("显著性水平必须在 (0, 1) 范围内")
        if not (0 < power < 1):
            raise ValueError("统计功效必须在 (0, 1) 范围内")
        if test_type not in ["one-sided", "two-sided"]:
            raise ValueError("检验类型必须是 'one-sided' 或 'two-sided'")

        if relative_effect:
            absolute_effect = baseline_cr * mde
        else:
            absolute_effect = mde
        
        expected_cr = baseline_cr + absolute_effect
        
        if absolute_effect <= 0:
            raise ValueError("最小可检测效应必须为正值")
        if expected_cr >= 1:
            raise ValueError("期望转化率不能超过 1.0")

        if test_type == "two-sided":
            z_alpha = stats.norm.ppf(1 - significance_level / 2)
        else:
            z_alpha = stats.norm.ppf(1 - significance_level)
        
        z_beta = stats.norm.ppf(power)
        
        p_pooled = (baseline_cr + expected_cr) / 2
        pooled_variance = p_pooled * (1 - p_pooled) * 2
        
        effect_se = math.sqrt(
            baseline_cr * (1 - baseline_cr) + 
            expected_cr * (1 - expected_cr)
        )
        
        sample_size = (
            (z_alpha * math.sqrt(pooled_variance) + z_beta * math.sqrt(effect_se)) ** 2
        ) / (absolute_effect ** 2)
        
        rounded_size = math.ceil(sample_size)
        
        return SampleSizeResult(
            group_size=rounded_size,
            total_size=rounded_size * 2,
            baseline_conversion_rate=baseline_cr,
            expected_conversion_rate=expected_cr,
            absolute_effect=absolute_effect,
            relative_effect=absolute_effect / baseline_cr if relative_effect else mde,
            significance_level=significance_level,
            statistical_power=power,
            test_type=test_type
        )

    @staticmethod
    def calculate_multi_variant(
        baseline_cr: float,
        variant_crs: List[float],
        significance_level: float = 0.05,
        power: float = 0.8,
        correction_method: str = "bonferroni"
    ) -> MultiVariantResult:
        num_variants = len(variant_crs)
        
        if correction_method == "bonferroni":
            adjusted_alpha = significance_level / num_variants
        elif correction_method == "holm":
            adjusted_alpha = significance_level
        else:
            adjusted_alpha = significance_level

        required_sizes = []
        for variant_cr in variant_crs:
            mde_abs = abs(variant_cr - baseline_cr)
            mde_relative = mde_abs / baseline_cr
            
            result = SampleSizeCalculator.calculate_sample_size(
                baseline_cr=baseline_cr,
                mde=mde_relative,
                significance_level=adjusted_alpha,
                power=power,
                relative_effect=True,
                test_type="two-sided"
            )
            required_sizes.append(result.group_size)
        
        max_required = max(required_sizes)
        
        groups = ["对照组"] + [f"变体{i+1}" for i in range(num_variants)]
        conversions = [baseline_cr] + variant_crs
        sample_sizes = [max_required] * (num_variants + 1)
        
        return MultiVariantResult(
            groups=groups,
            conversions=conversions,
            sample_sizes=sample_sizes,
            required_per_group=max_required,
            required_total=max_required * (num_variants + 1),
            baseline_group="对照组",
            significance_level=significance_level,
            statistical_power=power
        )

    @staticmethod
    def generate_report(
        result: SampleSizeResult,
        experiment_name: Optional[str] = None
    ) -> str:
        lines = []
        lines.append("=" * 60)
        if experiment_name:
            lines.append(f"实验: {experiment_name}")
            lines.append("=" * 60)
        lines.append("A/B 实验样本量估算报告")
        lines.append("=" * 60)
        lines.append(f"\n【实验参数】")
        lines.append(f"  对照组转化率: {result.baseline_conversion_rate:.2%}")
        lines.append(f"  实验组期望转化率: {result.expected_conversion_rate:.2%}")
        lines.append(f"  绝对效应: {result.absolute_effect:.4f} ({result.absolute_effect*100:.2f}pp)")
        lines.append(f"  相对效应: {result.relative_effect:.2%}")
        lines.append(f"\n【统计参数】")
        lines.append(f"  显著性水平 (α): {result.significance_level:.2%}")
        lines.append(f"  统计功效 (1-β): {result.statistical_power:.2%}")
        lines.append(f"  检验类型: {result.test_type}")
        lines.append(f"\n【样本量要求】")
        lines.append(f"  每组所需样本量: {result.group_size:,}")
        lines.append(f"  总样本量 (对照组+实验组): {result.total_size:,}")
        lines.append("=" * 60)
        
        return "\n".join(lines)

    @staticmethod
    def generate_multi_variant_report(
        result: MultiVariantResult,
        experiment_name: Optional[str] = None
    ) -> str:
        lines = []
        lines.append("=" * 70)
        if experiment_name:
            lines.append(f"实验: {experiment_name}")
            lines.append("=" * 70)
        lines.append("多变量 (A/B/n) 实验样本量估算报告")
        lines.append("=" * 70)
        lines.append(f"\n【实验参数】")
        lines.append(f"  实验组数量: {len(result.groups)}")
        lines.append(f"  显著性水平 (α): {result.significance_level:.2%}")
        lines.append(f"  统计功效 (1-β): {result.statistical_power:.2%}")
        lines.append(f"\n【各组配置】")
        lines.append(f"  {'组别':<15} {'转化率':<12} {'所需样本量':<15}")
        lines.append(f"  {'-'*14} {'-'*11} {'-'*14}")
        for i, (group, conv, size) in enumerate(zip(result.groups, result.conversions, result.sample_sizes)):
            marker = " (基准)" if group == result.baseline_group else ""
            lines.append(f"  {group:<15} {conv:.2%}{'':<6} {size:>,}")
        lines.append(f"\n【样本量要求】")
        lines.append(f"  每组所需样本量: {result.required_per_group:,}")
        lines.append(f"  总样本量 ({len(result.groups)}组): {result.required_total:,}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
