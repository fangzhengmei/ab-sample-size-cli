import click
from typing import List, Optional
from .sample_size_calculator import SampleSizeCalculator


@click.group()
@click.version_option(version="0.1.0")
def main():
    """A/B 实验样本量估算 CLI 工具"""
    pass


@main.command("ab-test")
@click.option("--baseline-cr", "-b", required=True, type=float,
              help="对照组转化率 (如 0.05 表示 5%)")
@click.option("--mde", "-m", required=True, type=float,
              help="最小可检测效应 (如 0.1 表示 10% 相对提升)")
@click.option("--significance-level", "-a", default=0.05, type=float,
              help="显著性水平 α (默认: 0.05)")
@click.option("--power", "-p", default=0.8, type=float,
              help="统计功效 1-β (默认: 0.8)")
@click.option("--absolute/--relative", "-e/-R", default=False,
              help="MDE 是绝对效应 (-e) 还是相对效应 (-R，默认)")
@click.option("--test-type", "-t", type=click.Choice(["one-sided", "two-sided"]),
              default="two-sided", help="检验类型 (默认: two-sided)")
@click.option("--name", "-n", help="实验名称 (可选)")
@click.option("--json", "-j", "output_json", is_flag=True,
              help="以 JSON 格式输出结果")
def ab_test(
    baseline_cr: float,
    mde: float,
    significance_level: float,
    power: float,
    absolute: bool,
    test_type: str,
    name: Optional[str],
    output_json: bool,
):
    """计算 A/B 实验所需样本量"""
    try:
        result = SampleSizeCalculator.calculate_sample_size(
            baseline_cr=baseline_cr,
            mde=mde,
            significance_level=significance_level,
            power=power,
            relative_effect=not absolute,
            test_type=test_type,
        )
        
        if output_json:
            import json
            output = {
                "experiment_name": name,
                "baseline_conversion_rate": result.baseline_conversion_rate,
                "expected_conversion_rate": result.expected_conversion_rate,
                "absolute_effect": result.absolute_effect,
                "relative_effect": result.relative_effect,
                "significance_level": result.significance_level,
                "statistical_power": result.statistical_power,
                "test_type": result.test_type,
                "sample_size_per_group": result.group_size,
                "total_sample_size": result.total_size,
            }
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            report = SampleSizeCalculator.generate_report(result, experiment_name=name)
            click.echo(report)
            
    except ValueError as e:
        click.echo(f"错误: {e}", err=True)
        raise click.Abort()


@main.command("multi-test")
@click.option("--baseline-cr", "-b", required=True, type=float,
              help="对照组转化率 (如 0.05 表示 5%)")
@click.option("--variant-cr", "-v", required=True, multiple=True, type=float,
              help="各变体组的转化率 (可多次使用，如 -v 0.055 -v 0.06)")
@click.option("--significance-level", "-a", default=0.05, type=float,
              help="显著性水平 α (默认: 0.05)")
@click.option("--power", "-p", default=0.8, type=float,
              help="统计功效 1-β (默认: 0.8)")
@click.option("--correction", "-c", 
              type=click.Choice(["bonferroni", "holm", "none"]),
              default="bonferroni",
              help="多重检验校正方法 (默认: bonferroni)")
@click.option("--name", "-n", help="实验名称 (可选)")
@click.option("--json", "-j", "output_json", is_flag=True,
              help="以 JSON 格式输出结果")
def multi_test(
    baseline_cr: float,
    variant_cr: List[float],
    significance_level: float,
    power: float,
    correction: str,
    name: Optional[str],
    output_json: bool,
):
    """计算多变量 (A/B/n) 实验所需样本量"""
    try:
        result = SampleSizeCalculator.calculate_multi_variant(
            baseline_cr=baseline_cr,
            variant_crs=list(variant_cr),
            significance_level=significance_level,
            power=power,
            correction_method=correction,
        )
        
        if output_json:
            import json
            output = {
                "experiment_name": name,
                "baseline_group": result.baseline_group,
                "baseline_conversion_rate": baseline_cr,
                "variants": [
                    {"group": result.groups[i], "conversion_rate": result.conversions[i]}
                    for i in range(len(result.groups))
                ],
                "significance_level": result.significance_level,
                "statistical_power": result.statistical_power,
                "correction_method": correction,
                "sample_size_per_group": result.required_per_group,
                "total_sample_size": result.required_total,
            }
            click.echo(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            report = SampleSizeCalculator.generate_multi_variant_report(result, experiment_name=name)
            click.echo(report)
            
    except ValueError as e:
        click.echo(f"错误: {e}", err=True)
        raise click.Abort()


@main.command("interactive")
def interactive():
    """交互式计算样本量"""
    click.echo("=" * 60)
    click.echo("欢迎使用 A/B 实验样本量估算工具 (交互式)")
    click.echo("=" * 60)
    
    exp_type = click.prompt(
        "请选择实验类型 [1=A/B 测试, 2=多变量测试]",
        type=click.Choice(["1", "2"])
    )
    
    baseline_cr = click.prompt(
        "请输入对照组转化率 (如 0.05 表示 5%)",
        type=float
    )
    
    significance_level = click.prompt(
        "请输入显著性水平 α (默认 0.05)",
        default=0.05,
        type=float
    )
    
    power = click.prompt(
        "请输入统计功效 1-β (默认 0.8)",
        default=0.8,
        type=float
    )
    
    if exp_type == "1":
        mde_type = click.prompt(
            "最小可检测效应类型 [1=相对效应, 2=绝对效应]",
            type=click.Choice(["1", "2"])
        )
        mde = click.prompt(
            "请输入最小可检测效应 (如 0.1 表示 10%)",
            type=float
        )
        
        test_type = click.prompt(
            "检验类型 [1=双侧检验, 2=单侧检验]",
            type=click.Choice(["1", "2"])
        )
        
        name = click.prompt("实验名称 (可选)", default="")
        
        try:
            result = SampleSizeCalculator.calculate_sample_size(
                baseline_cr=baseline_cr,
                mde=mde,
                significance_level=significance_level,
                power=power,
                relative_effect=(mde_type == "1"),
                test_type="two-sided" if test_type == "1" else "one-sided",
            )
            report = SampleSizeCalculator.generate_report(
                result, 
                experiment_name=name if name else None
            )
            click.echo("\n" + report)
        except ValueError as e:
            click.echo(f"错误: {e}", err=True)
            
    else:
        num_variants = click.prompt(
            "请输入变体组数量",
            type=int,
            default=2
        )
        
        variant_crs = []
        for i in range(num_variants):
            cr = click.prompt(
                f"请输入变体组 {i+1} 的转化率",
                type=float
            )
            variant_crs.append(cr)
        
        correction = click.prompt(
            "多重检验校正方法 [1=Bonferroni, 2=Holm, 3=不校正]",
            type=click.Choice(["1", "2", "3"])
        )
        correction_map = {"1": "bonferroni", "2": "holm", "3": "none"}
        
        name = click.prompt("实验名称 (可选)", default="")
        
        try:
            result = SampleSizeCalculator.calculate_multi_variant(
                baseline_cr=baseline_cr,
                variant_crs=variant_crs,
                significance_level=significance_level,
                power=power,
                correction_method=correction_map[correction],
            )
            report = SampleSizeCalculator.generate_multi_variant_report(
                result, 
                experiment_name=name if name else None
            )
            click.echo("\n" + report)
        except ValueError as e:
            click.echo(f"错误: {e}", err=True)


if __name__ == "__main__":
    main()
