# 具体板块收益：指定格式的直方图报告

板块间对比补充：`alpha_lab/outputs/sector_fuzzy_between_sectors/20260923T115005_661608Z/`。
`between_sectors_report.pdf` 为新增 4 页，含每板块等权的分布直方图及全部板块名称并列柱状图；
`confirmation_returns_report.pdf` 为前置这 4 页后的 72 页完整版。
蓝柱为板块期末均值，绿/红柱为该板块全部完整事件期间最高/最低值（跨事件极值，非均值）。
全部 195 个板块-分类版本-期限组合已核对；缺失未置零，同一期限跨页横轴一致。
复现时经冻结启动器运行 `alpha_lab/experiments/sector_fuzzy_v1/report_between.py`。

正式产物：`alpha_lab/outputs/sector_fuzzy_histogram_report/20260923T114347_981934Z/`。

- `confirmation_returns_report.pdf`：68 页；前 3 页为分面直方图总览，后 65 页为具体板块-分类版本详情。
- `overview.html`：按板块名称点击跳转，每个板块的图可单独查看。
- `page_index.csv`：页码、名称、编号、分类版本。
- `named_sector_summary.csv`：逐板块的 3/10/20 日完整事件数、均值、中位数及期末/期间极值。

参照用户指定的多窗口事件收益 PDF，保留 3/10/20 日并列、分箱直方图和收益颜色。
按最新要求取消累计曲线，逐板块展示板块事件收益、内部个股期末收益与期间最低/最高收益直方图。
61 个板块代码因分类版本拆分为 65 页，覆盖全部 82 个选中事件；完整窗口计数保持 77/72/68。
样本不足或没有完整窗口的板块仍保留并标注。此次只更改展示，没有训练或调整筛选。

复现：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python datahub/scripts/frozen_research.py run \
  --release datahub/releases/selection_v1_20260920 \
  alpha_lab/experiments/sector_fuzzy_v1/report_histograms.py
```
