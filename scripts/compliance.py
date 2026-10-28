"""Export truthful provenance and the required XLSX checklist using stdlib only."""
import argparse
import json
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from src.model import verified_weights
from src.utils import write_json, sha256


def xlsx(path, rows):
    cells = []
    for i, row in enumerate(rows, 1):
        cells.append('<row r="%d">%s</row>' % (i, ''.join(
            f'<c r="{chr(65+j)}{i}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>' for j, v in enumerate(row))))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr('_rels/.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr('xl/workbook.xml', '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="合规清单" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>' + ''.join(cells) + '</sheetData></worksheet>')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='configs/weights_manifest.template.json')
    p.add_argument('--output', default='outputs/compliance')
    p.add_argument('--ingest', help='Actual data ingest report to attach (does not certify provenance)')
    p.add_argument('--experiment', help='Attach the current registered formal experiment status')
    a = p.parse_args()
    m = json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    status = 'pending_review'
    ingest = json.loads(Path(a.ingest).read_text(encoding='utf-8')) if a.ingest else None
    experiment = None
    if a.experiment:
        from src.registry import Registry
        experiment = Registry().get(a.experiment)
    if m.get('status') == 'approved':
        verified_weights(a.manifest)
        status = 'provenance_verified_not_competition_certified'
    rows = [['项目', '内容', '状态']]
    for k, v in m.items():
        rows.append([k, v if v is not None else '未提供', status])
    rows += [['训练数据', '仅官方训练集', experiment['status'] if experiment else '本材料未关联正式训练'],
             ['验证集使用', '仅验证/早停/checkpoint选择，不并入训练', '工程策略'],
             ['外部医学数据/医学预训练', '禁止', '工程策略'],
             ['测试集训练/伪标签/调参', '禁止', '工程策略'],
             ['自监督/蒸馏', '第一阶段未启用；后续须仅用官方训练数据及合规师生权重并申报', '未使用'],
             ['推理精度', '见实际submission.csv.metadata.json', '待最终提交'],
             ['500MB/T4测速/一致性', '见实际inference_files及cost/report.json', '待T4实测']]
    if ingest:
        rows += [['数据包路径', ingest['archive'], ingest['status']],
                 ['数据包SHA256', ingest.get('archive_sha256', '未完成'), ingest['status']],
                 ['数据包来源声明', ingest['source'], '用户提供，下载凭据仍须留存'],
                 ['患者ID', str(ingest['patient_id_available']), '未提供时不能排除患者级泄漏'],
                 ['数据审计', str(ingest.get('audit_passed', '未完成')), ingest['status']]]
    if experiment:
        rows += [['实验ID', experiment['experiment_id'], experiment['status']],
                 ['配置SHA256', experiment['config_hash'], '真实登记'],
                 ['T4端到端均值', experiment.get('t4_latency_ms'), '空值表示未实测']]
    xlsx(out / '模型核验材料与数据声明清单.xlsx', rows)
    write_json(out / 'report.json', {'status': status, 'manifest_sha256': sha256(a.manifest), 'provenance': m,
        'official_data_connected': bool(ingest and ingest.get('status') in ['completed', 'audit_blocked']),
        'data_ingest': ingest, 'formal_training_started': bool(experiment and experiment['status'] in ['running', 'completed', 'reused']),
        'experiment': experiment,
        'note': 'This is a provenance checklist, not a license/legal or official certification.'})


if __name__ == '__main__':
    main()
