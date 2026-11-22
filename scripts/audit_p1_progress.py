"""Write actual P1 progress; a selected file is not a completed download."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from wammimic.data_pipeline.common import sha256, write_json


def file_size(path):
    try:
        if path.is_file():return path.stat().st_size
    except FileNotFoundError:pass
    return None


def audit(root):
    root=Path(root);control=root/'control'
    budget=json.loads((control/'budget_100g.json').read_text())
    if budget['projected_reserved_bytes']>budget['limit_bytes']:raise ValueError('P1 download budget exceeded')
    sources=json.loads((Path(__file__).resolve().parents[1]/'conf/data/pipeline.yaml').read_text())['sources']
    downloads={}
    for name in ['robomocap','openhlm','nymeria']:
        manifest_name=sources[name]['download_manifest'];receipt_name='verification.json' if name=='nymeria' else 'wammimic_100g_verification.json'
        path=control/manifest_name
        if not path.exists():continue
        manifest=json.loads(path.read_text());files=manifest.get('siblings',manifest.get('files',[]));dest=root/'raw'/sources[name]['raw']
        complete=[]
        for f in files:
            local=dest/f.get('rfilename',f.get('path'))
            if file_size(local)==f['size']:complete.append(f)
        receipt=dest/receipt_name;verified=False
        if receipt.exists():
            value=json.loads(receipt.read_text())
            verified=bool(value.get('complete') and value.get('manifest_sha256')==sha256(path) and len(complete)==len(files))
        partial=sum(file_size(p) or 0 for p in dest.rglob('*.ranges/*')
                    if p.name.removesuffix('.tmp').isdigit()) if dest.exists() else 0
        downloads[name]={'selected_files':len(files),'size_complete_files':len(complete),
            'selected_bytes':sum(f['size'] for f in files),'size_complete_bytes':sum(f['size'] for f in complete),
            'retained_partial_bytes':partial,'all_hashes_verified':verified}
    branches={p.stem.removesuffix('-pipeline-status'):json.loads(p.read_text()) for p in control.glob('*-pipeline-status.json')}
    human={}
    for folder in (root/'processed').glob('human-*'):
        receipts=folder/'wam_lerobot_v2/human_ego/receipts'
        rows=[json.loads(p.read_text()) for p in receipts.glob('*.json')]
        if rows:human[folder.name]={'prepared_sessions':len(rows),'valid_windows':sum(r['valid_samples'] for r in rows),
            'train_sessions':sum(r['split']=='train' for r in rows),
            'train_windows':sum(r['valid_samples'] for r in rows if r['split']=='train')}
    nymeria=root/'processed'/sources['nymeria']['output']/'qa_reports/nymeria_wam_training_ready.json'
    sim_ready=root/'processed'/sources['sim']['output']/'qa_reports/sim_wam_training_ready.json'
    gripper_ready=root/'processed'/sources['openhlm']['output']/'qa_reports/wam_training_ready.json'
    gripper_vis=root/'processed'/sources['openhlm']['output']/'qa_reports/gripper_visual_review.json'
    sim_vis=root/'processed'/sources['sim']['output']/'qa_reports/sim_visual_review.json'
    nymeria_doc=json.loads(nymeria.read_text()) if nymeria.exists() else {'ready':False}
    sim_doc=json.loads(sim_ready.read_text()) if sim_ready.exists() else {'ready':False}
    gripper_doc=json.loads(gripper_ready.read_text()) if gripper_ready.exists() else {'ready':False}
    gripper_vis_doc=json.loads(gripper_vis.read_text()) if gripper_vis.exists() else {'visual_gate_passed':False}
    sim_vis_doc=json.loads(sim_vis.read_text()) if sim_vis.exists() else {'visual_gate_passed':False}
    required_branches=['human-base','human-new','human-operations','human-protocol','gripper','sim','nymeria']
    p1_ready=(all(r['all_hashes_verified'] for r in downloads.values())
              and all(branches.get(b,{}).get('complete',False) for b in required_branches)
              and bool(nymeria_doc.get('ready')) and bool(sim_doc.get('ready')) and bool(gripper_doc.get('ready'))
              and bool(gripper_vis_doc.get('visual_gate_passed')) and bool(sim_vis_doc.get('visual_gate_passed')))
    scope_limits=[
        'Dexterous hands deferred by user; gripper data prioritized',
        'Nymeria three selected recordings are not the planned 50-100 hours; inertial protocol remains separate',
        'New WAM training and D4 rollout gate have not run yet',
    ]
    if not sim_doc.get('ready'):
        scope_limits.insert(0, 'D1 sim branch not yet validated and ready')
    if not gripper_vis_doc.get('visual_gate_passed') or not sim_vis_doc.get('visual_gate_passed'):
        scope_limits.append('Visual review contact sheets pending for sim and/or gripper')
    report={'updated_at_utc':datetime.now(timezone.utc).isoformat(),'p1_training_ready':p1_ready,
        'nymeria_training_data':nymeria_doc,
        'sim_training_data':sim_doc,
        'gripper_training_data':gripper_doc,
        'visual_reviews':{'gripper':gripper_vis_doc.get('visual_gate_passed',False),'sim':sim_vis_doc.get('visual_gate_passed',False)},
        'additional_download_limit_bytes':budget['limit_bytes'],'reserved_bytes':budget['projected_reserved_bytes'],
        'downloads':downloads,'pipelines':branches,'human_history':human,
        'scope_limits':scope_limits}
    write_json(control/'progress.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    root=Path(os.environ['P1_RUN_ROOT'])
    while True:
        report=audit(root)
        if not args.watch:print(json.dumps(report,ensure_ascii=False));return
        downloads_done=all(r['all_hashes_verified'] for r in report['downloads'].values())
        pipelines_done=all(r['complete'] or r['phase'].startswith('blocked') for r in report['pipelines'].values())
        if downloads_done and pipelines_done:return
        time.sleep(60)


if __name__=='__main__':main()
