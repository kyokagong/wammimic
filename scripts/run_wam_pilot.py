"""Run the human WAM pilot after optical operation caches finish validation."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from scripts.run_phase0 import save,file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('/data/kk/data'))
    parser.add_argument('--model-root',type=Path,default=Path('/data/kk/models'))
    parser.add_argument('--name',default='human-ego-pilot-20261118')
    parser.add_argument('--epochs',type=int,default=3);parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args();data=args.data_root.resolve();models=args.model_root.resolve()
    base=data/'wammimic_human_ego_20261117';operations=data/'wammimic_operations_processed_20261118'
    control=data/'wammimic_downloads_20261116';protocol=data/'wammimic_wam_protocol_20261118';protocol.mkdir(exist_ok=True)
    run=models/'wammimic_runs'/args.name
    if run.resolve().parent!=models/'wammimic_runs':raise ValueError('Unsafe run name')
    run.mkdir(parents=True,exist_ok=True)
    budget=json.loads((control/'budget_100g.json').read_text())
    if budget['limit_bytes']!=100000000000 or budget['projected_reserved_bytes']>budget['limit_bytes']:raise ValueError('Download budget exceeded')
    preserved=json.loads((base/'qa_reports/mimic_preserved_before.json').read_text())
    if any(file_hash(Path(p))!=v for p,v in preserved.items()):raise ValueError('Mimic baseline changed')
    state={'scope':'Cosmos-Predict2.5-2B human five-point pilot','complete':False,'mimic_preserved':True,'stages':{}}
    save(run/'pilot_status.json',state)
    try:
        while not (operations/'qa_reports/human_wam_training_ready.json').exists():
            state['phase']='waiting for independently validated operation caches';save(run/'pilot_status.json',state);time.sleep(20)
        if not json.loads((operations/'qa_reports/human_wam_training_ready.json').read_text())['ready']:raise ValueError('Operation cache validation failed')
        for stage in ['protocol','train','evaluate','visualize']:
            if stage=='protocol' and (protocol/'protocol.json').exists():continue
            if stage=='train' and (run/'training_summary.json').exists():continue
            if stage=='evaluate' and (run/'evaluation_summary.json').exists():continue
            if stage=='visualize' and (run/'visualizations/report.html').exists():continue
            state['stages'][stage]='processing';state['phase']=stage;save(run/'pilot_status.json',state)
            command=['docker','run','--rm','--network','none','--cpus','4','--memory','32g',
                '-v',f'{data}:{data}:ro','-v',f'{models}:{models}:ro','-v',f'{run}:/runs',
                '-v',f'{protocol}:{protocol}'+('' if stage=='protocol' else ':ro'),
                '-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-e','HF_HUB_DISABLE_TELEMETRY=1']
            if stage in ['train','evaluate']:
                free=int(subprocess.check_output(['nvidia-smi',f'--id={args.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<12000:raise RuntimeError('Pilot GPU has insufficient available memory')
                command+=['--gpus',f'device={args.gpu}']
            command+=['wammimic:train-pilot',stage,'--protocol',str(protocol/'protocol.json'),'--run-root','/runs']
            if stage=='protocol':command+=['--base-data',str(base),'--operation-data',str(operations),
                '--operation-manifest',str(control/'robomocap-operations-20261118.json'),
                '--raw-roots',str(data/'robomocap_raw'),str(data/'robomocap_operations_raw_20261118')]
            elif stage in ['train','evaluate']:command+=['--backbone',str(models/'nvidia/Cosmos-Predict2.5-2B'),
                '--backbone-manifest',str(control/'cosmos_backbone_20261118.json'),'--epochs',str(args.epochs)]
            with (run/(stage+'.log')).open('ab',buffering=0) as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['stages'][stage]='complete';save(run/'pilot_status.json',state)
        if any(file_hash(Path(p))!=v for p,v in preserved.items()):raise ValueError('Mimic changed during pilot')
        state.update(complete=True,phase='training, independent evaluation and visualization complete');save(run/'pilot_status.json',state)
    except BaseException as error:
        state.update(complete=False,phase='failed',error=repr(error));save(run/'pilot_status.json',state);raise


if __name__=='__main__':main()
