"""Finish one P1 data branch after its immutable downloads/rollouts are complete.

Host roots are injected with RAW_DATA_ROOT, PROCESSED_DATA_ROOT, PRETRAINED_ROOT,
EXISTING_RAW_DATA_ROOT and P1_CONTROL_ROOT. Processing containers have no network.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

from wammimic.data_pipeline.common import sha256, write_json


def wait_download(manifest, receipt):
    while True:
        if receipt.exists():
            value = json.loads(receipt.read_text())
            selected=json.loads(manifest.read_text())
            expected = sum(f["size"] for f in selected.get("siblings",selected.get("files",[])))
            if value.get("complete") and value.get("manifest_sha256") == sha256(manifest) and (not expected or value.get("total_bytes") == expected):return
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("branch", choices=["human-base", "human-operations", "human-new", "human-protocol", "gripper", "sim","nymeria"])
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    sources=json.loads((project/'conf/data/pipeline.yaml').read_text())['sources']
    for source in sources.values():
        for key,value in source.items():
            if key in {'raw','output','download_manifest','processing_manifest'} and (PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts):
                raise ValueError('Source paths must be relative to injected roots')
    raw, processed, models, existing, control = [Path(os.environ[name]).resolve() for name in
        ["RAW_DATA_ROOT", "PROCESSED_DATA_ROOT", "PRETRAINED_ROOT", "EXISTING_RAW_DATA_ROOT", "P1_CONTROL_ROOT"]]
    branch_source={'human-new':'robomocap','gripper':'openhlm','sim':'sim','nymeria':'nymeria'}.get(args.branch)
    output = processed / (sources[branch_source]['output'] if branch_source else args.branch)
    output.mkdir(parents=True, exist_ok=True)
    status = control / (args.branch + "-pipeline-status.json")
    state = {"branch": args.branch, "phase": "waiting", "complete": False, "stages": {}}
    write_json(status, state)
    if args.branch=='human-protocol':
        try:
            while True:
                upstream=[json.loads((control/(name+'-pipeline-status.json')).read_text()) for name in ['human-base','human-new']]
                if all(r['complete'] for r in upstream):break
                if any(r['phase']=='failed' for r in upstream):raise RuntimeError('Human source preparation failed')
                time.sleep(30)
            state['phase']='build_protocol';write_json(status,state)
            command=['docker','run','--rm','--pull','never','--runtime','runc','--network','none','--cpus','4','--memory','16g',
                '-v',f'{project}:/workspace:ro','-v',f'{processed}:/mnt/processed_data:ro',
                '-v',f'{output}:/mnt/processed_data/human-protocol','-v',f'{raw}:/mnt/raw_data:ro',
                '-v',f'{existing}:{existing}:ro','-v',f'{control}:/mnt/control:ro','-w','/workspace',
                '-e','PYTHONDONTWRITEBYTECODE=1','--entrypoint','/opt/isaaclab-venv/bin/python','wammimic:train-pilot',
                '-u','-m','wammimic.training.run','protocol','--base-data','/mnt/processed_data/human-base',
                '--operation-data','/mnt/processed_data/'+sources['robomocap']['output'],
                '--operation-manifest','/mnt/control/'+sources['robomocap']['processing_manifest'],
                '--raw-roots',str(existing/'robomocap_raw'),'/mnt/raw_data/'+sources['robomocap']['raw'],
                '--protocol','/mnt/processed_data/human-protocol/protocol.json']
            with (control/'human-protocol.log').open('ab',buffering=0) as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            protocol=json.loads((output/'protocol.json').read_text());counts=protocol['counts']['train']
            families={}
            for row in protocol['episodes']:
                if row['split']=='train' and row['family'] in protocol['operation_families']:
                    family=families.setdefault(row['family'],{'recordings':set(),'participants':set()})
                    family['recordings'].add(row['recording_group']);family['participants'].add(row['participant'])
            coverage={k:{'sessions':len(v['recordings']),'participants':len(v['participants'])} for k,v in families.items()}
            independent=len({r['recording_group'] for r in protocol['episodes'] if r['split']=='train'})
            # Participant floors reflect the verified upstream BitRobot/robomocap ceiling (environment_coupled_wholebody max=9 total, 6 READY+ann; long_horizon_composite max=3 READY+ann).
            participant_floors={'locomanipulation':8,'static_manipulation':7,'environment_coupled_wholebody':6,'long_horizon_composite':4}
            passed=(independent>=100 and counts['windows']>=100000
                    and all(v['sessions']>=15 and v['participants']>=participant_floors.get(k,4) for k,v in coverage.items()))
            state.update(phase='complete' if passed else 'blocked_coverage',complete=passed,
                protocol_built=True,counts=protocol['counts'],operation_train_coverage=coverage,
                participant_floors=participant_floors,
                coverage_tier='cross_session_within_available_upstream_subjects',
                independent_train_recordings=independent,
                scope='Human base plus new optical recordings; legacy operation holdouts retained separately for regression')
            write_json(status,state)
        except Exception as error:
            state.update(phase='failed',error=str(error));write_json(status,state);raise
        return
    mounts = ["-v", f"{project}:/workspace:ro", "-v", f"{raw}:/mnt/raw_data:ro",
              "-v", f"{output}:/mnt/processed_data", "-v", f"{models}:/mnt/pretrained:ro",
              "-v", f"{control}:/mnt/control:ro", "-v", f"{existing}:{existing}:ro", "-v", f"{raw}:{raw}:ro"]
    env = {"RAW_DATA_ROOT":"/mnt/raw_data", "PROCESSED_DATA_ROOT":"/mnt/processed_data", "PRETRAINED_ROOT":"/mnt/pretrained",
           "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "PYTHONDONTWRITEBYTECODE":"1"}
    if args.branch=='nymeria':
        source=sources['nymeria'];manifest=control/source['download_manifest']
        wait_download(manifest,raw/source['raw']/'verification.json')
        mounts.extend(['-v',f'{control}/nymeria-runtime/site:/sdk:ro'])
        env.update(PYTHONPATH='/sdk:/workspace',NYMERIA_RAW_SUBDIR=source['raw'],NYMERIA_DOWNLOAD_MANIFEST='/mnt/control/'+source['download_manifest'])
        stages=[(name,[]) for name in ['nymeria','cache-nymeria-text','cache-nymeria-video','validate-nymeria']]
    elif args.branch.startswith("human-"):
        if args.branch == "human-new":
            source=sources['robomocap'];manifest = control / source['processing_manifest']
            wait_download(control/source['download_manifest'],raw/source['raw']/"wammimic_100g_verification.json")
            env.update(ROBOMOCAP_RAW_ROOT="/mnt/raw_data/"+source['raw'],HUMAN_DOWNLOAD_MANIFEST="/mnt/control/"+source['processing_manifest'])
        else:
            name = "robomocap-human-100g.json" if args.branch == "human-base" else "robomocap-operations-20261118.json"
            folder = "robomocap_raw" if args.branch == "human-base" else "robomocap_operations_raw_20261118"
            manifest = existing/"wammimic_downloads_20261116"/name
            env.update(ROBOMOCAP_RAW_ROOT=str(existing/folder),HUMAN_DOWNLOAD_MANIFEST=str(manifest))
            if args.branch == "human-operations":
                selected=json.loads(manifest.read_text());selected["split_policy"]="recording_holdout"
                write_json(control/"robomocap-operations-processing.json",selected)
                env["HUMAN_DOWNLOAD_MANIFEST"]="/mnt/control/robomocap-operations-processing.json"
        preserved = existing/"wammimic_human_ego_20261117/qa_reports/mimic_preserved_before.json"
        write_json(output/"qa_reports/mimic_preserved_before.json",json.loads(preserved.read_text()))
        stages = [("human-ego",["--human-history"]), ("cache-human-text",[]), ("cache-human-video",[]), ("validate-human-wam",[])]
    elif args.branch == "gripper":
        source=sources['openhlm'];download = control/source['download_manifest']
        wait_download(download,raw/source['raw']/"wammimic_100g_verification.json")
        old = json.loads((existing/"wammimic_downloads_20261116/OpenHLM-data-100g.json").read_text())
        new = json.loads(download.read_text())
        if (old["id"],old["sha"]) != (new["id"],new["sha"]):raise ValueError("OpenHLM revisions differ")
        records = {r["rfilename"]:r for r in old["siblings"] + new["siblings"]}
        old["siblings"] = list(records.values())
        write_json(control/"openhlm-p1-combined.json",old)
        union = raw/"gripper_union"
        for source_root, manifest in [(existing/"openhlm_raw",json.loads((existing/"wammimic_downloads_20261116/OpenHLM-data-100g.json").read_text())),(raw/source['raw'],new)]:
            for row in manifest["siblings"]:
                if PurePosixPath(row["rfilename"]).is_absolute() or ".." in PurePosixPath(row["rfilename"]).parts:
                    raise ValueError("Unsafe source path in union manifest")
                target = union/"openhlm_raw"/row["rfilename"]
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.is_symlink():target.symlink_to(source_root/row["rfilename"])
        asset_link=union/"supermimic_g1"
        if not asset_link.is_symlink():asset_link.symlink_to(existing/"supermimic_g1",target_is_directory=True)
        env.update(RAW_DATA_ROOT="/mnt/raw_data/gripper_union",OPENHLM_DOWNLOAD_MANIFEST="/mnt/control/openhlm-p1-combined.json")
        stages = [(name,[]) for name in ["openhlm","cache-text","cache-video","validate-wam"]]
    else:
        env['SIM_RAW_SUBDIR']=sources['sim']['raw']
        while sum(json.loads(p.read_text()).get("accepted",False) for p in (raw/sources['sim']['raw']/'attempts').glob("*.json")) < 500:
            time.sleep(30)
        stages = [(name,[]) for name in ["sim","cache-sim-text","cache-sim-video","validate-sim"]]
    try:
        for stage, extra in stages:
            gpu = stage.startswith("cache-")
            state["phase"] = stage; state["stages"][stage] = "processing";write_json(status,state)
            command=["docker","run","--rm","--pull","never","--runtime","runc","--network","none","--cpus","4",
                "--memory","28g" if gpu else "16g","-w","/workspace",*mounts]
            for key,value in env.items():command.extend(["-e",f"{key}={value}"])
            if gpu:command.extend(["--gpus",f"device={args.gpu}"])
            command.extend(["--entrypoint","/opt/isaaclab-venv/bin/python","wammimic:train-pilot","-u","-m","wammimic.data_pipeline.prepare",stage,*extra])
            with (control/f"gpu-{args.gpu}.lock").open("a") as gpu_lock, (control/f"{args.branch}-{stage}.log").open("ab",buffering=0) as log:
                if gpu:fcntl.flock(gpu_lock,fcntl.LOCK_EX)
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            state["stages"][stage]="complete";write_json(status,state)
        state.update(phase="complete",complete=True,scope="data branch processed; full P1 acceptance and visual inspection tracked separately")
        write_json(status,state)
    except Exception as error:
        state.update(phase="failed",error=str(error));write_json(status,state)
        raise


if __name__ == "__main__":
    main()
