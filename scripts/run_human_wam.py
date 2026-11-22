"""Prepare the measured human ego branch without writing to the Mimic store.

Run from the repository root: python -m scripts.run_human_wam
"""
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_phase0 import available_gpu,download_verified,restart_download,save,file_hash


def main():
    raw=Path(os.environ["RAW_DATA_ROOT"]).resolve()
    output=Path(os.environ["PROCESSED_DATA_ROOT"]).resolve()
    models=Path(os.environ["PRETRAINED_ROOT"]).resolve()
    control=raw/"wammimic_downloads_20261116";manifest=control/"robomocap-human-100g.json"
    if output==raw/"wammimic_processed_20261116" or (output/"morph_5pt_store").exists():
        raise ValueError("Human WAM requires a separate output directory")
    preserved=json.loads((output/"qa_reports/mimic_preserved_before.json").read_text())
    if any(file_hash(Path(path))!=digest for path,digest in preserved.items()):
        raise ValueError("Mimic preservation baseline changed before processing")
    budget=json.loads((control/"budget_100g.json").read_text())
    if budget["projected_reserved_bytes"]>budget["limit_bytes"] or budget["limit_bytes"]!=100000000000:
        raise ValueError("Human data preparation exceeds cumulative download budget")
    status=output/"pipeline_status.json"
    state={"scope":"human ego optical five-point data","wam_ready":False,"mimic_preserved":True,"stages":{}}
    save(status,state)
    while not download_verified(raw/"robomocap_raw",manifest):
        restart_download(control,raw/"robomocap_raw",manifest.name,"human",dataset=True)
        state["downloads"]="waiting for verified complete human selection";save(status,state);time.sleep(30)
    state["downloads"]="complete and verified"
    for stage in ["human-ego","cache-human-text","cache-human-video","validate-human-wam"]:
        state["stages"][stage]="processing";save(status,state)
        gpu=stage in {"cache-human-text","cache-human-video"}
        args=["docker","run","--rm","--network","none","--cpus","4","--memory","24g" if gpu else "12g",
            "-v",f"{raw}:{raw}:ro","-v",f"{output}:/mnt/human_data","-v",f"{models}:/mnt/pretrained:ro",
            "-e",f"RAW_DATA_ROOT={raw}","-e","PROCESSED_DATA_ROOT=/mnt/human_data","-e","PRETRAINED_ROOT=/mnt/pretrained",
            "-e",f"HUMAN_DOWNLOAD_MANIFEST={manifest}","-e","HF_HUB_OFFLINE=1","-e","TRANSFORMERS_OFFLINE=1"]
        if gpu:
            while (index:=available_gpu(20000 if stage=="cache-human-text" else 4000)) is None:
                print(f"{stage}: waiting for GPU memory",flush=True);time.sleep(30)
            args += ["--gpus",f"device={index}"]
        args += ["wammimic:data-phase0",stage]
        with (output/f"{stage}.log").open("ab",buffering=0) as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        state["stages"][stage]="complete";save(status,state)
    report=json.loads((output/"qa_reports/human_wam_training_ready.json").read_text())
    if not report["ready"] or not report["mimic_preserved"]:raise ValueError("Human readiness validation failed")
    state.update(wam_ready=True,mimic_preserved=True,phase="human training inputs ready")
    save(status,state);print(json.dumps(state),flush=True)


if __name__=="__main__":
    try:main()
    except Exception as error:
        path=Path(os.environ["PROCESSED_DATA_ROOT"])/"pipeline_status.json"
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(wam_ready=False,phase="failed",error=repr(error));save(path,state)
        raise
