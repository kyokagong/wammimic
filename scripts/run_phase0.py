"""Finish the bounded downloads, prepare paired datasets, record real blockers.

All host paths come from the environment. Runs under the host's existing tmux
server so leaving a chat does not terminate downloads or CPU preparation.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import hashlib


def save(path, value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    os.replace(temp,path)


def available_gpu(required_mib):
    query=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.free","--format=csv,noheader,nounits"],text=True)
    free,index=max((int(memory),int(index)) for index,memory in (line.split(",") for line in query.splitlines()))
    return index if free>=required_mib else None


def run_stage(stage,raw,output,prepared,prefix,models):
    gpu=stage in {"cache-text","cache-video"}
    command=["docker","run","--rm","--network","none","--cpus","4","--memory","24g" if gpu else "12g",
        "-v",f"{raw}:/mnt/raw_data:ro","-v",f"{output}:/mnt/processed_data",
        "-v",f"{prepared}:/mnt/morph_prepared:ro",
        "-v",f"{models}:/mnt/pretrained:ro","-e","PRETRAINED_ROOT=/mnt/pretrained",
        "-e","HF_HUB_OFFLINE=1","-e","TRANSFORMERS_OFFLINE=1",
        "-e","RAW_DATA_ROOT=/mnt/raw_data","-e","PROCESSED_DATA_ROOT=/mnt/processed_data",
        "-e","MORPH_PREPARED_ROOT=/mnt/morph_prepared",
        "-e",f"MORPH_SOURCE_PREFIX={prefix}",
        "-e","MORPH_MANIFEST=/mnt/morph_prepared/allopt-20261112/manifest.json",
        "-e","OPENHLM_DOWNLOAD_MANIFEST=/mnt/raw_data/wammimic_downloads_20261116/OpenHLM-data-100g.json",
        ]
    if gpu:
        while (index:=available_gpu(20000 if stage=="cache-text" else 4000)) is None:
            print(f"{stage}: waiting for available GPU memory",flush=True);time.sleep(30)
        command += ["--gpus",f"device={index}"]
    command += ["wammimic:data-phase0",stage]
    with (output/f"{stage}.log").open("ab",buffering=0) as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def file_hash(path):
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def component_available(models,manifest,kind):
    files=[f for f in manifest["files"] if (f["path"]=="tokenizer.pth")== (kind=="video")]
    return bool(files) and all((models/f["path"]).is_file() and (models/f["path"]).stat().st_size==f["size"] for f in files)


def download_verified(root,manifest_path):
    try:
        report=json.loads((root/"wammimic_100g_verification.json").read_text())
    except (OSError,json.JSONDecodeError):return False
    return report.get("complete") is True and report.get("manifest_sha256")==file_hash(manifest_path)


def restart_download(control,root,manifest,name,dataset=False):
    session="wammimic-download-"+name
    if subprocess.run(["tmux","has-session","-t",session],capture_output=True).returncode==0:return
    args=["docker","run","--rm","--network","host","--cpus","4","--memory","8g",
          "-v",f"{control}:/mnt/control:ro","-v",f"{root}:{root}"]
    for key in ["HTTPS_PROXY","HTTP_PROXY","ALL_PROXY","https_proxy","http_proxy","all_proxy"]:args += ["-e",key]
    args += ["--entrypoint","/opt/isaaclab-venv/bin/python","wammimic:data-phase0",
             "/workspace/scripts/download_cache_models.py","--manifest",f"/mnt/control/{manifest}",
             "--root",str(root),"--budget","/mnt/control/budget_100g.json"]
    if dataset:args += ["--dataset"]
    command="exec "+shlex.join(args)+" >> "+shlex.quote(str(control/("cosmos_models.log" if name=="cosmos" else name+"-100g.log")))+" 2>&1"
    subprocess.run(["tmux","new-session","-d","-s",session,command],check=True)


def main():
    raw=Path(os.environ["RAW_DATA_ROOT"]).resolve()
    output=Path(os.environ["PROCESSED_DATA_ROOT"]).resolve()
    prepared=Path(os.environ["MORPH_PREPARED_ROOT"]).resolve()
    prefix=os.environ["MORPH_SOURCE_PREFIX"]
    models=Path(os.environ["PRETRAINED_ROOT"]).resolve()
    output.mkdir(parents=True,exist_ok=True)
    control=raw/"wammimic_downloads_20261116"
    budget=json.loads((control/"budget_100g.json").read_text())
    if budget["limit_bytes"]!=100000000000 or budget["projected_reserved_bytes"]>budget["limit_bytes"]:
        raise ValueError("Download plan exceeds authorized budget")
    model_plan=json.loads((control/"cosmos_cache_models.json").read_text())
    state={"backbone":"Cosmos-Predict2.5-2B","nymeria":"excluded until official URLs arrive",
           "morph":"processing","openhlm":"waiting for selected downloads",
           "text_cache":"waiting for pinned weights","video_cache":"waiting for WAM and tokenizer",
           "mimic_ready":False,"wam_ready":False,"downloads":{}}
    status=output/"pipeline_status.json"
    save(status,state)
    morph_report=output/"morph_5pt_store/qa.json"
    if (not morph_report.exists() or not json.loads(morph_report.read_text()).get("complete")
        or json.loads(morph_report.read_text()).get("reference_floor_sha256")!=file_hash(prepared/"allopt-20261112/contact-floor.json")):
        run_stage("morph",raw,output,prepared,prefix,models)
    elif json.loads(morph_report.read_text())["source_manifest_sha256"]!=file_hash(prepared/"allopt-20261112/manifest.json"):
        raise ValueError("Prepared Morph source manifest changed")
    state["morph"]="standardized; see morph_5pt_store/qa.json for accepted/rejected clips"
    save(status,state)
    run_stage("validate-mimic",raw,output,prepared,prefix,models)
    state["mimic_ready"]=True
    save(status,state)
    sources=[("openhlm","OpenHLM-data"),("oakink2","OakInk-v2"),("egowalk","trajectories")]
    while True:
        state["model_downloads"]={kind:component_available(models,model_plan,kind) for kind in ["text","video"]}
        for name,manifest_name in sources:
            manifest=json.loads((control/(manifest_name+"-100g.json")).read_text())
            root=raw/(name+"_raw")
            missing=[f for f in manifest["siblings"] if not (root/f["rfilename"]).is_file()
                     or (root/f["rfilename"]).stat().st_size!=f["size"]]
            verified=download_verified(root,control/(manifest_name+"-100g.json"))
            state["downloads"][name]={"selected":len(manifest["siblings"]),"missing":len(missing),"verified":verified}
            if missing or not verified:
                restart_download(control,root,manifest_name+"-100g.json",name,dataset=True)
            elif name=="openhlm" and state["openhlm"]!="standardized":
                state["openhlm"]="processing";save(status,state)
                run_stage("openhlm",raw,output,prepared,prefix,models)
                report=json.loads((output/"qa_reports/openhlm.json").read_text())
                if report.get("pending") or not report.get("accepted"):
                    raise RuntimeError("No valid WAM batch; inspect qa_reports/openhlm.json")
                state["openhlm"]="standardized"
                run_stage("validate-mimic",raw,output,prepared,prefix,models)
        if not (component_available(models,model_plan,"text") and component_available(models,model_plan,"video")):
            restart_download(control,models,"cosmos_cache_models.json","cosmos")
        if state["text_cache"]!="complete" and component_available(models,model_plan,"text"):
            state["text_cache"]="processing";save(status,state)
            run_stage("cache-text",raw,output,prepared,prefix,models)
            state["text_cache"]="complete"
        if state["openhlm"]=="standardized" and state["video_cache"]!="complete" and component_available(models,model_plan,"video"):
            state["video_cache"]="processing";save(status,state)
            run_stage("cache-video",raw,output,prepared,prefix,models)
            state["video_cache"]="complete"
        if not state["wam_ready"] and state["text_cache"]==state["video_cache"]=="complete":
            run_stage("validate-wam",raw,output,prepared,prefix,models)
            state["wam_ready"]=True
        save(status,state)
        print(json.dumps(state,ensure_ascii=False),flush=True)
        if state["wam_ready"] and all(v["missing"]==0 and v["verified"] for v in state["downloads"].values()):
            break
        time.sleep(60)
    state["phase0"]="training data ready: paired WAM caches and Mimic references; see readiness reports for coverage and physical limits"
    save(status,state)


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        path=Path(os.environ["PROCESSED_DATA_ROOT"])/"pipeline_status.json"
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(phase0="failed",error=repr(error))
        save(path,state)
        raise
