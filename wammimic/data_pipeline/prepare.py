"""Run Phase 0 with explicit mounted paths; raw sources are never modified."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["morph","openhlm","upgrade-openhlm-5pt","cache-text","cache-video","validate-mimic","validate-wam",
        "human-ego","cache-human-text","cache-human-video","validate-human-wam","sim","cache-sim-text","cache-sim-video","validate-sim",
        "nymeria","cache-nymeria-text","cache-nymeria-video","validate-nymeria"])
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--human-history",action="store_true",help="Export four causal measured poses and velocities into a new human dataset")
    args=parser.parse_args()
    output=Path(os.environ["PROCESSED_DATA_ROOT"])
    if args.stage=="nymeria":
        from .nymeria import import_nymeria
        result=import_nymeria(Path(os.environ['RAW_DATA_ROOT'])/os.environ.get('NYMERIA_RAW_SUBDIR','nymeria'),os.environ['NYMERIA_DOWNLOAD_MANIFEST'],output)
    elif args.stage=="cache-nymeria-text":
        from .nymeria import DATASET
        from .cache import cache_text
        result=cache_text(os.environ['PRETRAINED_ROOT'],output/DATASET/'meta/tasks.jsonl',output)
    elif args.stage=="cache-nymeria-video":
        import torch
        torch.set_num_threads(args.workers)
        from .nymeria import NAME
        from .cache import cache_videos
        result=cache_videos(os.environ['PRETRAINED_ROOT'],output,batch_size=32,dataset_name=NAME)
    elif args.stage=="validate-nymeria":
        from .nymeria import validate_nymeria
        result=validate_nymeria(output)
    elif args.stage=="sim":
        from .sim import import_sim
        result=import_sim(Path(os.environ["RAW_DATA_ROOT"])/os.environ.get("SIM_RAW_SUBDIR","sim_loco_tabletop"),output)
    elif args.stage=="cache-sim-text":
        from .cache import cache_text
        result=cache_text(os.environ["PRETRAINED_ROOT"],output/"wam_lerobot_v2/q1_stub/meta/tasks.jsonl",output)
    elif args.stage=="cache-sim-video":
        from .cache import cache_videos
        result=cache_videos(os.environ["PRETRAINED_ROOT"],output,dataset_name="q1_stub")
    elif args.stage=="validate-sim":
        from .sim import validate_sim
        result=validate_sim(output)
    elif args.stage=="upgrade-openhlm-5pt":
        from .openhlm import upgrade_openhlm_5pt_contract
        result=upgrade_openhlm_5pt_contract(output)
    elif args.stage=="human-ego":
        from .human_ego import import_human_ego
        result=import_human_ego(os.environ.get("ROBOMOCAP_RAW_ROOT",str(Path(os.environ["RAW_DATA_ROOT"])/"robomocap_raw")),os.environ["HUMAN_DOWNLOAD_MANIFEST"],output,history=args.human_history)
    elif args.stage=="cache-human-text":
        from .cache import cache_text
        result=cache_text(os.environ["PRETRAINED_ROOT"],output/"wam_lerobot_v2/human_ego/meta/tasks.jsonl",output)
    elif args.stage=="cache-human-video":
        from .cache import cache_videos
        result=cache_videos(os.environ["PRETRAINED_ROOT"],output,dataset_name="human_ego")
    elif args.stage=="validate-human-wam":
        from .human_ego import validate_human_wam
        result=validate_human_wam(os.environ.get("ROBOMOCAP_RAW_ROOT",str(Path(os.environ["RAW_DATA_ROOT"])/"robomocap_raw")),os.environ["HUMAN_DOWNLOAD_MANIFEST"],output)
    elif args.stage=="morph":
        from .morph import import_morph
        result=import_morph(os.environ["MORPH_MANIFEST"],os.environ["MORPH_SOURCE_PREFIX"],
                           os.environ["MORPH_PREPARED_ROOT"],output/"morph_5pt_store",args.workers)
    elif args.stage=="openhlm":
        from .openhlm import import_openhlm
        raw=Path(os.environ["RAW_DATA_ROOT"])
        result=import_openhlm(raw,os.environ["OPENHLM_DOWNLOAD_MANIFEST"],output,
                             raw/"supermimic_g1/g1_description/g1_29dof.xml")
    elif args.stage=="cache-text":
        from .cache import cache_text
        result=cache_text(os.environ["PRETRAINED_ROOT"],Path(os.environ["RAW_DATA_ROOT"])/"openhlm_raw/g1_HLM-12_full_teleop/meta/tasks.jsonl",output)
    elif args.stage=="cache-video":
        from .cache import cache_videos
        result=cache_videos(os.environ["PRETRAINED_ROOT"],output)
    elif args.stage=="validate-mimic":
        from .validate import validate_mimic
        result=validate_mimic(output)
    else:
        from .validate import validate_wam
        result=validate_wam(output)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":main()
