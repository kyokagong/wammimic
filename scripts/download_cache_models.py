"""Download only pinned files from an authorized model or dataset manifest."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time

import requests
from urllib3.exceptions import HTTPError

RANGE_BYTES=4*1024*1024


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    digest=hashlib.sha256() if record.get("sha256") else hashlib.sha1()
    if not (record.get("sha256") or record.get("sha1")):
        digest.update(f"blob {record['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()==(record.get("sha256") or record.get("sha1") or record["blob_sha1"])


def dataset_records(manifest,budget,root):
    plan=next(p for p in budget["current_subset_plans"] if Path(p["root"]).resolve()==root.resolve())
    files=[{"repo":manifest["id"],"revision":manifest["sha"],"remote":f["rfilename"],
            "path":f["rfilename"],"size":f["size"],"sha256":f.get("lfs",{}).get("sha256"),
            "blob_sha1":f["blobId"]} for f in manifest["siblings"]]
    return {"files":files,"total_bytes":sum(f["size"] for f in files),"cap_bytes":plan["cap_bytes"]}


def download_ranges(url,path,record,auth_headers=None,*,workers=8):
    """Bound each HTTP transfer so a slow large response cannot block a file."""
    prefix=path.with_name(path.name+".download.part")
    chunks=path.with_name(path.name+".ranges");marker=chunks/"source.json"
    identity={k:record.get(k) for k in ["size","sha256","blob_sha1"]}
    if record.get("sha1"):identity["sha1"]=record["sha1"]
    range_bytes=RANGE_BYTES if record.get("sha256") or record.get("sha1") else record["size"]
    if marker.exists():
        source=json.loads(marker.read_text())
        if source["identity"]!=identity:raise ValueError("Range fragments belong to another source")
        begin=source["prefix_bytes"]
    else:
        if chunks.exists() and any(chunks.iterdir()):raise FileExistsError("Unowned range fragments preserved")
        begin=prefix.stat().st_size if prefix.exists() else 0
        if begin>record["size"]:raise ValueError("Oversized download prefix")
        if not (record.get("sha256") or record.get("sha1")):
            prefix.unlink(missing_ok=True);begin=0
        chunks.mkdir(parents=True,exist_ok=True)
        marker.write_text(json.dumps({"identity":identity,"prefix_bytes":begin}))
    if (prefix.stat().st_size if prefix.exists() else 0)!=begin:raise ValueError("Download prefix changed")

    connections=threading.local();sessions=[]
    def fetch(start):
        if not hasattr(connections,"session"):
            connections.session=requests.Session();sessions.append(connections.session)
        end=min(start+range_bytes,record["size"])-1
        target=chunks/str(start);partial=target.with_suffix(".tmp");size=end-start+1
        if target.exists() and target.stat().st_size==size:return
        for attempt in range(8):
            try:
                offset=partial.stat().st_size if partial.exists() else 0
                if offset>size:raise ValueError("Oversized range fragment")
                if offset<size:
                    with connections.session.get(url,params={"download":"true","resume_at":start+offset,"range_end":end,"request_at":time.time_ns()} if record.get("sha256") else None,
                        headers={**(auth_headers or {}),"Range":f"bytes={start+offset}-{end}"},stream=True,timeout=(20,45)) as response:
                        response.raise_for_status()
                        if response.status_code==200 and start==0 and size==record["size"]:
                            partial.unlink(missing_ok=True);offset=0
                        elif response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start+offset}-{end}/{record['size']}":
                            raise ValueError("Server did not honor the bounded range")
                        deadline=time.monotonic()+90
                        read_block=getattr(response.raw,"read1",response.raw.read)
                        with partial.open("ab") as stream:
                            while block:=read_block(64*1024,decode_content=True):
                                if time.monotonic()>deadline:raise requests.Timeout("Bounded transfer exceeded 90 seconds")
                                if offset+len(block)>size:raise ValueError("Oversized range response")
                                stream.write(block);offset+=len(block)
                if partial.stat().st_size!=size:raise ValueError("Incomplete range response")
                os.replace(partial,target)
                return
            except (requests.RequestException,HTTPError,ValueError):
                if attempt==7:raise
                time.sleep(min(30,2*(attempt+1)))
    starts=range(begin,record["size"],range_bytes)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(fetch,starts):pass
    finally:
        for session in sessions:session.close()
    assembled=chunks/"assembled.tmp"
    with assembled.open("wb") as dest:
        if begin:
            with prefix.open("rb") as stream:shutil.copyfileobj(stream,dest,8*1024*1024)
        for start in starts:
            with (chunks/str(start)).open("rb") as stream:shutil.copyfileobj(stream,dest,8*1024*1024)
    if not verify(assembled,record):
        prefix.unlink(missing_ok=True);shutil.rmtree(chunks)
        raise ValueError("Range download checksum mismatch; corrupt owned fragments cleared")
    os.replace(assembled,path);prefix.unlink(missing_ok=True);shutil.rmtree(chunks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--budget",required=True,type=Path)
    parser.add_argument("--dataset",action="store_true",help="Read an already-budgeted dataset subset manifest")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--range-workers",type=int,default=8)
    parser.add_argument("--token-file",type=Path,help="Private credential file; never stored in manifests or receipts")
    args=parser.parse_args()
    if args.workers<1 or args.range_workers<1:parser.error("Worker counts must be positive")
    auth_headers={"Authorization":"Bearer "+args.token_file.read_text().strip()} if args.token_file else {}
    manifest=json.loads(args.manifest.read_text());budget=json.loads(args.budget.read_text())
    if args.dataset:manifest=dataset_records(manifest,budget,args.root)
    if (budget["limit_bytes"]!=100000000000 or budget["projected_reserved_bytes"]>budget["limit_bytes"]
        or manifest["total_bytes"]>manifest["cap_bytes"]
        or sum(f["size"] for f in manifest["files"])!=manifest["total_bytes"]
        or len({f["path"] for f in manifest["files"]})!=len(manifest["files"])):
        raise ValueError("Authorized download budget exceeded")
    args.root.mkdir(parents=True,exist_ok=True)

    def download(record):
        path=args.root/record["path"]
        if not path.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("Unsafe output path")
        part=path.with_name(path.name+".download.part")
        if verify(path,record):
            part.unlink(missing_ok=True)
            return
        if path.exists():raise ValueError(f"Preserved invalid existing file: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
        endpoint="resolve" if record.get("sha256") else "raw"
        url=record.get("url") or f"https://huggingface.co/{'datasets/' if args.dataset else ''}{record['repo']}/{endpoint}/{record['revision']}/{record['remote']}"
        try:
            download_ranges(url,path,record,auth_headers,workers=args.range_workers)
        except (requests.RequestException,HTTPError) as error:
            if record.get("url"):
                raise RuntimeError(f"{type(error).__name__} while downloading {record['path']}; signed URL omitted") from None
            raise
        print("Verified",record["path"],flush=True)
    # Small VAE first, so video caching can be tested while text shards download.
    files=sorted(manifest["files"],key=lambda f:(f["path"]!="tokenizer.pth",f["size"]))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download,files))
    report={"complete":True,"verified_files":len(files),"total_bytes":sum(f["size"] for f in files),
            "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    receipt=args.root/("wammimic_100g_verification.json" if args.dataset else "verification.json")
    temporary=receipt.with_suffix(".tmp");temporary.write_text(json.dumps(report,indent=2));os.replace(temporary,receipt)


if __name__=="__main__":main()
