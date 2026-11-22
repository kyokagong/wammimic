"""Whole-recording splits and statistics fitted only on the active training set."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from wammimic.data_pipeline.common import sha256,write_json
from wammimic.data_pipeline.datasets import HumanWAMDataset
from wammimic.data_pipeline.human_ego import DATASET,PROVENANCE,SCHEMA


def build_protocol(base_root,operation_root,operation_manifest,output,raw_roots):
    roots=[Path(base_root).resolve(),Path(operation_root).resolve()];output=Path(output)
    assignments={r['release_path']:r['evaluation_split'] for r in json.loads(Path(operation_manifest).read_text())['sessions']}
    rows=[];sources=[];groups={};video_hashes={};bvh_hashes={};history=None
    for source_id,root in enumerate(roots):
        dataset=root/DATASET;source=json.loads((dataset/'source.json').read_text())
        if source['schema']!=SCHEMA or source['label_provenance']!=PROVENANCE:raise ValueError('Non-optical source')
        if source_id==0:history=source.get('observation_state')
        elif history!=source.get('observation_state'):raise ValueError('Mixed human history contracts')
        ready=json.loads((root/'qa_reports/human_wam_training_ready.json').read_text())
        if not ready['ready']:raise ValueError('Source preparation not validated')
        cache_binding={'text_bank_sha256':sha256(root/'prompt_bank_bf16.safetensors'),
            'video_cache_report_sha256':sha256(root/'qa_reports/cache_video.json')}
        if cache_binding['text_bank_sha256']!=ready['prompt_bank_sha256'] or cache_binding['video_cache_report_sha256']!=ready['video_cache_report_sha256']:
            raise ValueError('Validated cache changed')
        sources.append({'root':str(root),'source_sha256':sha256(dataset/'source.json'),
            'episodes_sha256':sha256(dataset/'meta/episodes.jsonl'),**cache_binding})
        for row in map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines()):
            for path,digest in row['outputs'].items():
                if sha256(root/path)!=digest:raise ValueError('Prepared source changed')
            split=row['split'] if source_id==0 else assignments[row['source_session']]
            raw=Path(raw_roots[source_id])/row['source_session']
            capture=json.loads((raw/'raw/manifest.json').read_text())
            group=str(capture['upload_batch_id'])+'/'+capture['mocap_capture']['source_directory']
            video=next(v for k,v in row['outputs'].items() if k.endswith('.mp4'))
            bvh=sha256(raw/'raw/mocap/Body0.bvh')
            # A recording and an identical video must never appear across splits.
            for key,mapping in [(group,groups),(video,video_hashes),(bvh,bvh_hashes)]:
                if key in mapping and mapping[key]!=split:raise ValueError('Recording leakage across splits')
                mapping[key]=split
            rows.append({k:row[k] for k in ['episode_index','source_session','participant','family','primitive','length','valid_samples','human_leg_length']}
                        |{'source_id':source_id,'split':split,'original_split':row['split'],'recording_group':group})
    if any(not any(r['split']==s for r in rows) for s in ['train','val','test']):raise ValueError('Missing protocol split')
    operation_families={r['family'] for r in rows if r['source_id']==1}
    for family in operation_families:
        if {r['split'] for r in rows if r['family']==family}!={'train','val','test'}:
            raise ValueError('Operation family lacks train/val/test coverage: '+family)
    # Refit after holdout selection, never reuse source statistics containing held-out actions.
    values=[];states=[]
    for row in rows:
        if row['split']!='train':continue
        root=roots[row['source_id']]/DATASET
        table=pq.read_table(root/f'data/chunk-000/episode_{row["episode_index"]:06d}.parquet',columns=['action','valid_for_training']+(['observation.state'] if history else []))
        values.append(np.array(table['action'].to_pylist(),dtype=np.float32)[table['valid_for_training'].to_numpy()].reshape(-1,45))
        if history:states.append(np.array(table['observation.state'].to_pylist(),dtype=np.float32)[table['valid_for_training'].to_numpy()].reshape(-1,75))
    values=np.concatenate(values)
    if values.shape[1]!=45 or not np.isfinite(values).all():raise ValueError('Invalid active training targets')
    statistics={'action':{'mean':values.mean(0).tolist(),'std':values.std(0).tolist(),
        'q01':np.quantile(values,.01,axis=0).tolist(),'q99':np.quantile(values,.99,axis=0).tolist(),'count':len(values)}}
    if history:
        states=np.concatenate(states)
        if not np.isfinite(states).all():raise ValueError('Invalid active training history')
        statistics['observation.state']={'mean':states.mean(0).tolist(),'std':states.std(0).tolist(),'count':len(states)}
    people={s:sorted({r['participant'] for r in rows if r['split']==s}) for s in ['train','val','test']}
    report={'schema':'human-wam-recording-holdout-v1','sources':sources,'episodes':rows,'statistics':statistics,
        'operation_families':sorted(operation_families),'participants':people,
        'evaluation_scope':{'operations':'held-out whole recordings; participants may overlap training',
            'walking':'original official participant-disjoint validation/test retained'},
        'counts':{s:{'episodes':sum(r['split']==s for r in rows),'windows':sum(r['valid_samples'] for r in rows if r['split']==s)} for s in people}}
    write_json(output/'protocol.json',report);return report


class PilotDataset(Dataset):
    def __init__(self,protocol,split,per_episode=16,*,min_frame_gap=16):
        self.protocol=json.loads(Path(protocol).read_text());self.split=split;self.readers={};self.samples=[]
        self.rows={(r['source_id'],r['episode_index']):r for r in self.protocol['episodes']}
        for source_id,source in enumerate(self.protocol['sources']):
            root=Path(source['root']);dataset=root/DATASET
            if sha256(dataset/'source.json')!=source['source_sha256'] or sha256(dataset/'meta/episodes.jsonl')!=source['episodes_sha256']:
                raise ValueError('Protocol source changed')
            selected=[r['episode_index'] for r in self.protocol['episodes'] if r['source_id']==source_id and r['split']==split]
            if not selected:continue
            reader=HumanWAMDataset(root,episode_ids=selected,statistics=self.protocol['statistics']);self.readers[source_id]=reader
            for episode in selected:
                available=[(i,frame) for i,(ep,_,frame) in enumerate(reader.samples) if ep==episode]
                # The default gap keeps evaluation windows from sharing source frames.
                spaced=[];last=-min_frame_gap
                for index,frame in available:
                    if frame-last>=min_frame_gap:spaced.append(index);last=frame
                if per_episode and len(spaced)>per_episode:
                    spaced=[spaced[i] for i in np.linspace(0,len(spaced)-1,per_episode,dtype=int)]
                self.samples.extend((source_id,index) for index in spaced)
        if not self.samples:raise ValueError('No valid pilot samples for '+split)

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        source_id,local=self.samples[index];reader=self.readers[source_id]
        episode,_,frame=reader.samples[local]
        return reader[local]|{'source_id':source_id,'episode_index':episode,'frame_index':frame}

    def descriptor(self,index):
        source_id,local=self.samples[index];episode,_,frame=self.readers[source_id].samples[local]
        return self.rows[source_id,episode]|{'frame_index':frame,'root':self.protocol['sources'][source_id]['root']}
