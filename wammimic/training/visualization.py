"""Actual checkpoint predictions, synchronized RGB, physical errors, MP4 and Rerun."""
from html import escape
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import rerun as rr

from wammimic.data_pipeline.common import write_json,sha256
from wammimic.data_pipeline.geometry import matrices
from wammimic.data_pipeline.human_ego import DATASET
from .evaluation import pose_errors,world_actions,temporal_errors

NAMES=['Pelvis','L ankle','R ankle','L wrist','R wrist']
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']


def rgb_frames(path,start):
    result=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vf',
        f'select=between(n\\,{start+1}\\,{start+10})','-fps_mode','passthrough','-frames:v','10',
        '-threads','2','-f','rawvideo','-pix_fmt','rgb24','pipe:1'])
    if len(result)!=10*224*224*3:raise ValueError('Visualization video/pose pairing mismatch')
    return np.frombuffer(result,dtype=np.uint8).reshape(10,224,224,3)


def case_video(protocol,record,prediction,reference,destination):
    root=Path(protocol['sources'][record['source_id']]['root'])/DATASET;episode=record['episode_index'];frame=record['frame_index'];leg=record['human_leg_length']
    with np.load(root/f'labels/episode_{episode:06d}.npz',allow_pickle=False) as labels:
        anchor=labels['poses_world'][frame];actual=labels['poses_world'][frame+1:frame+11].copy();timestamp=labels['video_timestamp'][frame+1:frame+11]-labels['video_timestamp'][frame]
    gt,gt_r,gt_valid=world_actions(reference,anchor,leg);pred,pred_r,pred_valid=world_actions(prediction,anchor,leg)
    if not np.allclose(gt,actual[...,:3],atol=1e-5):raise ValueError('Visualization reference is not the measured world pose')
    if not np.allclose(gt_r,matrices(actual[...,3:]),atol=1e-5):raise ValueError('Visualization optical rotations differ')
    origin=anchor[0,:3].copy();gt-=origin;pred-=origin
    errors=pose_errors(prediction,reference,leg);images=rgb_frames(root/f'videos/chunk-000/observation.images.cam_head/episode_{episode:06d}.mp4',frame)
    point_cloud=np.concatenate([gt.reshape(-1,3),pred.reshape(-1,3)]);low=point_cloud.min(0)-.15;high=point_cloud.max(0)+.15
    record_path=destination.with_suffix('.rrd');stream=rr.RecordingStream('WAM-Predict2.5-'+destination.stem);stream.save(str(record_path))
    stream.log('world',rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True)
    stream.log('provenance',rr.TextDocument('Cosmos-Predict2.5-2B trained checkpoint vs optical skeleton reference. RGB futures are reference replay only; policy observes data through t0. Positions use a fixed t0 pelvis origin; z=0 is not calibrated ground. '+record['recording']),static=True)
    figure=plt.figure(figsize=(12,8),dpi=110);writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-crf','20','-pix_fmt','yuv420p','-threads','2'])
    with writer.saving(figure,str(destination),dpi=110):
        for k in range(10):
            figure.clear();image_ax=figure.add_subplot(2,2,1);space=figure.add_subplot(2,2,2,projection='3d');top=figure.add_subplot(2,2,3);curve=figure.add_subplot(2,2,4)
            image_ax.imshow(images[k]);image_ax.axis('off');image_ax.set_title(f'Reference RGB at t0 + {timestamp[k]:.2f}s\nPolicy observation ends at t0',fontsize=10)
            for j,name in enumerate(NAMES):
                for values,style in [(gt,'-'),(pred,'--')]:
                    space.plot(*values[:k+1,j].T,style,color=COLORS[j],linewidth=1.5)
                    top.plot(values[:k+1,j,0],values[:k+1,j,1],style,color=COLORS[j],linewidth=1.5)
                space.scatter(*gt[k,j],color=COLORS[j],marker='o',s=30);space.scatter(*pred[k,j],color=COLORS[j],marker='x',s=40)
                top.scatter(*gt[k,j,:2],color=COLORS[j],marker='o',s=18);top.scatter(*pred[k,j,:2],color=COLORS[j],marker='x',s=24)
                curve.plot(timestamp,errors['position_mm'][:,j],label=name,color=COLORS[j])
            for points,rotations,valid,alpha in [(gt,gt_r,gt_valid,1.),(pred,pred_r,pred_valid,.55)]:
                for j in [3,4]:
                    if not valid[k,j]:continue
                    for a,c in enumerate(['r','g','b']):
                        vector=rotations[k,j,:,a]*.1;end=points[k,j]+vector
                        space.plot(*np.stack([points[k,j],end]).T,color=c,alpha=alpha,linewidth=1.)
            space.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
            space.set_box_aspect(high-low);space.view_init(elev=20,azim=-55);space.set_title('Five points: solid/o optical; dashed/x WAM\nWrist axes use source segment definitions',fontsize=10)
            top.set(xlabel='X (m)',ylabel='Y (m)',title='Top view in fixed t0 origin (not a ground plane)');top.set_aspect('equal',adjustable='box');top.grid(alpha=.2)
            curve.axvline(timestamp[k],color='.35',linewidth=1);curve.set(xlabel='Prediction horizon (s)',ylabel='Position error (mm)',title='Prediction vs optical reference');curve.grid(alpha=.2);curve.legend(fontsize=8,ncol=3)
            figure.suptitle(f'Cosmos-Predict2.5-2B WAM | {record["scope"]} | episode {episode}, frame {frame} | 0.5x replay',fontsize=11);figure.tight_layout(rect=(0,0,1,.95),h_pad=3.5,w_pad=2);writer.grab_frame()
            if k==9:figure.savefig(destination.with_suffix('.png'))
            stream.set_time('prediction_horizon',duration=float(timestamp[k]));stream.log('reference_rgb',rr.Image(images[k]))
            stream.log('world/optical',rr.Points3D(gt[k],radii=.02,labels=NAMES,colors=[[0,150,220]]*5))
            stream.log('world/wam',rr.Points3D(pred[k],radii=.015,labels=NAMES,colors=[[230,100,20]]*5))
            for j,name in enumerate(NAMES):
                entity=name.replace(' ','_')
                stream.log('position_error_mm/'+entity,rr.Scalars(float(errors['position_mm'][k,j])))
                stream.log('rotation_error_deg/'+entity,rr.Scalars(float(errors['rotation_deg'][k,j])))
                for label,points,rotations,valid in [('optical',gt,gt_r,gt_valid),('wam',pred,pred_r,pred_valid)]:
                    if valid[k,j]:stream.log('world/'+label+'_axes/'+entity,rr.Arrows3D(origins=np.repeat(points[k,j][None],3,axis=0),vectors=rotations[k,j].T*.1,colors=[[255,0,0],[0,180,0],[0,90,255]]))
    stream.flush();plt.close(figure)
    return {'video':destination.name,'image':destination.with_suffix('.png').name,'rerun':record_path.name,'record':record,
        'ade_mm':float(errors['position_mm'].mean()),'rotation_deg':float(errors['rotation_deg'].mean())}


def learning_curves(run,destination):
    events=[json.loads(s) for s in (run/'training.jsonl').read_text().splitlines()];epochs=[json.loads((run/'initial_validation.json').read_text())]+[json.loads(p.read_text()) for p in sorted(run.glob('validation_epoch_*.json'))]
    figure,axes=plt.subplots(1,2,figsize=(12,4))
    for key,label in [('flow_loss','Action flow matching'),('video_flow_loss','Video flow matching'),('baseline_loss','RGB/text baseline MSE')]:axes[0].plot([r['microstep'] for r in events],[r[key] for r in events],label=label,alpha=.7)
    axes[0].set(xlabel='Microstep',ylabel='Training objective',title='Losses have different targets/scales');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for key,label in [('wam','Cosmos WAM'),('rgb_text_baseline','Same-input baseline'),('training_mean','Training mean')]:axes[1].plot(range(len(epochs)),[r['metrics'][key]['overall']['recording_macro_ade_mm'] for r in epochs],marker='o',label=label)
    axes[1].set(xlabel='Epoch (0 = initial)',ylabel='Recording-macro ADE (mm)',title='Validation only; test never selects checkpoints');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    figure.tight_layout();figure.savefig(destination,dpi=150);plt.close(figure)


def visualize(protocol_path,run_root):
    run=Path(run_root);protocol=json.loads(Path(protocol_path).read_text());destination=run/'visualizations';destination.mkdir(exist_ok=True)
    cases=[];metrics=[];temporal=[]
    for split in ['val','test']:
        report=json.loads((run/f'evaluation_{split}_seed0.json').read_text());predictions=np.load(run/f'evaluation_{split}_seed0.npz',allow_pickle=False)
        for index,record in enumerate(report['records']):
            labels=Path(protocol['sources'][record['source_id']]['root'])/DATASET/f'labels/episode_{record["episode_index"]:06d}.npz'
            with np.load(labels,allow_pickle=False) as original:times=original['video_timestamp'][record['frame_index']+1:record['frame_index']+11]
            for model,key in [('wam','prediction'),('rgb_text_baseline','rgb_text_baseline')]:
                temporal.append({'split':split,'seed':0,'scope':record['scope'],'model':model,
                    'source_id':record['source_id'],'episode':record['episode_index'],'frame':record['frame_index'],
                    **temporal_errors(predictions[key][index],predictions['reference'][index],record['human_leg_length'],times)})
        for scope in sorted({r['scope'] for r in report['records']}):
            candidates=[i for i,r in enumerate(report['records']) if r['scope']==scope]
            selections=[('fixed',candidates[0]),('worst',max(candidates,key=lambda i:pose_errors(predictions['prediction'][i],predictions['reference'][i],report['records'][i]['human_leg_length'])['position_mm'].mean()))]
            for label,index in selections:
                path=destination/f'{split}_{scope}_{label}.mp4';cases.append(case_video(protocol,report['records'][index],predictions['prediction'][index],predictions['reference'][index],path)|{'split':split,'selection':label})
        for model,values in report['metrics'].items():
            for scope,values_for_scope in values['groups']['scope'].items():metrics.append({'split':split,'model':model,'scope':scope,**values_for_scope})
    learning_curves(run,destination/'learning_curves.png');write_json(destination/'cases.json',cases)
    write_json(destination/'temporal_diagnostics_seed0.json',{'scope':'Fixed seed 0; derivative errors relative to optical reference, using actual synchronized timestamps. This is not a motor-jerk or robot-stability certification.','records':temporal})
    write_json(destination/'provenance.json',{'checkpoint_sha256':{name:sha256(run/f'best/{name}.safetensors') for name in ['wam','baseline']},
        'protocol_sha256':sha256(protocol_path),'prediction_npz_sha256':{split:sha256(run/f'evaluation_{split}_seed0.npz') for split in ['val','test']},
        'visualization_code_sha256':sha256(Path(__file__))})
    table=''.join(f'<tr><td>{escape(r["split"])}</td><td>{escape(r["scope"])}</td><td>{escape(r["model"])}</td><td>{r["ade_mm"]:.1f}</td><td>{r["fde_mm"]:.1f}</td><td>{r["rotation_mean_deg"]:.1f}</td></tr>' for r in metrics)
    videos=''.join(f'<section><h2>{escape(c["split"]+" / "+c["record"]["scope"]+" / "+c["selection"])}</h2><video controls preload="metadata" src="{escape(c["video"])}"></video><p>ADE {c["ade_mm"]:.1f} mm · {escape(c["record"]["recording"])}</p><a href="{escape(c["rerun"])}">下载 Rerun 交互回放</a></section>' for c in cases)
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WAM 小规模训练评估</title><style>body{font:16px system-ui;max-width:1200px;margin:24px auto;padding:0 16px;color:#202124}video,img{width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}section{margin-top:32px}.scroll{overflow:auto}</style><h1>Cosmos-Predict2.5-2B WAM 小规模训练评估</h1><p>所有 WAM 曲线均来自真实训练 checkpoint。光学骨架是测量解算参考；未来 RGB 仅用于评估回放。操作评估隔离完整会话，参与者可能与训练重叠；行走保留官方跨人留出集。常速度参考使用过去动捕，属于特权参考。表格为固定 seed=0，其他种子结果见 evaluation_summary.json；本轮未评测机器人闭环成功率。</p><img src="learning_curves.png" alt="训练损失和验证集位置误差"><div class="scroll"><table><thead><tr><th>划分</th><th>范围</th><th>模型/参考</th><th>ADE mm</th><th>FDE mm</th><th>旋转 °</th></tr></thead><tbody>'+table+'</tbody></table></div>'+videos+'</html>'
    html=html.replace('</html>','<p>附：<a href="temporal_diagnostics_seed0.json">固定种子时间一致性诊断</a> · <a href="provenance.json">checkpoint 与预测来源校验</a></p></html>')
    (destination/'report.html').write_text(html)
    print('Visualization report:',destination/'report.html',flush=True)
