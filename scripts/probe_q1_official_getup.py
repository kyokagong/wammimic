"""Capture the official initial pose and record the existing policy's actual outcome."""

import argparse
import asyncio
import json
from pathlib import Path
import time

from aiohttp import ClientSession, ClientTimeout
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription


async def probe(base, interface, output):
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    incoming = asyncio.get_running_loop().create_future()

    @pc.on('track')
    def receive(track):
        incoming.set_result(track)

    try:
        async with ClientSession(timeout=ClientTimeout(total=30), raise_for_status=True) as http:
            async def request(path, payload=None):
                async with http.request('GET' if payload is None else 'POST', base+path, json=payload) as response:
                    return await response.json()

            release = f'q1-official-{interface}-12000'
            await request('/api/control', dict(action='select', robot='q1', motion='official_q1_getup', release=release,
                          **({'human_points':'five'} if interface=='human' else {})))
            async with asyncio.timeout(900):
                while True:
                    status = await request('/api/status')
                    if status['error']:
                        raise RuntimeError(status['error'])
                    if status['release_id']==release and not status['loading'] and status['frame_available']:
                        break
                    await asyncio.sleep(1)
            pc.addTransceiver('video', direction='recvonly')
            await pc.setLocalDescription(await pc.createOffer())
            offer=pc.localDescription
            answer=await request('/api/offer', dict(type=offer.type, sdp=offer.sdp))
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            track=await asyncio.wait_for(incoming, 10)
            frame=await asyncio.wait_for(track.recv(), 30)
            frame.to_image().save(output.with_suffix('.png'))
            assert status['initial_frame']==0 and status['steps']==0
            await request('/api/control', dict(action='start'))
            deadline=time.monotonic()+45
            while time.monotonic()<deadline:
                frame=await asyncio.wait_for(track.recv(), 10)
                status=await request('/api/status')
                if status['error'] or status['failed'] or status['completed']:
                    break
            if not status['error']:
                await request('/api/control', dict(action='pause'))
            frame = await asyncio.wait_for(track.recv(), 10)
            frame.to_image().save(output.with_name(output.stem+'-final.png'))
            report={k: status.get(k) for k in ('release_id','motion','steps','simulated_seconds','failed','completed',
                     'termination_terms','error','input_mode','five_point_source','checkpoint','physics')}
            report['webrtc_connected']=pc.connectionState=='connected'
            report['probe_timeout']=not (status['error'] or status['failed'] or status['completed'])
            output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(report,ensure_ascii=False),flush=True)
            if not status['error']:
                await request('/api/control', dict(action='reset'))
    finally:
        await pc.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base')
    parser.add_argument('--interface',choices=('motion','human'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    asyncio.run(probe(args.base,args.interface,args.output))
