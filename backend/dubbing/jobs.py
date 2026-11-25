"""Durable queue. CPU source preparation and the single GPU worker are separate."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import secrets
import threading
import time
from .engine import ENGINE_REVISION
from .gpu import (gpu_memory_mib, gpu_capacity_mib, is_cuda_oom, clear_cuda_memory,
                  MIN_GPU_MIB, GPU_RETRY_SECONDS, MAX_OOM_RETRIES)
from .render import (MAX_TEMPO, TimingFitError, fit_audio, fit_audio_array,
                     fit_micro_window, fit_micro_window_array,
                     trim_dead_silence, trim_dead_silence_array, wav_duration)
from .source import SOURCE_REVISION, SourceProvider, is_non_speech
from .leases import ClientLeases
from .persistence import JobStore, QuarantinedJobError, SCHEMA_VERSION

SAFE_ID=re.compile(r'^[A-Za-z0-9_-]{1,80}$')
QUALITY_REVISION='spoken-v3-sentence-flow'
QUALITY_MAX_TEMPO=1.25
QUALITY_FALLBACK_MAX_TEMPO=1.75
SHORT_WINDOW_FALLBACK_S=1.20
MICRO_WINDOW_FALLBACK_S=0.85
MAX_REWRITES=2
MAX_CONTINUATION_ADVANCE=3.0
CONTINUATION_PAUSE=.12

def speech_start(segments, segment, raw_seconds, max_tempo=QUALITY_MAX_TEMPO, max_borrow=0.25):
    """Schedule forced phrase continuations in verified silence, at most 3s early.

    Legacy short-window fitting retains its 250ms allowance (expandable to 500ms on
    post-rewrite fallback). Source intervals remain immutable; new flow scheduling
    requires an explicit continuation tag.
    """
    start = segment['start']
    index = segments.index(segment)
    if not index:
        return start
    previous = segments[index - 1]
    if previous['state'] != 'ready' or not number(previous.get('audio_duration')):
        return start
    finished = previous.get('audio_start', previous['start']) + previous['audio_duration']
    if (segment.get('continues_previous') and segment.get('speaker') is not None and
            previous.get('speaker') == segment['speaker'] and
            0 <= start-previous['end'] <= 1.0):
        # A forced split inside one utterance is not a subtitle-clock pause.
        # Shift only into verified silence, with a per-phrase synchronization cap.
        candidate = max(0, start-MAX_CONTINUATION_ADVANCE, finished+CONTINUATION_PAUSE)
        if candidate < start:
            return candidate
    if raw_seconds <= (segment['end'] - start) * max_tempo:
        return start
    candidate = max(0, start - max_borrow, finished + .04)
    if candidate >= start or raw_seconds > (segment['end'] - candidate) * max_tempo:
        return start
    return candidate

class PreparationCancelled(Exception):
    """Stop a preparation stage without poisoning its resumable job."""

def number(v):return type(v) in (int,float) and math.isfinite(v)

def _is_timing_error(error):
    return isinstance(error, str) and error.startswith('TimingFitError:')

def _timing_feedback(segment, window, raw_seconds, attempt):
    history, seen = [], set()
    for entry in segment.get('generation_attempts', [])[-8:]:
        text = str(entry.get('text_ru', '')).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        raw = entry.get('raw_seconds')
        history.append(dict(
            text_ru=text[:200],
            chars=len(text),
            raw_seconds=round(raw, 2) if number(raw) else None,
        ))
    history = history[-6:]
    last_len = len(segment.get('text_ru') or '') or (history[-1]['chars'] if history else 0)
    scaled = int(last_len * (window / raw_seconds) * 0.90) if raw_seconds > 0 else round(window * 10)
    if last_len > 4:
        scaled = min(scaled, last_len - 1)
    target = max(4, scaled)
    return dict(
        raw_audio_seconds=round(raw_seconds, 2),
        window_s=round(window, 3),
        max_raw_seconds=round(window * QUALITY_MAX_TEMPO, 2),
        overshoot_ratio=round(raw_seconds / window, 2) if window > 0 else None,
        target_chars=target,
        attempt=attempt,
        previous_attempts=history,
    )
def validate_client(value):
    if value is not None and (not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}',value)):
        raise ValueError('invalid client_id')
_DIGEST_CACHE = OrderedDict()
_DIGEST_LOCK = threading.Lock()
_MAX_DIGEST_CACHE = 4096


def digest(path):
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    with _DIGEST_LOCK:
        cached = _DIGEST_CACHE.get(key)
        if cached is not None:
            _DIGEST_CACHE.move_to_end(key)
            return cached
    value = hashlib.sha256(p.read_bytes()).hexdigest()
    with _DIGEST_LOCK:
        _DIGEST_CACHE[key] = value
        _DIGEST_CACHE.move_to_end(key)
        while len(_DIGEST_CACHE) > _MAX_DIGEST_CACHE:
            _DIGEST_CACHE.popitem(last=False)
    return value


def atomic_json(path, value):
    temp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        with temp.open('w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def validate(payload):
    if not isinstance(payload,dict):raise ValueError('object required')
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}',str(payload.get('video_id',''))):raise ValueError('invalid YouTube video_id')
    duration=payload.get('duration')
    if not number(duration) or not 0<duration<=86400:raise ValueError('invalid duration')
    for lang in ('source_language','target_language'):
        if not isinstance(payload.get(lang),str) or not re.fullmatch(r'[A-Za-z0-9-]{2,32}',payload[lang]):raise ValueError('invalid language')
    if payload.get('start_from',0)!=0:raise ValueError('start_from must be zero')
    cues=payload.get('cues')
    if not isinstance(cues,list) or not 0<len(cues)<=30000:raise ValueError('invalid cues')
    result=[];seen=set();last=-1
    for i,c in enumerate(cues):
        if not isinstance(c,dict):raise ValueError('invalid cue')
        start,end=c.get('start'),c.get('end');ident=c.get('id',str(i))
        if not isinstance(ident,str) or not SAFE_ID.fullmatch(ident) or ident in seen:raise ValueError('invalid cue id')
        if not number(start) or not number(end) or not 0<=start<end<=duration or start<last:raise ValueError('invalid cue timing')
        text=c.get('text')
        if not isinstance(text,str) or not text.strip() or len(text)>10000:raise ValueError('invalid cue text')
        result.append(dict(id=ident,start=start,end=end,text=text.strip()));seen.add(ident);last=start
    return dict(video_id=payload['video_id'],duration=duration,source_language=payload['source_language'],target_language=payload['target_language'],cues=result)

class JobManager:
    def __init__(self,data_dir,source_provider=None,engine_factory=None,autostart=True,idle_timeout=60,editor=None,quality=False,memory_probe=None,capacity_probe=None):
        self.root=Path(data_dir).resolve();self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.instance_lock=(self.root/'service.lock').open('a')
        try:fcntl.flock(self.instance_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.instance_lock.close();raise RuntimeError('another service owns this data directory')
        token_path=self.root/'token'
        if not token_path.exists():
            fd=os.open(token_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32))
        self.token=token_path.read_text().strip()
        self.provider=source_provider;self.engine_factory=engine_factory;self.engine=None
        self.memory_probe=memory_probe if memory_probe is not None else (gpu_memory_mib if engine_factory is None else None)
        self.capacity_probe=capacity_probe if capacity_probe is not None else (gpu_capacity_mib if engine_factory is None else None)
        self.editor=editor;self.quality=quality
        self.idle_timeout=idle_timeout
        self.closest_partners=str(os.environ.get('VOXDUB_CLOSEST_PARTNERS','0')).strip().lower() in ('1','true','yes','on')
        self.lock=threading.RLock();self.stop=threading.Event();self.jobs={};self.threads=[];self.worker_errors={};self.workers_started=autostart
        self.store=JobStore(self.root, validate)
        for path, job in self.store.load():
            try:
                changed = self._restore_job(job, path.parent)
                if changed or job.get('schema_version') != SCHEMA_VERSION:
                    self._save(job)
                self.jobs[job['id']]=job
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.store.quarantine(path, error)
        if autostart:
            for target in (self._source_loop,self._edit_loop,self._engine_loop,self._lease_loop):
                t=threading.Thread(target=self._run_worker,args=(target,),name=target.__name__,daemon=True);t.start();self.threads.append(t)
    def _restore_job(self, job, directory):
        """Recover interrupted work only after metadata/paths passed validation."""
        changed = False
        for segment in job['segments']:
            if segment['state']=='generating':segment['state']='pending';changed=True
            if segment['state']=='editing':segment['state']='needs_edit';changed=True
            audio=directory/(segment['id']+'.wav')
            reference=Path(segment['reference'])
            if not reference.is_file() or digest(reference)!=segment['reference_sha256']:
                segment.update(state='error',error='reference changed or missing; cached speech invalidated');changed=True
            elif segment['state']=='ready' and (not audio.is_file() or digest(audio)!=segment.get('audio_sha256')):
                segment['state']='pending';changed=True
        for index, segment in enumerate(job['segments']):
            if segment['state']=='ready' and segment.get('audio_start',segment['start']) < segment['start']:
                previous=job['segments'][index-1] if index else None
                if (previous is None or previous['state']!='ready' or
                    previous.get('audio_start',previous['start'])+previous['audio_duration']+.04 > segment['audio_start']):
                    segment['state']='pending';changed=True
        had_clients = 'clients' in job
        ClientLeases.initialize(job, time.time())
        if not had_clients and 'clients' in job:
            changed = True
        if ClientLeases.expire(job, time.time()):
            changed = True
        return changed

    def _run_worker(self, target):
        """A dead worker is an unhealthy service, never an invisible empty queue."""
        try:
            target()
            if not self.stop.is_set():raise RuntimeError('worker exited unexpectedly')
        except Exception as error:
            logging.exception('Dubbing worker %s failed', target.__name__)
            with self.lock:self.worker_errors[target.__name__] = type(error).__name__

    def health(self):
        with self.lock:
            failed = dict(self.worker_errors)
            if self.workers_started and not self.stop.is_set():
                for thread in self.threads:
                    if not thread.is_alive():failed.setdefault(thread.name, 'worker stopped')
            status='unhealthy' if failed or self.stop.is_set() else 'degraded' if self.store.quarantined else 'ok'
            from .gpu import detect_gpu_profile
            from .editing import get_public_config
            cfg = get_public_config(self.root)
            return dict(status=status,version=1,failed_workers=sorted(failed),
                        quarantined_jobs=len(self.store.quarantined),
                        editor_mode=cfg['mode'],
                        gpu_profile=detect_gpu_profile())

    def get_config(self):
        from .gpu import detect_gpu_profile
        from .editing import get_public_config
        cfg = get_public_config(self.root)
        cfg['gpu_profile'] = detect_gpu_profile()
        return cfg

    def update_config(self, patch):
        from .gpu import detect_gpu_profile
        from .editing import save_editor_config
        cfg = save_editor_config(self.root, patch)
        with self.lock:
            if self.editor is not None and hasattr(self.editor, 'close'):
                try:
                    self.editor.close()
                except Exception:
                    pass
            self.editor = None
        cfg['gpu_profile'] = detect_gpu_profile()
        return cfg

    def test_config(self, override=None):
        from .editing import test_editor_connection
        return test_editor_connection(self.root, override=override)

    def close(self):
        self.stop.set()
        for t in self.threads:t.join(timeout=65)
        if self.editor is not None and hasattr(self.editor, 'close'):
            try:
                self.editor.close()
            except Exception:
                pass
        if not any(t.is_alive() for t in self.threads):self.instance_lock.close()
    def _save(self,j):
        j['schema_version']=SCHEMA_VERSION
        atomic_json(self.root/j['id']/'job.json',j)
    def create(self,payload):
        client_id=payload.get('client_id') if isinstance(payload,dict) else None
        validate_client(client_id)
        payload=validate(payload)
        revision=Path(__file__).resolve().parent.parent/'model-revision.txt'
        cache=dict(payload=payload,engine=ENGINE_REVISION,model=revision.read_text().strip() if revision.exists() else 'missing',translation_version=1,source_revision=SOURCE_REVISION)
        if self.quality:cache['quality_profile']=QUALITY_REVISION
        jid=hashlib.sha256(json.dumps(cache,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:32]
        with self.lock:
            if jid in self.store.quarantined:raise QuarantinedJobError('Stored job requires review; original files preserved')
            if jid not in self.jobs:
                (self.root/jid).mkdir(exist_ok=True)
                self.jobs[jid]=dict(id=jid,payload=payload,segments=[],prepared=False,cancelled=False,error=None,priority=0,created=time.time(),quality_profile=QUALITY_REVISION if self.quality else None)
            j=self.jobs[jid]
            ClientLeases.acquire(j, client_id, time.time())
            j.pop('gpu_wait',None)
            # A repeated create is the user's explicit retry, never a background loop.
            if j['error']:
                j['last_error']=j['error'];j['error']=None
                if not j['prepared']:j['source_progress']={}
            for segment in j['segments']:
                if segment['state']=='error':
                    segment['last_error']=segment.pop('error',None)
                    if segment.get('text_ru') and (segment.get('gpu_oom_attempts') or segment.get('gpu_capacity_mib') or 'CUDA out of memory' in str(segment.get('last_error'))):
                        segment['state']='pending'
                        segment.pop('gpu_oom_attempts',None)
                        segment.pop('gpu_required_mib',None)
                        segment.pop('gpu_capacity_mib',None)
                    elif j.get('quality_profile') and segment.get('draft_ru'):
                        last_err=str(segment.get('last_error',''))
                        is_timing_err='TimingFitError' in last_err or 'exceeds window' in last_err or 'exceeds source window' in last_err
                        prior=segment.get('generation_attempts') or []
                        if is_timing_err and prior:
                            last_raw=prior[-1].get('raw_seconds')
                            window=segment['end']-segment.get('audio_start',segment['start'])
                            fb=_timing_feedback(segment,window,last_raw if number(last_raw) else window*1.5,1)
                            segment.update(state='needs_edit',text_ru='',rewrite_attempts=0,rewrite_feedback=fb)
                        else:
                            segment.update(state='needs_edit',text_ru='',rewrite_attempts=0)
                            segment.pop('rewrite_feedback',None)
                    else:segment['state']='pending' if segment['text_ru'] else 'needs_translation'
            self._save(j);return self.summary(jid)
    def _get(self,jid):
        if jid in self.store.quarantined:raise QuarantinedJobError('Stored job requires review; original files preserved')
        if jid not in self.jobs:raise KeyError('unknown job')
        return self.jobs[jid]
    def summary(self,jid):
        with self.lock:
            j=self._get(jid)
            if ClientLeases.expire(j, time.time()):self._save(j)
            duration=j['payload']['duration'];segments=j['segments'];ranges=[]
            timing_errors = {s['id'] for s in segments if s['state'] == 'error' and _is_timing_error(s.get('error'))}
            # A phrase that exhausted timing retries is playable as a gap when
            # other phrases can still provide dubbing. Keep the persisted error for retries.
            skipped = timing_errors if len(timing_errors) < len(segments) else set()
            fatal_segment = next((s for s in segments if s['state'] == 'error' and s['id'] not in skipped), None)
            fatal_error = j['error'] or (fatal_segment or {}).get('error')
            if j['prepared']:
                cursor=0
                for s in segments:
                    if s['start']>cursor:ranges.append([cursor,s['start']])
                    if s['state']=='ready' or s['id'] in skipped:ranges.append([s['start'],s['end']])
                    cursor=s['end']
                if cursor<duration:ranges.append([cursor,duration])
            merged=[]
            for start,end in ranges:
                if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
                else:merged.append([start,end])
            ready_until=merged[0][1] if merged and merged[0][0]==0 else 0
            pos=j.get('priority',0)
            ready_from=pos if 0<pos<duration else 0
            pos_end=next((end for start,end in merged if start<=ready_from<=end),ready_from)
            buffered_s=max(0.0,pos_end-ready_from)
            required_buffer_s=min(300,max(0.0,duration-ready_from))
            ready=sum(s['state']=='ready' for s in segments)
            if j['cancelled']:status='cancelled'
            elif j['error'] or fatal_segment is not None:status='failed'
            elif not j['prepared']:status='preparing'
            elif j.get('gpu_wait'):status='waiting_gpu'
            elif ready+len(skipped)==len(segments):status='complete'
            elif any(s['state'] in ('pending','generating') for s in segments):status='generating'
            elif any(s['state'] in ('needs_edit','editing') for s in segments):status='editing'
            else:status='waiting_translation'
            public=[]
            raw_meta=j.get('source_metadata') or {}
            metadata=dict(raw_meta)
            if isinstance(raw_meta.get('references'), dict):
                metadata['references']={
                    k: {rk: rv for rk, rv in v.items() if rk != 'path'} if isinstance(v, dict) else v
                    for k, v in raw_meta['references'].items()
                }
            for s in segments:
                row={k:s[k] for k in ('id','start','end','text','text_ru','speaker','state')}
                if s['id'] in skipped:
                    row['state']='skipped'
                if s.get('error'):row['error']=s['error']
                if s.get('metrics'):row['metrics']=dict(s['metrics'])
                em=s.get('edit_metrics')
                if em:
                    em_copy=dict(em)
                    if isinstance(em_copy.get('usage'), dict):
                        em_copy['usage']=dict(em_copy['usage'])
                    row['edit_metrics']=em_copy
                if s['state']=='ready':
                    row.update(audio_duration=s['audio_duration'],audio_url=f"/v1/jobs/{jid}/segments/{s['id']}")
                    if s.get('audio_start',s['start']) < s['start']:
                        row['source_start']=s['start'];row['start']=s['audio_start']
                        if public:public[-1]['end']=min(public[-1]['end'],row['start'])
                public.append(row)
            gw=j.get('gpu_wait')
            sp=j.get('source_progress')
            return dict(id=jid,status=status,duration=duration,ready_until=ready_until,priority=pos,buffered_s=buffered_s,ranges=merged,
                quality_profile=j.get('quality_profile'),gpu_wait=dict(gw) if isinstance(gw, dict) else gw,
                active_clients=len(j.get('clients',[])),
                source_progress=dict(sp) if isinstance(sp, dict) else {},model_load_count=j.get('model_load_count',0),
                source_metadata=metadata,last_model_load_seconds=j.get('last_model_load_seconds'),
                ready_to_play=status not in ('cancelled','failed') and j['prepared'] and buffered_s>=required_buffer_s,
                required_buffer_s=required_buffer_s,
                total_segments=len(segments),ready_segments=ready,skipped_segments=len(skipped),error=fatal_error,segments=public)
    def translations(self,jid,items):
        if not isinstance(items,list) or not 1<=len(items)<=1000:raise ValueError('invalid translation batch')
        with self.lock:
            j=self._get(jid);by_id={s['id']:s for s in j['segments']};changes=[];seen=set()
            for item in items:
                if not isinstance(item,dict) or not isinstance(item.get('id'),str) or item['id'] not in by_id:raise ValueError('unknown segment')
                if item['id'] in seen:raise ValueError('duplicate translation id')
                seen.add(item['id'])
                s=by_id[item['id']];text=item.get('text_ru')
                if not isinstance(text,str) or not text.strip() or len(text)>10000:raise ValueError('invalid translation')
                text=text.strip()
                previous=s.get('draft_ru','') if j.get('quality_profile') else s['text_ru']
                if previous and previous!=text:raise ValueError('translation is immutable; create a changed source job')
                changes.append((s,text))
            for s,text in changes:
                if j.get('quality_profile'):
                    s['draft_ru']=text
                    if s['state']=='needs_translation':s['state']='needs_edit'
                else:
                    s['text_ru']=text
                    if s['state']=='needs_translation':s['state']='pending'
            self._save(j);return self.summary(jid)
    def priority(self,jid,position):
        with self.lock:
            j=self._get(jid)
            if not number(position) or not 0<=position<=j['payload']['duration']:raise ValueError('invalid position')
            j['priority']=position;self._save(j);return self.summary(jid)
    def cancel(self,jid,client_id=None):
        validate_client(client_id)
        with self.lock:
            j=self._get(jid)
            ClientLeases.release(j, client_id, time.time())
            self._save(j);return self.summary(jid)
    def heartbeat(self, jid, client_id):
        validate_client(client_id)
        if client_id is None:raise ValueError('client_id required')
        with self.lock:
            j = self._get(jid)
            now = time.time()
            expired = ClientLeases.expire(j, now)
            try:
                deadline = ClientLeases.renew(j, client_id, now)
            except Exception:
                if expired:
                    self._save(j)
                raise
            self._save(j)
            return dict(id=jid, lease_expires_at=deadline, active_clients=len(j['clients']))

    def _lease_loop(self):
        # Independent of long source/editor/GPU calls and closed browser tabs.
        while not self.stop.wait(1):
            with self.lock:
                for job in self.jobs.values():
                    if ClientLeases.expire(job, time.time()):self._save(job)

    def audio_path(self,jid,sid):
        with self.lock:
            j=self._get(jid)
            if not any(s['id']==sid and s['state']=='ready' for s in j['segments']):raise KeyError('audio not ready')
            return self.root/jid/(sid+'.wav')
    def next_segment(self):
        with self.lock:
            for j in sorted(self.jobs.values(),key=lambda j:j['created']):
                if j['cancelled'] or j['error']:continue
                if j.get('gpu_wait',{}).get('retry_after',0)>time.time():continue
                pending=[s for s in j['segments'] if s['state']=='pending']
                if pending:
                    pos=j['priority']
                    return j,min(pending,key=lambda s:(s['end']<=pos,0 if s['start']<=pos<s['end'] else 1,s['start']))
        return None
    def _ensure_editor(self):
        with self.lock:
            if self.editor is None:
                from .editing import create_editor
                self.editor = create_editor(self.root)
            return self.editor

    def _run_edit_batch(self, j, batch, items, context):
        try:
            editor = self._ensure_editor()
            brief = j.get('video_brief')
            if brief is None and hasattr(editor, 'brief'):
                with self.lock:
                    segs = list(j['segments']) or list(j['payload']['cues'])
                step = max(1, len(segs) // 14)
                samples = [seg['text'] for seg in segs[::step][:14] if seg.get('text')]
                try:
                    brief = editor.brief(j['payload'].get('video_id', ''), samples,
                        cancelled=lambda: self.stop.is_set() or j['cancelled'])
                except InterruptedError:
                    raise
                except Exception:
                    brief = dict(fallback=True, genre='', register='', glossary=[], use_vox_style_tag=False, vox_style_tag='')
                with self.lock:
                    if j.get('video_brief') is None:
                        j['video_brief'] = brief
                        self._save(j)
                    else:
                        brief = j['video_brief']
            edit_kwargs = dict(context=context, cancelled=lambda: self.stop.is_set() or j['cancelled'])
            if isinstance(brief, dict) and (brief.get('genre') or brief.get('register') or brief.get('glossary')):
                edit_kwargs['video_brief'] = brief
            result = editor.edit(items, **edit_kwargs)
            mapping = {s['id']: s['text_ru'] for s in result}
            if len(mapping) != len(batch) or set(mapping) != {s['id'] for s in batch} or any(not isinstance(t, str) or not t.strip() for t in mapping.values()):
                raise ValueError('incomplete editor batch')
            metrics_snapshot = copy.deepcopy(getattr(editor, 'last_metrics', {}))
            with self.lock:
                for s in batch:
                    if self.stop.is_set() or j['cancelled']:
                        s['state'] = 'needs_edit'
                        continue
                    s.update(text_ru=mapping[s['id']].strip(), state='pending', edit_metrics=copy.deepcopy(metrics_snapshot))
                    s['edit_passes'] = s.get('edit_passes', 0) + 1
                self._save(j)
        except Exception as exc:
            with self.lock:
                for s in batch:
                    if self.stop.is_set() or j['cancelled']:
                        s['state'] = 'needs_edit'
                    else:
                        s.update(state='error', error=f'Editing failed: {type(exc).__name__}: {exc}')
                if not self.stop.is_set() and not j['cancelled']:
                    j['error'] = f'Editing failed: {type(exc).__name__}: {exc}'
                    from .editing import ResponseEditingError, REVISION
                    if isinstance(exc, ResponseEditingError):
                        j['editing_failure'] = dict(revision=REVISION, items=items,
                            feedback=exc.feedback, output=exc.previous_output)
                self._save(j)

    def _edit_loop(self):
        max_workers = 4
        active = set()
        brief_inflight = set()
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='edit-worker') as pool:
            while not self.stop.is_set():
                done = {f for f in active if f.done()}
                for f in done:
                    active.discard(f)
                    f.result()

                # Eagerly compute video_brief in parallel with Stage 1 (source.py) using raw cues!
                eager_brief_job = None
                if len(active) < max_workers:
                    with self.lock:
                        for cand in self.jobs.values():
                            if (cand.get('quality_profile') and not cand['cancelled'] and not cand['error'] and
                                    cand.get('video_brief') is None and cand['id'] not in brief_inflight and
                                    (not cand['prepared'] or any(s['state'] == 'needs_edit' for s in cand['segments']))):
                                if self.editor is None or hasattr(self.editor, 'brief'):
                                    eager_brief_job = cand
                                    brief_inflight.add(cand['id'])
                                    break
                if eager_brief_job is not None:
                    def _do_brief(job):
                        try:
                            editor = self._ensure_editor()
                            if not hasattr(editor, 'brief'):
                                return
                            with self.lock:
                                segs = list(job['segments']) or list(job['payload']['cues'])
                            step = max(1, len(segs) // 14)
                            samples = [seg['text'] for seg in segs[::step][:14] if seg.get('text')]
                            try:
                                brief = editor.brief(job['payload'].get('video_id', ''), samples,
                                    cancelled=lambda: self.stop.is_set() or job['cancelled'])
                            except InterruptedError:
                                return
                            except Exception:
                                brief = dict(fallback=True, genre='', register='', glossary=[], use_vox_style_tag=False, vox_style_tag='')
                            with self.lock:
                                if job.get('video_brief') is None:
                                    job['video_brief'] = brief
                                    self._save(job)
                        except Exception:
                            logging.exception('Eager video_brief failed for job %s', job.get('id'))
                        finally:
                            with self.lock:
                                brief_inflight.discard(job['id'])
                    active.add(pool.submit(_do_brief, eager_brief_job))

                dispatched = False
                while len(active) < max_workers and not self.stop.is_set():
                    with self.lock:
                        j = next((j for j in self.jobs.values()
                                  if j.get('quality_profile') and not j['cancelled'] and not j['error'] and
                                  j['id'] not in brief_inflight and
                                  any(s['state'] == 'needs_edit' for s in j['segments'])), None)
                        if j is None:
                            break
                        pos = j.get('priority', 0)
                        candidates = [s for s in j['segments'] if s['state'] == 'needs_edit']
                        # Prioritize segments at/after playback position, preserving contiguous slices.
                        first_idx = next((idx for idx, s in enumerate(candidates) if s['end'] > pos), 0)
                        batch = candidates[first_idx:first_idx + 6]
                        if len(batch) < 6 and first_idx > 0:
                            batch = batch + candidates[:6 - len(batch)]
                        id_to_idx = {seg['id']: idx for idx, seg in enumerate(j['segments'])}
                        indices = [id_to_idx[s['id']] for s in batch]
                        neighbors = sorted({n for index in indices for n in range(max(0, index - 2), min(len(j['segments']), index + 3)) if n not in indices})
                        context = [dict(position=n, source=j['segments'][n]['text'], draft=j['segments'][n].get('text_ru') or j['segments'][n].get('draft_ru') or '') for n in neighbors]
                        items = []
                        for s, seg_idx in zip(batch, indices):
                            window = s['end'] - s['start']
                            feedback = s.get('rewrite_feedback')
                            item = dict(id=s['id'], position=seg_idx, source=s['text'], draft=(s['text_ru'] if feedback and s.get('text_ru') else s['draft_ru']), window_s=window, target_chars=max(4, round(window * 12.5)))
                            if feedback:
                                item.update(feedback=feedback, target_chars=feedback['target_chars'], window_s=feedback['window_s'])
                            items.append(item)
                            s['state'] = 'editing'
                    active.add(pool.submit(self._run_edit_batch, j, batch, items, context))
                    dispatched = True
                if not dispatched:
                    self.stop.wait(.05)
            for f in list(active):
                f.result()
    def _source_loop(self):
        while not self.stop.is_set():
            with self.lock:
                j=next((j for j in self.jobs.values() if not j['prepared'] and not j['cancelled'] and not j['error']),None)
            if j is None:self.stop.wait(.1);continue
            try:
                if self.provider is None:
                    self.provider=SourceProvider(allow_cuda=lambda: self.engine is None)
                last_progress_save = [0.0, None]
                def progress(value):
                    if self.stop.is_set() or j['cancelled']:raise PreparationCancelled()
                    filtered = {k: v for k, v in value.items() if k in (
                        'phase', 'seconds', 'analyzed_until', 'duration', 'total_segments',
                        'downloaded_bytes', 'total_bytes', 'total_estimated',
                    )}
                    now=time.monotonic()
                    phase=filtered.get('phase')
                    should_save=(phase!=last_progress_save[1]) or (now-last_progress_save[0]>=2.0)
                    with self.lock:
                        j['source_progress']=filtered
                        if should_save:
                            self._save(j)
                            last_progress_save[0]=now
                            last_progress_save[1]=phase
                result=self.provider.prepare(j['payload']['video_id'],copy.deepcopy(j['payload']['cues']),self.root/j['id'],progress=progress)
                segments=[];last=0;seen=set()
                for s in result['segments']:
                    if not isinstance(s.get('id'),str) or not SAFE_ID.fullmatch(s['id']) or s['id'] in seen:raise ValueError('invalid prepared segment id')
                    if not number(s.get('start')) or not number(s.get('end')) or not last<=s['start']<s['end']<=j['payload']['duration']:raise ValueError('source segments overlap or exceed duration')
                    ref=Path(s['reference']).resolve()
                    if not ref.is_file() or not ref.is_relative_to(self.root/j['id']):raise ValueError('reference outside job directory')
                    segments.append(dict(s,reference=str(ref),reference_sha256=digest(ref),text_ru='',state='needs_translation'))
                    seen.add(s['id']);last=s['end']
                if not segments:raise ValueError('source returned no speech segments')
                normalized=SourceProvider._cues(j['payload']['cues'])
                events=result.get('metadata',{}).get('non_speech_cues',[])
                if not isinstance(events,list) or any(event not in normalized or not is_non_speech(event['text']) for event in events):
                    raise ValueError('invalid non-speech caption exclusion')
                for cue in normalized:
                    if cue in events:continue
                    cursor=cue['start']
                    for segment in segments:
                        if segment['end']<=cursor:continue
                        if segment['start']>cursor+0.001:break
                        cursor=max(cursor,segment['end'])
                        if cursor>=cue['end']-0.001:break
                    if cursor<cue['end']-0.001:raise ValueError('prepared segments omit caption coverage')
                with self.lock:
                    j['segments']=segments;j['prepared']=True;j['source_metadata']=result.get('metadata',{});self._save(j)
            except PreparationCancelled:pass
            except Exception as exc:
                with self.lock:j['error']=f'{type(exc).__name__}: {exc}';self._save(j)
    def _finalize_segment(self, j, s, raw, metrics, engine_ref=None, anchor_candidate=None, raw_audio=None):
        """CPU post-processing stage: dead-silence trim, speech_start, PyAV mastering, anchor commit, atomic publish."""
        trimmed_raw = self.root / j['id'] / (s['id'] + '.trimmed.wav')
        final = self.root / j['id'] / (s['id'] + '.wav')
        try:
            if not isinstance(metrics, dict):
                raise ValueError('engine metrics must be an object or None')
            metrics = {k: v for k, v in metrics.items()
                       if k in ('generation_seconds', 'peak_allocated_mib', 'peak_reserved_mib',
                                'raw_audio_seconds', 'vae_decode_frames', 'takes_generated',
                                'rerank_seconds', 'rerank_score', 'rerank_cer', 'rerank_sim_spk',
                                'rerank_sim_anchor') and number(v)}
            audio_start = s['start']
            active_raw = raw
            active_arr = raw_audio[0] if raw_audio is not None else None
            raw_sr = raw_audio[1] if raw_audio is not None else 0
            can_rewrite = False
            trimmed_info = None
            trimmed_arr_info = None
            raw_seconds = 0.0
            if j.get('quality_profile'):
                raw_seconds = (len(active_arr) / float(raw_sr)) if (raw_audio is not None and raw_sr > 0) else wav_duration(raw)
                eff_seconds = raw_seconds
                with self.lock:
                    audio_start = speech_start(j['segments'], s, eff_seconds)
                window = s['end'] - audio_start
                # Tier 2: strip dead leading/trailing VoxCPM2 silence before spending LLM rewrites.
                if eff_seconds > window * QUALITY_MAX_TEMPO:
                    if raw_audio is not None:
                        trimmed_arr_info = trim_dead_silence_array(active_arr, raw_sr)
                        if trimmed_arr_info is not None:
                            t_arr, t_sec, saved_s = trimmed_arr_info
                            with self.lock:
                                t_start = speech_start(j['segments'], s, t_sec)
                            if t_sec <= (s['end'] - t_start) * QUALITY_MAX_TEMPO:
                                active_arr = t_arr
                                eff_seconds = t_sec
                                audio_start = t_start
                                window = s['end'] - audio_start
                                metrics['silence_trimmed_seconds'] = round(saved_s, 3)
                    else:
                        trimmed_info = trim_dead_silence(raw, trimmed_raw)
                        if trimmed_info is not None:
                            t_sec, saved_s = trimmed_info
                            with self.lock:
                                t_start = speech_start(j['segments'], s, t_sec)
                            if t_sec <= (s['end'] - t_start) * QUALITY_MAX_TEMPO:
                                active_raw = trimmed_raw
                                eff_seconds = t_sec
                                audio_start = t_start
                                window = s['end'] - audio_start
                                metrics['silence_trimmed_seconds'] = round(saved_s, 3)
                attempts = s.get('rewrite_attempts', 0)
                with self.lock:
                    s.setdefault('generation_attempts', []).append(
                        dict(text_ru=s['text_ru'], raw_seconds=raw_seconds)
                    )
                    rewrite_eligible = (audio_start == s['start'] or bool(s.get('continues_previous'))) and attempts < MAX_REWRITES
                    if rewrite_eligible and eff_seconds > window * QUALITY_MAX_TEMPO:
                        if j['cancelled'] or self.stop.is_set():
                            s['state'] = 'pending'
                            self._save(j)
                            return
                        fb = _timing_feedback(s, window, eff_seconds, attempts + 1)
                        s.update(state='needs_edit', rewrite_attempts=attempts + 1, rewrite_feedback=fb)
                        self._save(j)
                        return
                    can_rewrite = rewrite_eligible and not j['cancelled'] and not self.stop.is_set()
                metrics['rewrite_attempts'] = attempts

            started = time.perf_counter()
            fit_window = s['end'] - audio_start
            max_t = QUALITY_MAX_TEMPO if j.get('quality_profile') else None
            try:
                if raw_audio is not None:
                    duration = fit_audio_array(active_arr, raw_sr, final, fit_window, metrics=metrics, max_tempo=max_t)
                else:
                    duration = fit_audio(active_raw, final, fit_window, metrics=metrics, max_tempo=max_t)
            except TimingFitError:
                if can_rewrite:
                    with self.lock:
                        if not j['cancelled'] and not self.stop.is_set():
                            fb = _timing_feedback(s, window, eff_seconds, attempts + 1)
                            s.update(state='needs_edit', rewrite_attempts=attempts + 1, rewrite_feedback=fb)
                            self._save(j)
                            return
                if j.get('quality_profile'):
                    fb_limit = MAX_TEMPO if (s['end'] - s['start']) < SHORT_WINDOW_FALLBACK_S else QUALITY_FALLBACK_MAX_TEMPO
                    if raw_audio is not None:
                        if trimmed_arr_info is None:
                            trimmed_arr_info = trim_dead_silence_array(raw_audio[0], raw_sr)
                        fb_arr = trimmed_arr_info[0] if trimmed_arr_info is not None else raw_audio[0]
                        fb_sec = trimmed_arr_info[1] if trimmed_arr_info is not None else raw_seconds
                        with self.lock:
                            fb_start = speech_start(j['segments'], s, fb_sec, max_tempo=fb_limit, max_borrow=0.50)
                        fb_window = s['end'] - fb_start
                        try:
                            duration = fit_audio_array(fb_arr, raw_sr, final, fb_window, metrics=metrics, max_tempo=fb_limit)
                            audio_start = fb_start
                            metrics['tempo_fallback'] = True
                            if trimmed_arr_info is not None:
                                metrics['silence_trimmed_seconds'] = round(trimmed_arr_info[2], 3)
                        except TimingFitError:
                            if fb_window < MICRO_WINDOW_FALLBACK_S:
                                duration = fit_micro_window_array(raw_audio[0], raw_sr, final, fb_window, metrics=metrics)
                                audio_start = fb_start
                            else:
                                raise
                    else:
                        if trimmed_info is None:
                            trimmed_info = trim_dead_silence(raw, trimmed_raw)
                        fb_raw = trimmed_raw if trimmed_info is not None else raw
                        fb_sec = trimmed_info[0] if trimmed_info is not None else raw_seconds
                        with self.lock:
                            fb_start = speech_start(j['segments'], s, fb_sec, max_tempo=fb_limit, max_borrow=0.50)
                        fb_window = s['end'] - fb_start
                        try:
                            duration = fit_audio(fb_raw, final, fb_window, metrics=metrics, max_tempo=fb_limit)
                            audio_start = fb_start
                            metrics['tempo_fallback'] = True
                            if trimmed_info is not None:
                                metrics['silence_trimmed_seconds'] = round(trimmed_info[1], 3)
                        except TimingFitError:
                            if fb_window < MICRO_WINDOW_FALLBACK_S:
                                duration = fit_micro_window(raw, final, fb_window, metrics=metrics)
                                audio_start = fb_start
                            else:
                                raise
                else:
                    raise

            metrics['fit_seconds'] = time.perf_counter() - started
            clean_take = (not metrics.get('tempo_fallback') and
                          not metrics.get('micro_window_passthrough') and
                          not metrics.get('micro_window_trimmed'))
            if engine_ref is not None and clean_take:
                if anchor_candidate is not None and hasattr(engine_ref, '_commit_candidate'):
                    engine_ref._commit_candidate(anchor_candidate)
                elif hasattr(engine_ref, 'commit_continuation'):
                    engine_ref.commit_continuation(s['id'])
            with self.lock:
                s['audio_start'] = audio_start
                metrics['early_start_seconds'] = s['start'] - audio_start
                s['audio_duration'] = duration
                s['audio_sha256'] = digest(final)
                s['metrics'] = metrics
                s['state'] = 'ready'
                self._save(j)
        except Exception as exc:
            with self.lock:
                s['state'] = 'error'
                s['error'] = f'{type(exc).__name__}: {exc}'
                self._save(j)
        finally:
            raw.unlink(missing_ok=True)
            trimmed_raw.unlink(missing_ok=True)

    def _select_batch_partners(self, j, s, seg_payload, rescue_pending, rescue_lock):
        """Select up to 3 length-matched pending partners for B=8 / B=6 / B=4 super-batching."""
        partner_segs, partner_payloads = [], []
        len_1 = max(1, len((s.get('text_ru') or '').strip()))
        use_rr_1 = (
            self.engine._should_rerank(seg_payload, (s.get('text_ru') or '').strip())[0]
            if hasattr(self.engine, '_should_rerank') else True
        )
        slots_per_seg = 2 if use_rr_1 else 1
        eng_max_slots = getattr(self.engine, 'max_batch_size', 4)
        max_segs = max(1, eng_max_slots // slots_per_seg)
        with self.lock:
            if j['cancelled'] or j['error'] or max_segs < 2:
                return partner_segs, partner_payloads
            pos = j['priority']
            cands = [x for x in j['segments'] if x['state'] == 'pending']
            cands.sort(key=lambda x: (x['end'] <= pos, 0 if x['start'] <= pos < x['end'] else 1, x['start']))
            lookahead = 24 if max_segs >= 4 else (18 if max_segs >= 3 else 6)
            closest = bool(getattr(self, 'closest_partners', True))
            if closest and s['start'] < pos + 45.0:
                lookahead = min(lookahead, 10)
            eligible = []
            for rank, cand in enumerate(cands[:lookahead]):
                with rescue_lock:
                    if (j['id'], cand['id']) in rescue_pending:
                        continue
                cand_text = (cand.get('text_ru') or '').strip()
                len_2 = max(1, len(cand_text))
                ratio = max(len_1, len_2) / min(len_1, len_2)
                if ratio > 1.65:
                    continue
                cand_view = {
                    'id': cand['id'], 'job_id': j['id'], 'speaker': cand['speaker'],
                    'reference': cand['reference'], 'reference_sha256': cand.get('reference_sha256'),
                    'start': cand['start'], 'end': cand['end'],
                }
                if hasattr(self.engine, 'needs_anchor') and self.engine.needs_anchor(cand_view):
                    continue
                if hasattr(self.engine, '_should_rerank'):
                    use_rr_2 = self.engine._should_rerank(cand_view, cand_text)[0]
                    if use_rr_2 != use_rr_1:
                        continue
                try:
                    if digest(cand['reference']) != cand['reference_sha256']:
                        continue
                except OSError:
                    continue
                eligible.append((ratio, rank, cand))

            if closest and len(eligible) > 1:
                eligible.sort(key=lambda e: (e[0] + 0.004 * e[1], e[1]))

            chosen = []
            if max_segs >= 4:
                tight8 = [e for e in eligible if e[0] <= 1.35]
                if len(tight8) >= 3:
                    chosen = tight8[:3]
            if not chosen and max_segs >= 3:
                tight6 = [e for e in eligible if e[0] <= 1.45]
                if len(tight6) >= 2:
                    chosen = tight6[:2]
            if not chosen and eligible:
                chosen = eligible[:1]
            for _, _, cand in chosen:
                cand['state'] = 'generating'
                cand_payload = copy.deepcopy(cand)
                cand_payload['job_id'] = j['id']
                partner_segs.append(cand)
                partner_payloads.append(cand_payload)
        return partner_segs, partner_payloads

    def _select_rescue_partners(self, j, s, rescue_pending, rescue_lock):
        """Select up to 3 pending Take-3 rescue partners (1 slot each) to batch in one GPU call."""
        partner_segs, partner_payloads, partner_priors = [], [], []
        len_1 = max(1, len((s.get('text_ru') or '').strip()))
        eng_max_slots = getattr(self.engine, 'max_batch_size', 4)
        max_partners = max(0, min(3, eng_max_slots - 1))
        if max_partners < 1:
            return partner_segs, partner_payloads, partner_priors
        with self.lock:
            if j['cancelled'] or j['error']:
                return partner_segs, partner_payloads, partner_priors
            for cand in j['segments']:
                if len(partner_segs) >= max_partners:
                    break
                if cand['state'] != 'pending':
                    continue
                with rescue_lock:
                    if (j['id'], cand['id']) not in rescue_pending:
                        continue
                len_2 = max(1, len((cand.get('text_ru') or '').strip()))
                if max(len_1, len_2) / min(len_1, len_2) > 1.50:
                    continue
                try:
                    if digest(cand['reference']) != cand['reference_sha256']:
                        continue
                except OSError:
                    continue
                with rescue_lock:
                    c_prior = rescue_pending.pop((j['id'], cand['id']), None)
                if c_prior is None:
                    continue
                cand['state'] = 'generating'
                cand_payload = copy.deepcopy(cand)
                cand_payload['job_id'] = j['id']
                partner_segs.append(cand)
                partner_payloads.append(cand_payload)
                partner_priors.append(c_prior)
        return partner_segs, partner_payloads, partner_priors

    def _engine_loop(self):
        last_used = time.monotonic()
        post_queue = queue.Queue(maxsize=8)
        rerank_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='cpu-rerank')
        rescue_pending = {}
        rescue_lock = threading.Lock()
        post_stop = object()

        def _post_worker():
            import soundfile as sf
            while True:
                item = post_queue.get()
                if item is post_stop:
                    post_queue.task_done()
                    return
                j, s, eval_future, raw_bundle, prior_info, eng = item
                raw = self.root / j['id'] / (s['id'] + '.raw.wav')
                try:
                    outcome = eval_future.result()
                    if outcome.get('needs_rescue'):
                        with self.lock:
                            is_stopped = j['cancelled'] or self.stop.is_set()
                            if not is_stopped:
                                with rescue_lock:
                                    rescue_pending[(j['id'], s['id'])] = {
                                        'candidates': outcome['candidates'],
                                        'gpu_seconds': raw_bundle['gpu_seconds'],
                                        'rerank_seconds': outcome['rerank_seconds'],
                                    }
                            else:
                                with rescue_lock:
                                    rescue_pending.pop((j['id'], s['id']), None)
                            s['state'] = 'pending'
                            self._save(j)
                        continue
                    metrics = outcome['metrics']
                    if prior_info:
                        prior_rr = float(prior_info.get('rerank_seconds', 0.0))
                        metrics['rerank_seconds'] = round(float(metrics.get('rerank_seconds', 0.0)) + prior_rr, 4)
                        metrics['generation_seconds'] = float(metrics.get('generation_seconds', 0.0)) + prior_rr
                    self._finalize_segment(
                        j, s, raw, metrics,
                        engine_ref=eng,
                        anchor_candidate=outcome.get('anchor_candidate'),
                        raw_audio=(outcome['wav'], outcome['sr']),
                    )
                except Exception as exc:
                    raw.unlink(missing_ok=True)
                    with self.lock:
                        s['state'] = 'error'
                        s['error'] = f'{type(exc).__name__}: {exc}'
                        self._save(j)
                finally:
                    # Release references before blocking on post_queue.get() so idle engine unload frees VRAM.
                    item = j = s = eval_future = raw_bundle = prior_info = eng = None
                    post_queue.task_done()

        def _submit_post(j, seg, raw_bundle, prior_info, eng):
            prior_cands = prior_info['candidates'] if prior_info else None
            fut = rerank_pool.submit(eng.evaluate_raw, raw_bundle, prior_cands)
            post_queue.put((j, seg, fut, raw_bundle, prior_info, eng))

        post_thread = threading.Thread(target=_post_worker, name='cpu-postprocess', daemon=True)
        post_thread.start()
        try:
            while not self.stop.is_set():
                selected = self.next_segment()
                if selected is None:
                    if not post_queue.empty():
                        post_queue.join()
                        last_used = time.monotonic()
                        continue
                    if self.engine is not None and time.monotonic() - last_used >= self.idle_timeout:
                        post_queue.join()
                        self._release_engine()
                    self.stop.wait(.1)
                    continue
                j, s = selected
                # Release local LLM VRAM (e.g. Ollama) before loading VoxCPM2 onto GPU.
                if self.engine is None and self.editor is not None and hasattr(self.editor, 'release_vram'):
                    self.editor.release_vram()
                available = self.memory_probe() if self.memory_probe is not None else None
                required = s.get('gpu_required_mib', MIN_GPU_MIB)
                if available is not None and available < required:
                    post_queue.join()
                    self._release_engine()
                    capacity = self.capacity_probe() if self.capacity_probe is not None else None
                    if capacity is not None and required > capacity:
                        with self.lock:
                            j.pop('gpu_wait', None)
                            s.update(state='error', error='Требуемый резерв видеопамяти превышает объём GPU; ожидание не поможет.',
                                     gpu_required_mib=required, gpu_capacity_mib=capacity)
                            self._save(j)
                        continue
                    with self.lock:
                        j['gpu_wait'] = dict(available_mib=available, required_mib=required,
                            retry_after=time.time() + GPU_RETRY_SECONDS, reason='memory_busy')
                        self._save(j)
                    continue
                with self.lock:
                    if j['cancelled']:
                        continue
                    j.pop('gpu_wait', None)
                    s['state'] = 'generating'
                oom_cleanup = False
                partner_segs = []
                raw = self.root / j['id'] / (s['id'] + '.raw.wav')
                try:
                    if digest(s['reference']) != s['reference_sha256']:
                        raise ValueError('reference changed since preparation')
                    if self.engine is None:
                        if self.engine_factory is None:
                            from .engine import VoxEngine
                            self.engine_factory = VoxEngine
                        started = time.perf_counter()
                        self.engine = self.engine_factory()
                        with self.lock:
                            j['model_load_count'] = j.get('model_load_count', 0) + 1
                            j['last_model_load_seconds'] = time.perf_counter() - started
                    started = time.perf_counter()
                    seg_payload = copy.deepcopy(s)
                    seg_payload['job_id'] = j['id']

                    use_async_pipeline = (
                        hasattr(self.engine, 'generate_raw')
                        and hasattr(self.engine, 'evaluate_raw')
                        and getattr(self.engine, '_supports_batched', True)
                    )
                    if use_async_pipeline:
                        with rescue_lock:
                            prior_info = rescue_pending.pop((j['id'], s['id']), None)
                        wait_for_anchor = hasattr(self.engine, 'needs_anchor') and self.engine.needs_anchor(seg_payload)

                        partner_payloads, partner_priors = [], []
                        if not wait_for_anchor and hasattr(self.engine, 'generate_raw_multi'):
                            if prior_info is None:
                                partner_segs, partner_payloads = self._select_batch_partners(
                                    j, s, seg_payload, rescue_pending, rescue_lock,
                                )
                            else:
                                partner_segs, partner_payloads, partner_priors = self._select_rescue_partners(
                                    j, s, rescue_pending, rescue_lock,
                                )

                        if partner_segs and partner_payloads:
                            batch_segs = [s] + partner_segs
                            batch_payloads = [seg_payload] + partner_payloads
                            batch_priors = [prior_info] + partner_priors if prior_info is not None else [None] * len(batch_segs)
                            try:
                                if prior_info is not None:
                                    bundles = self.engine.generate_raw_multi(batch_payloads, rescue_only=True)
                                else:
                                    bundles = self.engine.generate_raw_multi(batch_payloads)
                            except Exception:
                                with self.lock:
                                    for ps in partner_segs:
                                        if ps['state'] == 'generating':
                                            ps['state'] = 'pending'
                                if prior_info is not None:
                                    with rescue_lock:
                                        for ps, pp in zip(partner_segs, partner_priors):
                                            rescue_pending[(j['id'], ps['id'])] = pp
                                partner_segs = []
                                post_queue.join()
                                self._release_engine()
                                raise
                            submitted = 0
                            try:
                                for b_seg, b_raw, b_prior in zip(batch_segs, bundles, batch_priors):
                                    if b_prior is not None:
                                        b_raw['gpu_seconds'] += float(b_prior.get('gpu_seconds', 0.0))
                                    _submit_post(j, b_seg, b_raw, b_prior, self.engine)
                                    submitted += 1
                            except Exception:
                                with self.lock:
                                    for rem_seg in batch_segs[submitted:]:
                                        if rem_seg['state'] == 'generating':
                                            rem_seg['state'] = 'pending'
                                partner_segs = []
                                raise
                            partner_segs = []
                            last_used = time.monotonic()
                            continue

                        try:
                            raw_bundle = self.engine.generate_raw(seg_payload, rescue_only=prior_info is not None)
                        except Exception:
                            post_queue.join()
                            self._release_engine()
                            raise
                        if prior_info is not None:
                            raw_bundle['gpu_seconds'] += float(prior_info.get('gpu_seconds', 0.0))
                        _submit_post(j, s, raw_bundle, prior_info, self.engine)
                        if wait_for_anchor:
                            # Wait once on the speaker's first segment so its hop-1 anchor locks for segment #2+.
                            post_queue.join()
                        last_used = time.monotonic()
                        continue

                    try:
                        metrics = self.engine.synthesize(seg_payload, raw)
                    except Exception:
                        # A failed CUDA context/model must not be reused on explicit retry.
                        post_queue.join()
                        self._release_engine()
                        raise
                    if metrics is None:
                        metrics = {'generation_seconds': time.perf_counter() - started}
                    self._finalize_segment(j, s, raw, metrics, engine_ref=self.engine)
                except Exception as exc:
                    raw.unlink(missing_ok=True)
                    if is_cuda_oom(exc):
                        post_queue.join()
                        self._release_engine()
                        oom_cleanup = True
                        with self.lock:
                            attempts = s.get('gpu_oom_attempts', 0) + 1
                            s['gpu_oom_attempts'] = attempts
                            s['last_gpu_error'] = str(exc)[:4000]
                            s['gpu_required_mib'] = max(required, available or required) + 256
                            if attempts < MAX_OOM_RETRIES:
                                s['state'] = 'pending'
                                j['gpu_wait'] = dict(available_mib=available, required_mib=s['gpu_required_mib'],
                                    retry_after=time.time() + GPU_RETRY_SECONDS, reason='out_of_memory')
                            else:
                                s.update(state='error', error='Недостаточно видеопамяти для озвучки. Освободите память GPU и запустите озвучку снова.')
                            self._save(j)
                    else:
                        with self.lock:
                            s['state'] = 'error'
                            s['error'] = f'{type(exc).__name__}: {exc}'
                            self._save(j)
                finally:
                    last_used = time.monotonic()
                if oom_cleanup:
                    # Exception frames held CUDA tensors during close(); release the
                    # allocator only after leaving the except block and clearing them.
                    clear_cuda_memory()
        finally:
            post_queue.join()
            post_queue.put(post_stop)
            post_thread.join(timeout=60)
            rerank_pool.shutdown(wait=True)
            if not post_thread.is_alive():
                self._release_engine()
    def _release_engine(self):
        engine,self.engine=self.engine,None
        if engine is not None and hasattr(engine,'close'):engine.close()
