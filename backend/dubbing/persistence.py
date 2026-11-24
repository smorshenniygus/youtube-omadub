"""Validate durable jobs independently; quarantine metadata without deleting audio."""
import json
import logging
import math
from pathlib import Path
import re

SCHEMA_VERSION = 1
MAX_JOB_BYTES = 128 * 1024 * 1024
JOB_ID = re.compile(r'[a-f0-9]{32}\Z')
SEGMENT_ID = re.compile(r'[A-Za-z0-9_-]{1,80}\Z')
SHA256 = re.compile(r'[a-f0-9]{64}\Z')
STATES = {'needs_translation', 'needs_edit', 'editing', 'pending', 'generating', 'ready', 'error'}


class QuarantinedJobError(RuntimeError):
    """Explicit retry must never overwrite an unvalidated persisted job."""


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def invalid_constant(value):
    raise ValueError('nonfinite JSON constant')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


class JobStore:
    def __init__(self, root, validate_payload):
        self.root = Path(root).resolve()
        self.validate_payload = validate_payload
        self.quarantined = {}

    def validate(self, job, directory):
        if not isinstance(job, dict) or not JOB_ID.fullmatch(directory.name) or job.get('id') != directory.name:
            raise ValueError('job identity mismatch')
        version = job.get('schema_version', 0)
        if type(version) is not int or version not in (0, SCHEMA_VERSION):
            raise ValueError('unsupported job schema version')
        self.validate_payload(job['payload'])
        duration = job['payload']['duration']
        if any(type(job.get(key)) is not bool for key in ('prepared', 'cancelled')):
            raise ValueError('invalid job flags')
        if not finite(job.get('created')) or not finite(job.get('priority')) or not 0 <= job['priority'] <= duration:
            raise ValueError('invalid job scheduling metadata')
        if job.get('error') is not None and not isinstance(job['error'], str):
            raise ValueError('invalid job error')
        segments = job.get('segments')
        if not isinstance(segments, list) or len(segments) > 100000 or (job['prepared'] and not segments):
            raise ValueError('invalid segment collection')
        seen, previous_end = set(), 0
        for segment in segments:
            previous_end = self.validate_segment(segment, directory, duration, previous_end, seen)
        clients = job.get('clients', [])
        if not isinstance(clients, list) or len(clients) > 100000:
            raise ValueError('invalid client collection')
        if any(not isinstance(client, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,96}', client) for client in clients):
            raise ValueError('invalid persisted client')
        deadlines = job.get('lease_deadlines', {})
        if not isinstance(deadlines, dict) or any(not finite(end) for end in deadlines.values()):
            raise ValueError('invalid lease deadlines')
        if 'lease_deadlines' in job and set(deadlines) != set(clients):
            raise ValueError('lease identities disagree')
        metadata = job.get('source_metadata', {})
        if not isinstance(metadata, dict) or not isinstance(metadata.get('references', {}), dict):
            raise ValueError('invalid source metadata')
        if any(not isinstance(value, dict) for value in metadata.get('references', {}).values()):
            raise ValueError('invalid reference metadata')
        return job

    @staticmethod
    def validate_segment(segment, directory, duration, previous_end, seen):
        if not isinstance(segment, dict):
            raise ValueError('invalid segment')
        identity = segment.get('id')
        if not isinstance(identity, str) or not SEGMENT_ID.fullmatch(identity) or identity in seen:
            raise ValueError('invalid segment identity')
        seen.add(identity)
        if segment.get('state') not in STATES:
            raise ValueError('invalid segment state')
        start, end = segment.get('start'), segment.get('end')
        if not finite(start) or not finite(end) or not previous_end <= start < end <= duration:
            raise ValueError('invalid segment timing')
        if any(not isinstance(segment.get(key), str) for key in ('speaker', 'text', 'text_ru', 'reference', 'reference_sha256')):
            raise ValueError('invalid segment text/reference fields')
        reference = Path(segment['reference']).resolve()
        if not reference.is_relative_to(directory) or not SHA256.fullmatch(segment['reference_sha256']):
            raise ValueError('invalid reference lineage')
        if segment['state'] == 'ready':
            if not finite(segment.get('audio_duration')) or segment['audio_duration'] <= 0:
                raise ValueError('invalid ready audio duration')
            if not isinstance(segment.get('audio_sha256'), str) or not SHA256.fullmatch(segment['audio_sha256']):
                raise ValueError('invalid ready audio hash')
            audio_start = segment.get('audio_start', start)
            if not finite(audio_start) or not 0 <= audio_start <= start or audio_start + segment['audio_duration'] > end + .001:
                raise ValueError('invalid ready audio bounds')
        return end

    def quarantine(self, path, error):
        self.quarantined[path.parent.name] = type(error).__name__
        logging.warning('Quarantined persisted job %s: %s', path.parent.name, type(error).__name__)

    def load(self):
        for path in sorted(self.root.glob('*/job.json')):
            directory = path.parent.resolve()
            try:
                if directory.parent != self.root or path.is_symlink():
                    raise ValueError('job path escapes storage')
                if path.stat().st_size > MAX_JOB_BYTES:
                    raise ValueError('oversized job metadata')
                job = json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid_constant, object_pairs_hook=unique_object)
                yield path, self.validate(job, directory)
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
                # No rename, deletion, normalization or overwrite of the original
                # job.json/WAVs. Diagnostics are memory-only and in the local log.
                self.quarantine(path, error)
