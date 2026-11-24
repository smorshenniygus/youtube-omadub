"""Persisted client ownership. Expired clients cannot silently resume a job."""
import math

LEASE_SECONDS = 180
MAX_CLIENTS = 64


class ClientLeases:
    @staticmethod
    def initialize(job, now):
        # Old releases persisted IDs only. Give those clients one migration grace
        # period; never reset existing deadlines on each service restart.
        if 'lease_deadlines' not in job:
            job['lease_deadlines'] = {client: now + LEASE_SECONDS for client in job.get('clients', [])}
        return job['lease_deadlines']

    @staticmethod
    def expire(job, now):
        deadlines = ClientLeases.initialize(job, now)
        expired = [client for client, end in deadlines.items()
                   if type(end) not in (int, float) or not math.isfinite(end) or end <= now]
        for client in expired:
            del deadlines[client]
        job['clients'] = list(deadlines)
        if expired and not deadlines:
            job['cancelled'] = True
        return bool(expired)

    @staticmethod
    def acquire(job, client, now):
        ClientLeases.expire(job, now)
        deadlines = job['lease_deadlines']
        if client is not None:
            if client not in deadlines and len(deadlines) >= MAX_CLIENTS:
                raise ValueError('too many active clients')
            deadlines[client] = now + LEASE_SECONDS
        job['clients'] = list(deadlines)
        job['cancelled'] = False  # Explicit create is the only resume operation.

    @staticmethod
    def renew(job, client, now):
        ClientLeases.expire(job, now)
        deadlines = job['lease_deadlines']
        if client not in deadlines or job['cancelled']:
            raise ValueError('client lease expired; create the job again')
        deadlines[client] = now + LEASE_SECONDS
        return deadlines[client]

    @staticmethod
    def release(job, client, now):
        ClientLeases.expire(job, now)
        deadlines = job['lease_deadlines']
        if client is None:
            deadlines.clear()
        else:
            deadlines.pop(client, None)
        job['clients'] = list(deadlines)
        job['cancelled'] = not deadlines
