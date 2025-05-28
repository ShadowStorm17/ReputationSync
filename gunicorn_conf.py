import multiprocessing
import os

# Server socket
bind = "unix:/run/instagram_api.sock"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
timeout = 30
keepalive = 2

# Process naming
proc_name = "instagram_api"
pythonpath = "/opt/instagram_stats_api"

# Logging
accesslog = "/var/log/instagram_api/access.log"
errorlog = "/var/log/instagram_api/error.log"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(L)s'

# SSL
keyfile = os.getenv("SSL_KEYFILE")
certfile = os.getenv("SSL_CERTFILE")

# Process management
daemon = False
pidfile = "/run/instagram_api.pid"
umask = 0o027
user = "instagram_api"
group = "instagram_api"

# Server mechanics
preload_app = True
reload = False
max_requests = 1000
max_requests_jitter = 50

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def pre_fork(server, worker):
    pass

def pre_exec(server):
    server.log.info("Forked child, re-executing.")

def when_ready(server):
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def worker_abort(worker):
    worker.log.info("worker received SIGABRT signal") 