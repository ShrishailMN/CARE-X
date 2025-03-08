# gunicorn_config.py
bind = "0.0.0.0:10000"
workers = 1
threads = 1
worker_class = 'sync'  # Changed to sync for lower memory usage
timeout = 120
max_requests = 50
max_requests_jitter = 10
preload_app = False
worker_tmp_dir = '/tmp'
daemon = False
keepalive = 2
