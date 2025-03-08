# gunicorn_config.py
bind = "0.0.0.0:10000"
workers = 1
threads = 2
timeout = 300
worker_class = 'gthread'
max_requests = 1000
max_requests_jitter = 50
preload_app = True
