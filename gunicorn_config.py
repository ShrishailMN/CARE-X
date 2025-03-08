# gunicorn_config.py
workers = 2
threads = 2
timeout = 120
worker_class = 'gthread'
max_requests = 1000
max_requests_jitter = 50
