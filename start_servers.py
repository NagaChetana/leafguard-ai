import os
import subprocess
import sys
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

env = os.environ.copy()
env['PYTHONPATH'] = BASE_DIR
env['KERAS_HOME'] = os.path.join(BASE_DIR, '.keras')

print("Starting FastAPI Backend Server on http://localhost:8000 ...")
backend_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=BASE_DIR,
    env=env
)

print("Starting Vite Frontend Dev Server on http://localhost:5173 ...")
frontend_proc = subprocess.Popen(
    ["npm", "--prefix", "frontend", "run", "dev"],
    cwd=BASE_DIR,
    env=env
)

time.sleep(4)
print("\n🚀 Both servers started successfully!")
print("Backend API : http://localhost:8000")
print("Frontend UI : http://localhost:5173")
