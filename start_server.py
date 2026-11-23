import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Start uvicorn in a subprocess
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    cwd=Path(__file__).resolve().parent,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# Wait for startup
time.sleep(3)

# Test the health endpoint
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/health") as resp:
        print("Health:", resp.read().decode())
except Exception as e:
    print("Error:", e)

# Test lanes endpoint
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/lanes") as resp:
        print("Lanes:", resp.read().decode())
except Exception as e:
    print("Error getting lanes:", e)

# Test models endpoint
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/api/models") as resp:
        print("Models:", resp.read().decode())
except Exception as e:
    print("Error getting models:", e)

# Stop the server after testing
proc.terminate()
proc.wait()
print("Done testing")
