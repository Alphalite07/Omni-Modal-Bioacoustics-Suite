import webview
import subprocess
import time
import sys
import os

def start_streamlit():
    """Silently boots the AI server in the background."""
    print("Initializing Bioacoustics Server...")
    # Runs Streamlit in 'headless' mode so it doesn't try to open Chrome
    env = os.environ.copy()
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

if __name__ == '__main__':
    # 1. Start the backend AI engine
    start_streamlit()
    
    # 2. Give the heavy PyTorch models a second to load into RAM
    time.sleep(3) 
    
    # 3. Spawn the native desktop window
    print("Launching Desktop UI...")
    window = webview.create_window(
        title="Bioacoustics Pro Suite", 
        url="http://localhost:8501",
        width=1400,
        height=900,
        min_size=(1000, 700)
    )
    
    webview.start()