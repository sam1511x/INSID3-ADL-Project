# ==== Paste this whole cell into a Colab notebook (Runtime > Change runtime type > T4 GPU) and run it ====
# Before running: add your HuggingFace token in Colab's 🔑 Secrets panel as  HF_TOKEN  (enable notebook access).
import os, subprocess, time, re, pathlib

PUBKEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIBV8r9aootSXWT6DXguJlOC0xitgO90G9PfDbt9cjg0k insid3-colab"

# HF token -> standard huggingface cache location (readable from SSH sessions)
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    pathlib.Path("/root/.cache/huggingface").mkdir(parents=True, exist_ok=True)
    pathlib.Path("/root/.cache/huggingface/token").write_text(tok.strip())
    print("HF token stored.")
except Exception as e:
    print("!! HF_TOKEN secret not found:", e)

# sshd with key-only auth
subprocess.run("apt-get -qq update && apt-get -qq install -y openssh-server > /dev/null", shell=True)
pathlib.Path("/root/.ssh").mkdir(mode=0o700, exist_ok=True)
pathlib.Path("/root/.ssh/authorized_keys").write_text(PUBKEY + "\n")
os.chmod("/root/.ssh/authorized_keys", 0o600)
subprocess.run("mkdir -p /run/sshd && echo 'PermitRootLogin prohibit-password\nPasswordAuthentication no' >> /etc/ssh/sshd_config && /usr/sbin/sshd", shell=True)
# expose the GPU/CUDA env of the notebook to SSH sessions
with open("/root/.bashrc", "a") as f:
    for k in ["PATH", "LD_LIBRARY_PATH", "CUDA_HOME"]:
        if k in os.environ:
            f.write(f'export {k}="{os.environ[k]}"\n')

# free cloudflare quick tunnel (no account needed)
subprocess.run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared", shell=True)
log = open("/content/cloudflared.log", "w")
subprocess.Popen(["cloudflared", "tunnel", "--no-autoupdate", "--url", "ssh://localhost:22"], stdout=log, stderr=log)
host = None
for _ in range(60):
    time.sleep(1)
    m = re.search(r"https://([a-z0-9-]+\.trycloudflare\.com)", open("/content/cloudflared.log").read())
    if m:
        host = m.group(1); break
print("\n>>> SEND THIS TO CLAUDE:", host)
!nvidia-smi --query-gpu=name,memory.total --format=csv
# Keep this tab open; the session must stay alive while experiments run.
