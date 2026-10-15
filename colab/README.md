# Colab verification

The settings in [RUN_GUIDE.md](../RUN_GUIDE.md) were checked on a free-tier Google Colab T4, driven from a terminal with the
official [Colab CLI](https://github.com/googlecolab/google-colab-cli):

```bash
uv tool install google-colab-cli
colab --auth=oauth2 new -s insid3 --gpu T4            # browser consent once; tick all permission boxes
echo 'import subprocess; subprocess.run("git clone https://github.com/sam1511x/INSID3-ADL-Project.git INSID3", shell=True)' \
  | colab --auth=oauth2 exec -s insid3
# then, on the VM:  bash scripts/download_data.sh lung spair isic coco && bash colab/verify_queue.sh
colab --auth=oauth2 download -s insid3 /content/INSID3/output/<run>/results.json .   # copy results back
colab --auth=oauth2 stop -s insid3
```

`verify_queue.sh` caps PyTorch at 8 GB (`--max-vram-gb 8`) so every verified setting also fits the 8 GB target GPU.
Free-tier sessions are reclaimed without warning, so download `output/*/results.json` after each stage.
Results: [results/COLAB_VERIFICATION.md](../results/COLAB_VERIFICATION.md).
