# Alfie

```bash
conda create --name alfie python==3.11.7
conda activate alpha
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install git+https://github.com/huggingface/diffusers
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/SunzeY/AlphaCLIP.git
```
