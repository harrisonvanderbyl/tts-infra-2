apt update && apt install tmux -y && apt install ffmpeg -y



pip install -r ./requirements.txt 
pip install transformers -U
huggingface-cli download fishaudio/fish-speech-1.4 --local-dir voice_servers/models/fish/checkpoints/fish-speech-1.4/
cd voice_servers/models/fish/
tmux new-session -d -s fish 'python3 ./start.py'