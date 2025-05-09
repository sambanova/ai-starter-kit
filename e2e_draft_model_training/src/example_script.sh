PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /your/path/to/draft_model_training
# (Build and) Activate your virtual environment
python3.12 -m venv .venv  # (optional, comment if already done)
source .venv/bin/activate
pip install -r requirements.txt  # (optional, comment if already done)

python data_generation.py
python data_preparation.py
TORCH_USE_CUDA_DSA=1 python draft_model_training.py