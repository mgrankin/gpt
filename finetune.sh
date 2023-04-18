yes|sudo apt install libaio-dev llvm-10-dev autoconf  libnuma-dev 
yes|sudo apt uninstall llvm-9-dev
########################################################################################
cd 
git clone https://github.com/mgrankin/ru-gpts.git
cd ru-gpts
mamba env create 
conda activate rugpt
########################################################################################
cd ~/dev/gpt
conda activate base
conda env remove -n rugptdev
mamba env create
conda activate rugptdev
########################################################################################
cd
mkdir .install
cd .install
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
########################################################################################
ds_report
pip uninstall -y deepspeed
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install git+https://github.com/mgrankin/DeepSpeed.git
ds_report
########################################################################################
conda activate rugpt
cd
cd ru-gpts/scripts

world_size=1 OMP_NUM_THREADS=1 sh deepspeed_gpt3_xl_finetune.sh
# validation loss at iteration 0 | LM loss: 2.7066 | LM PPL: 14.978
# validation loss at iteration 2300 | LM loss: 2.6026 | LM PPL: 13.498
cd pelevin
python zero_to_fp32.py . pelevin.model

world_size=1 OMP_NUM_THREADS=1 sh deepspeed_gpt3_xl_finetune_poetry.sh
# validation loss at iteration 0 | LM loss: 2.9472 | LM PPL: 19.053
# validation loss at iteration 1100 | LM loss: 2.8809 | LM PPL: 17.830
cd poetry
python zero_to_fp32.py . poetry.model
