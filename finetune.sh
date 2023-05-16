########################################################################################
#
cd 
git clone https://github.com/mgrankin/gpt.git
model=pelevin
docker run --name GPT --gpus all -it --shm-size 1g -p 8201:8080 \
        -v $HOME/gpt:/gpt -e model=$model \
        --rm nvcr.io/nvidia/pytorch:23.01-py3 
         
# inside
apt update
yes|apt install libaio-dev autoconf  libnuma-dev libpq-dev
pip install py-cpuinfo
pip install triton==1.0.0
DS_BUILD_CPU_ADAM=1 DS_BUILD_SPARSE_ATTN=1 pip install git+https://github.com/Microsoft/DeepSpeed.git
ds_report
cd /gpt
pip install -r requirements.txt

# outside 
docker commit GPT gptimage

#
docker run  --rm --name GPT1 --network="host" --gpus all -it --shm-size 1g -p 8201:8080 \
        -v $HOME/gpt:/gpt -e model=$model gptimage

cd /gpt; 
XL=1 CUDA_VISIBLE_DEVICES=0 INSTANCE="0"  MODEL="pelevin" uvicorn api:app --reload --host 0.0.0.0 --port 8080

docker run --name GPT --gpus all -it --shm-size 1g -p 8201:8080 \
        -v $volume:/gpt -e model=$model \
        --rm nvcr.io/nvidia/pytorch:23.01-py3 

docker commit --change 'ENTRYPOINT ["/bin/run.sh"]' 3d555451f07a mymatlab:r2020a


docker run --name LAMA --gpus '"device=3"' --shm-size 1g -p 999:80  \
   -v $volume:/gpt ghcr.io/huggingface/text-generation-inference:0.6   \
    -e model=$model 

docker update LAMA --restart unless-stopped 


yes|sudo apt install libaio-dev autoconf  libnuma-dev 
 llvm-10-dev
yes|sudo apt uninstall llvm-9-dev
########################################################################################
cd 
git clone https://github.com/mgrankin/ru-gpts.git
cd ru-gpts
mamba env create 
conda activate rugpt
########################################################################################
pip install -U --pre triton
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
sudo rm -R apex
git clone https://github.com/NVIDIA/apex
cd apex
MAX_JOBS=32 pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
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
