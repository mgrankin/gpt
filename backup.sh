#!/bin/zsh
# run command "env -i sh" and test there berore putting in cron

# remove?
# eval "$($HOME/.anaconda/bin/conda shell.zsh hook)"

# order matters - first give non-critical errors
source ~/.zshrc # zshrc because of SNAP in PATH for gsutil
set -e # exit on errors

conda activate
#fname=$(date +"%Y-%m-%d").zstd.gpg
fname=/home/u/.logs/porf_logs.zstd.gpg
pg_dump postgresql://postgres:rlp4ZKc6oC0OzgK1FSsJ@localhost:5535 |zstd | gpg -q --symmetric --cipher-algo AES256 --yes --batch --passphrase 9xvXtMVjolXBkaJEXp8W > $fname
gsutil cp $fname gs://porfirevich/ 
#rm $fname
