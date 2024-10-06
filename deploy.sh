cd ~/gpt/

cd front
docker build -t front .
cd ..

docker buildx build -t model .

docker buildx build -t vllm -f Dockerfile_vllm .

docker-compose down --remove-orphans -v

docker-compose up --remove-orphans --force-recreate

#docker-compose stop front
#docker-compose up --remove-orphans --force-recreate front

#docker image prune
