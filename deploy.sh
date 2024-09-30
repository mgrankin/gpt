cd front
docker build -t front .
cd ..

docker buildx build -t model .

docker buildx build -t vllm -f Dockerfile_vllm .


docker-compose stop frida1
docker-compose up --remove-orphans --force-recreate frida1

docker-compose down --remove-orphans -v
docker-compose up --remove-orphans --force-recreate

-d


docker image prune
#docker-compose pull


#docker run --gpus '"device=0"' --shm-size 1g  -p 8001:8000 --env MODEL=xl/pelevin model
#docker run --gpus '"device=1"' --shm-size 1g  -p 8000:8000 --env MODEL=large/pelevin model 

docker run -p 8280:8000 front