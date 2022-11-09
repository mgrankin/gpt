CONTAINER=porfirevich2
CONTAINER2=porfirevich
export PGHOST=localhost
export PGPORT=5535
export PGUSER=postgres
 export PGPASSWORD=rlp4ZKc6oC0OzgK1FSsJ
CONNECT=postgresql://postgres:$PGPASSWORD@$PGHOST:$PGPORT
REPO=postgres


# backup 
time pg_dump -Fd -j 32 postgres -f backup 
# pg_dump > back.sql
# pg_dump |zstd | gpg -q --symmetric --cipher-algo AES256 --yes --batch --passphrase hpuBzn4lQAS3la76T0Db > backup.zstd.gpg 

docker pull $REPO
docker stop $CONTAINER; docker rm $CONTAINER2 ; \
docker run --name $CONTAINER2 --shm-size=2g -e POSTGRES_PASSWORD=$PGPASSWORD -p $PGPORT:5432 -d $REPO 
docker update $CONTAINER2 --restart unless-stopped 

psql 
select version();

time pg_restore -j 32 -d postgres -C backup
# recovery
# cat back.sql | pv | psql 
# cat backup.zstd.gpg | gpg -d --batch --passphrase hpuBzn4lQAS3la76T0Db | zstd -d -c > back.sql 
# cat back.sql | pv | psql postgresql://postgres:rlp4ZKc6oC0OzgK1FSsJ@localhost:5433 

docker rename $CONTAINER old_container
docker rename $CONTAINER2 $CONTAINER


################################################################################################################
# pg_hint_plan

docker exec -it $CONTAINER /bin/bash

apt-get update && apt-get install -y git build-essential postgresql-server-dev-14
cd && git clone http://scm.osdn.jp/gitroot/pghintplan/pg_hint_plan.git
cd pg_hint_plan
#git checkout PG13
#ln -s /usr/include/postgresql/13 /usr/include/postgresql/14
#ln -s /usr/share/postgresql/13 /usr/share/postgresql/14
#ln -s /usr/lib/postgresql/13 /usr/lib/postgresql/14
make && make install

psql 

select version();
LOAD 'pg_hint_plan';
CREATE EXTENSION pg_hint_plan;

################################################################################################################
# debug
alter database set pg_hint_plan.debug_print on;

SET pg_hint_plan.debug_print TO on;
SET pg_hint_plan.enable_hint TO on;

/*+ SeqScan(c) */
explain
select * from cities c where id = 1;


################################################################################################################
# DB Version: 14
# OS Type: linux
# DB Type: dw
# Total Memory (RAM): 32 GB
# CPUs num: 32
# Connections num: 120
# Data Storage: ssd

# https://pgtune.leopard.in.ua

ALTER SYSTEM SET
 max_connections = '120';
ALTER SYSTEM SET
 shared_buffers = '8GB';
ALTER SYSTEM SET
 effective_cache_size = '24GB';
ALTER SYSTEM SET
 maintenance_work_mem = '2GB';
ALTER SYSTEM SET
 checkpoint_completion_target = '0.9';
ALTER SYSTEM SET
 wal_buffers = '16MB';
ALTER SYSTEM SET
 default_statistics_target = '500';
ALTER SYSTEM SET
 random_page_cost = '1.1';
ALTER SYSTEM SET
 effective_io_concurrency = '200';
ALTER SYSTEM SET
 work_mem = '32MB';
ALTER SYSTEM SET
 min_wal_size = '4GB';
ALTER SYSTEM SET
 max_wal_size = '16GB';
ALTER SYSTEM SET
 max_worker_processes = '32';
ALTER SYSTEM SET
 max_parallel_workers_per_gather = '16';
ALTER SYSTEM SET
 max_parallel_workers = '32';
ALTER SYSTEM SET
 max_parallel_maintenance_workers = '4';
