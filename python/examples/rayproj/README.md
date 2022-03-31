# Dockerised Ray cluster for testing Ray + kevlar

## Build Docker image

```
cd ./python/examples/rayproj
docker build -t rayproj/kevlar .
```

## Start the Ray head

Start a head docker instance named `ray_head` and expose ports for management and monitoring on the Docker server.

Note that we give the Docker head instance 2 vCPUs, but configure the Ray head to provide 0 CPUs. This dedicates the 2 Docker vCPUs to the Ray head node.

```
docker run -d --rm --shm-size=20gb -p 8265:8265 -p 10001:10001 --cpus=2 --name=ray_head rayproj/kevlar --head --dashboard-host=0.0.0.0 --num-cpus=0
```

## Start a number of Ray workers

Next, decide on the number of Ray worker nodes and the number of vCPUs allocated to each of them. 

Note that `--num-cpus=1` below tells Ray there's only one CPU resource per `$NUM_CPUS_WORKER` vCPU container. We also set each worker's container CPU share to 50% to ensure the head node container isn't starved of CPU.

Correspondingly, you need to then set `n_threads` in the Kevlar driver to tell it to use `$NUM_CPUS_WORKER` threads per Ray node.

```
export NUM_RAY_WORKERS=15
export NUM_CPUS_WORKER=4
export RAY_HEAD_IP=`docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ray_head`
for N in `seq -w 1 $NUM_RAY_WORKERS`; do docker run --rm -d --shm-size=20gb --cpus=$NUM_CPUS_WORKER -c 512 --name="ray_worker$N" rayproj/kevlar --address="$RAY_HEAD_IP:6379" --redis-password='5241590000000000' --num-cpus=1; done
```

## Stop the entire Docker cluster

This will find all containers named `ray_*`, including the head node, and stop them all. Since started all the containers with `--rm` stopping them will also remove them.

```
for C in `docker ps | awk '$NF~/^ray_[head|worker]/ {print $1}'`; do docker stop $C; done
```