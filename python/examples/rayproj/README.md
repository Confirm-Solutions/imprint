# Dockerised Ray cluster for testing Ray + kevlar

## Build Docker image

```
cd ./python/examples/rayproj
docker build -t rayproj/kevlar .
```

## Start Ray Head

Start a head docker instance and expose ports for management and monitoring on the Docker server.

Note that we give the Docker head instance 2 CPUs, but configure the Ray head to provide 0 CPUs. This dedicates the 2 Docker CPUs to the Ray head node.

```
docker run -d -p 8265:8265 -p 10001:10001 --cpus=2 rayproj/kevlar --head --dashboard-host=0.0.0.0 --num-cpus=0
```

## Start a lot of Ray Workers

Next, decide on the number of Ray worker nodes and the number of vCPUs given to each of them. 

We then find the IP of the Ray head node started above so we can provide it to each worker.

Note that `--num-cpus=1` below tells Ray there's only one CPU per `$NUM_CPUS_WORKER` vCPU container. Pass `n_threads` to the Kevlar driver to tell it to start `$NUM_CPUS_WORKER` threads per Ray node.

```
export NUM_RAY_WORKERS=16
export NUM_CPUS_WORKER=4
export RAY_CONTAINER_ID=`docker ps | awk '$2~"rayproj/kevlar" {print $1}'`
export RAY_HEAD_IP=`docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $RAY_CONTAINER_ID`
for N in `seq 1 $NUM_RAY_WORKERS`; do docker run -d --cpus=$NUM_CPUS_WORKER rayproj/kevlar --address="$RAY_HEAD_IP:6379" --redis-password='5241590000000000' --num-cpus=1; done
```

## Stop the entire Docker cluster

This will find all `rayproj/kevlar` containers including the head node, and stop them. Use with care!

```
for C in `docker ps | awk '$2~"rayproj/kevlar" {print $1}'`; do   docker stop $C; done
```