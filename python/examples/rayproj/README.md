# Build Docker image

`cd ./python/examples/rayproj; docker build -t rayproj/kevlar .`

# Ray Head

We start a head docker instance and expose ports for management and monitoring on the Docker server:

`docker run -p 8265:8265 -p 10001:10001 --cpus=2 --interactive --tty --entrypoint /bin/bash rayproj/kevlar`

Then inside Docker instance we start the head node and bind the dashboard to the external interface so it is accessible outside Docker:
  
`ray start --head --dashboard-host=0.0.0.0 --num-cpus=0`

# Ray Worker

`docker run --cpus=8 --interactive --tty --entrypoint /bin/bash rayproj/kevlar`
  
Then inside Docker instance:

`ray start --redis-password="5241590000000000" --num-cpus=1 --address=<IP-of-HEAD-node>:6379`

Note that the Docker container is configured with 8 cores, but the ray node is configured with 1 core, as the Kevlar driver should spin up 8 threads.