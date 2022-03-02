# Build Docker image

`cd ./python/examples/rayproj; docker build .`

# Ray Head

`docker run -p 8265:8265 -p 10001:10001 --cpus=2 --interactive --tty --entrypoint /bin/bash <DOCKER-IMAGE-ID>`

Then inside Docker instance:
  
`ray start --head --dashboard-host=0.0.0.0 --num-cpus=0`

# Ray Worker

`docker run --cpus=8 --interactive --tty --entrypoint /bin/bash <DOCKER-IMAGE-ID>`
  
Then inside Docker instance:

`ray start --redis-password="5241590000000000" --num-cpus=8 --address="<IP-of-HEAD-node>:6379"`
