#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 script_name.py [directory_path]"
    exit 1
fi

script_name="$1"
if [ "$#" -eq 2 ]; then
    directory_path="$2"
else
    directory_path="src/bipartite_embeddings/"
fi

docker-compose exec base bash -c "cd $directory_path && exec python $script_name"