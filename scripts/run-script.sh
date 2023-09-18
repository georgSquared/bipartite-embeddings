#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 script_name.py [directory_path] [python_script_args...]"
    exit 1
fi

script_name="$1"
shift

if [ "$#" -gt 0 ] && [[ ! $1 =~ ^- ]]; then
    directory_path="$1"
    shift
else
    directory_path="src/bipartite_embeddings/"
fi

docker compose exec base bash -c "cd $directory_path && exec python $script_name $*"
