#!/bin/bash

# Get the name of the python script to execute
script_name=$1

# Install the local package via poetry
poetry install --no-root

# Execute the script inside the base service
docker-compose exec base python tests/$script_name
