#!/bin/bash
if [ -z "$1" ] ; then
    PORT=7777
else
    PORT=$1
fi
jupyter notebook --no-browser --port=$PORT
echo "On your local machine, do:"
echo "ssh -N -L localhost:$PORT:localhost:$PORT $(whoami)@brux7.hep.brown.edu"