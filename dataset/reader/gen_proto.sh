#!/bin/bash
python3 -m grpc_tools.protoc -I=./ssl_protobuff --python_out=./gen --pyi_out=./gen --grpc_python_out=./gen ./ssl_protobuff/*.proto
