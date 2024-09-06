#!/bin/bash

job_id_inf=$(sacct -X --name=tgi_server.sh --format=JobID,JobName --noheader | sort | tail -n 1 | awk '{print $1}')
IPD=$(grep -oP  'IP ADDRESS: \K([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})' tgi-${job_id_inf}.out)
time curl ${IPD}:8080/generate -X POST -d '{"inputs":"What do you know about me?","parameters":{"max_new_tokens":32}}' -H 'Content-Type: application/json'
grep -oE 'ssh -p 8822 .*:8080' tgi-${job_id_inf}.out
