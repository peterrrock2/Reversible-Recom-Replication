#!/bin/bash

run_with_spinner() {
    local spinner="|/-\\"
    local pid
    local i=0

    "$@" & pid=$!

    while kill -0 $pid 2> /dev/null; do
        printf "\rRunning... %c" "${spinner:i++%${#spinner}:1}"
        sleep 0.25
    done

    printf "\r                        \r"
    wait $pid
}


run_with_spinner bash -c "julia run_7.jl | msms_parser -r precinct -s precinct -g ./JSON/7x7.json -o ./outputs/7x7_100k.jsonl -w"
ben -m encode ./outputs/7x7_100k.jsonl -v -w
ben-tally -g ./JSON/7x7.json -b ./outputs/7x7_100k.jsonl.ben

source .venv/bin/activate

python read_cuts_7.py
