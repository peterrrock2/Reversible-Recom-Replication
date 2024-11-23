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


run_with_spinner bash -c "julia ./aux_script_files/run_4.jl | msms_parser -r precinct -s precinct -g ./JSON/4x4.json -o ./outputs/4x4_100k.jsonl -w"
ben -m encode ./outputs/4x4_100k.jsonl -v -w
ben-tally -g ./JSON/4x4.json -b ./outputs/4x4_100k.jsonl.ben

source .venv/bin/activate

python ./aux_script_files/read_cuts_4.py
