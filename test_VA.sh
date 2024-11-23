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


run_with_spinner bash -c "julia ./aux_script_files/run_VA.jl | msms_parser -r loc_prec -s loc_prec -g ./JSON/VA_precincts.json -o ./outputs/VA_100k.jsonl -w"
ben -m encode ./outputs/VA_100k.jsonl -v -w
ben-tally -g ./JSON/VA_precincts.json -b ./outputs/VA_100k.jsonl.ben
ben-tally -g ./JSON/VA_precincts.json -b ./outputs/VA_100k.jsonl.ben -m tally-keys --keys G16DPRS G16RPRS

