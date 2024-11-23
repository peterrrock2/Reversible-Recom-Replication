#!/bin/bash


julia aux_script_files/setup.jl

python3 -m venv .venv 

source .venv/bin/activate
pip3 install -r py_requirements.txt


cargo install binary-ensemble
cargo install --git https://github.com/peterrrock2/msms_parser.git
cargo install --path ./Ben_Tally
