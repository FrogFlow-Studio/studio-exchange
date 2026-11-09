#!/usr/bin/env bash
# Одна команда: ./run.sh [скан.stl] [заметки]
cd "$(dirname "$0")"
SCAN="${1:-samples/bracket_scan.stl}"
NOTES="${2:-угол плиты отколот, отверстие в нём потеряно}"
python3 demo.py "$SCAN" --notes "$NOTES"
