#!/bin/bash
inotifywait -m -r -e create,delete ../pretrain/2/ |
    while read path action file; do
        file_count=$(ls ../pretrain/2/ | grep "loss" | wc -l)
        if [ $file_count -gt 5 ]; then
            oldest_file=$(ls -tr ../pretrain/2/ | grep "loss" | head -1)
            rm "../pretrain/2/$oldest_file"
            echo "Deleted file: $oldest_file"
        fi
    done
