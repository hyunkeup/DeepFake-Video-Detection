#!/bin/bash
nohup python3 -u ./extract_audios.py > extract_audios.log &
nohup python3 -u ./extract_videos.py > extract_videos.log &