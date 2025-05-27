#!/bin/bash
echo "VRAM Watchdog script active. Monitoring..."
while true; do
    nvidia-smi
    sleep 60
done
