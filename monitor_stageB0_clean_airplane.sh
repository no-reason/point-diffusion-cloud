#!/bin/bash
cd /data/personal_data/zyy/point-diffusion-cloud
LOG_FILE=$(ls -t nohup_logs/stageB0_clean_airplane/train_clean_airplane_*.log | head -1)

echo "Monitoring log: $LOG_FILE"
echo "--- GPU Status ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | grep -E "1, NVIDIA|index"
echo "------------------"
echo "Latest Train Progress:"
grep "Loss" $LOG_FILE | tail -n 3
echo "------------------"
echo "Latest Checkpoint Saved:"
ls -lh logs_gen/GEN_*_Clean_Airplane_From_Scratch_KL001/*.pt 2>/dev/null | tail -n 1
