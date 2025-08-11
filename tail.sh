clear

LOG_DATE=/home/juhyung/data/longitudinal/log/log_recon_2025_07_13
echo "[INFO] Tail logs in: $LOG_DATE"

find $LOG_DATE -type f -name "*.log" -exec tail -n 100 -f {} +
