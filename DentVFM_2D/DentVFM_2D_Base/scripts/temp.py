import os
import subprocess

cmd = "baizectl job ls -t PYTORCH | awk '$3 == \"SUCCEEDED\"' | wc -l"

result = subprocess.check_output(cmd, shell=True)
count = int(result.decode().strip())

print("文件/目录数量：", count)