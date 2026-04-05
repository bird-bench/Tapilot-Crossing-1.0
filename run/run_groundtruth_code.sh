#!/bin/bash  
  
DATA_PATH="TAPILOT-DATA-PATH"
OUT_PATH="OUTPUT-FILE-PATH"  

success_count=0  
failed_count=0
failed_files=()  
  
for dir in $(find $DATA_PATH -type d)  
do  
    if [ -f "$dir/ref_code_hist.py" ]; then  
        original_dir=$(pwd)  
  
        cd "$dir"  
  
        python3 "./ref_code_hist.py"  
  
        if [ $? -eq 0 ]; then  
            success_count=$((success_count + 1))  
        else  
            failed_count=$((failed_count + 1))  
            failed_files+=("$dir/ref_code_hist.py")  
        fi  
  
        cd "$original_dir"  
    fi  
done  
  
echo "# successful ref_code_hist.py: $success_count" > $OUT_PATH  
echo "# Failed ref_code_hist.py: $failed_count" >> $OUT_PATH  
  