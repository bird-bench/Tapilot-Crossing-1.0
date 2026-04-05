#!/bin/bash  
  
DIR_PATH="TAPILOT-DATA-PATH"
FIL_NAME="YOUR-CODE-GEN-PYTHON-FILE-NAME"
  
success_count=0  
failed_count=0  
  
failed_files=()  
  
for dir in $(find $DIR_PATH -type d)  
do  
    if [ -f "$dir/$FIL_NAME" ]; then  
        original_dir=$(pwd)  

        cd "$dir" 
        rm -f pred_result/* 

        timeout 5m python3 "./$FIL_NAME"  

        if [ $? -eq 0 ]; then  
            success_count=$((success_count + 1))  
        else  
            failed_count=$((failed_count + 1))  
            failed_files+=("$dir/$FIL_NAME")  
        fi  

        cd "$original_dir"  
    fi  
done  
  
