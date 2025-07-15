@echo off
echo Running inference on dataset1...
python .\\vision\\finetuning\\inference.py --data-dir freeway --model-path "C:\\Python_workspace\\TAISC\\code\\smollm-main\\SmolVLM2-500M-Video-Instruct-taisc(freeway-vlm-5epoch)\\checkpoint-720"

echo.
echo Finished dataset1, starting dataset2...
python .\\vision\\finetuning\\inference.py --data-dir road  --model-path "C:\\Python_workspace\\TAISC\\code\\smollm-main\\SmolVLM2-500M-Video-Instruct-taisc(road-vlm-5epoch)\\checkpoint-715"

echo.
echo All done.
pause
