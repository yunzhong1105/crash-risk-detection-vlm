@echo off
echo Running freeze text_model and full.py training on freeway epoch=2...
python .\\vision\\finetuning\\smolvlm2_video_FT_full.py --dataset freeway --epoch 2 --freeze-text True

echo.
echo Inferencing freeze text_model and full.py training on freeway epoch=2...
python vision\finetuning\custom_infer.py --dataset freeway --epoch 2 --model-path ./your_checkpoint_path

echo.
echo Running freeze text_model and full.py training on road epoch=2...
python .\\vision\\finetuning\\smolvlm2_video_FT_full.py --dataset road --epoch 2 --freeze-text True

echo.
echo Inferencing freeze text_model and full.py training on freeway epoch=2...
python vision\finetuning\custom_infer.py --dataset road --epoch 2 --model-path ./your_checkpoint_path

echo.
echo All done.
pause
