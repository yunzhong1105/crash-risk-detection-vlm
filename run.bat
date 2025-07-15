@echo off
echo Running freeze text_model and strategy3 training on freeway epoch=4...
python .\\vision\\finetuning\\smolvlm2_video_FT_strategy3.py --dataset freeway --epoch 4

echo.
echo Running freeze text_model and strategy3 training on freeway epoch=3...
python .\\vision\\finetuning\\smolvlm2_video_FT_strategy3.py --dataset freeway --epoch 3

echo.
echo Running freeze text_model and strategy3 training on freeway epoch=2...
python .\\vision\\finetuning\\smolvlm2_video_FT_strategy3.py --dataset freeway --epoch 2

echo.
echo Running freeze text_model and strategy3 training on freeway epoch=1...
python .\\vision\\finetuning\\smolvlm2_video_FT_strategy3.py --dataset freeway --epoch 1

echo.
echo Running freeze text_model and strategy3 training on road epoch=1...
python .\\vision\\finetuning\\smolvlm2_video_FT_strategy3.py --dataset road --epoch 1

echo.
echo Running freeze text_model and strategy3 training on road epoch=2...
python .\\vision\\finetuning\\smolvlm2_video_FT_strategy3.py --dataset road --epoch 2

echo.
echo Running freeze text_model and strategy3 inference on freeway epoch=4...
python .\\vision\\finetuning\\custom_infer.py --dataset freeway --epoch 4

echo.
echo Running freeze text_model and strategy3 inference on freeway epoch=3...
python .\\vision\\finetuning\\custom_infer.py --dataset freeway --epoch 3

echo.
echo Running freeze text_model and strategy3 inference on freeway epoch=2...
python .\\vision\\finetuning\\custom_infer.py --dataset freeway --epoch 2

echo.
echo Running freeze text_model and strategy3 inference on freeway epoch=1...
python .\\vision\\finetuning\\custom_infer.py --dataset freeway --epoch 1

echo.
echo Running freeze text_model and strategy3 inference on road epoch=2...
python .\\vision\\finetuning\\custom_infer.py --dataset road --epoch 2

echo.
echo Running freeze text_model and strategy3 inference on road epoch=1...
python .\\vision\\finetuning\\custom_infer.py --dataset road --epoch 1

echo.
echo All done.
pause
