python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 24}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 36}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 48}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/VAR_model"

python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "Covid-19.csv" --strategy-args '{"horizon": 60}' --model-name "self_impl.VAR_model" --model-hyper-params '{}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "Covid-19/VAR_model"

