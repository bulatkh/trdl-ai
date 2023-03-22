To set-up and activate conda environment:

``` 
$ conda env create -f environment.yml 
$ conda activate trdl-ai
```

To run test inference you need to move `chunk_0.txt` (or the one you want to test) to `test_chunks/` and `model_config.yml`, `model_weights.ckpt` and `threshold.yml` to `model_assests/`:
```
$ python inference.py test_chunks/chunk_0.txt  model_assets/model_config.yml model_assets/model_weights.ckpt  model_assets/threshold.yml 
```