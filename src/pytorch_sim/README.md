Make sure actual video files are stored in `video_server/video[1-6]`, then run
```
python get_video_sizes.py
```

Put training data in `pytorch_sim/cooked_traces` and testing data in `pytorch_sim/cooked_test_traces` (need to create folders). The trace format for simulation is `[time_stamp (sec), throughput (Mbit/sec)]`. The training traces are already separated and put into `train_sim_traces` and `test_sim_traces` . Or you can download yourself the training traces from https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0. More details of data preparation can be found in `traces/`.

Different from the original code base, we use PyTorch to implement A3C and you can train a model by running:

```
python train.py
```

Trained model will be saved in `pytorch_sim/trained_models/`.

Besides, you can also choose the alternative model with LSTM by setting islstm=1, implementing by running:

```bash
python train.py --islstm 1 --saved_path trained_lstm
```

