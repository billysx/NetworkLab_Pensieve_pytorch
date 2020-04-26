Make sure actual video files are stored in `video_server/video[1-6]`, then run

```
python get_video_sizes
```

Put training data in `pytorch_sim/cooked_traces` and testing data in `pytorch_sim/cooked_test_traces` (need to create folders). The trace format for simulation is `[time_stamp (sec), throughput (Mbit/sec)]`. The training traces are already separated and put into `train_sim_traces` and `test_sim_traces` . Or you can download yourself the training traces from https://www.dropbox.com/sh/ss0zs1lc4cklu3u/AAB-8WC3cHD4PTtYT0E4M19Ja?dl=0. More details of data preparation can be found in `traces/`.



To test a trained model, run 
```
python torch_rl_test.py
```

Results will be saved in `test/results/`. 

Similarly, one can also run `bb.py` for buffer-based simulation, `mpc.py` for robustMPC simulation, and `dp.cc` for offline optimal simulation.

To view the results, modify `SCHEMES` in `plot_results.py` (it checks the file name of the log and matches to the corresponding ABR algorithm), then run 
```
python plot_results.py
```

The result images are saved in `test/picres`

