## Pensieve

Pensieve uses SOTA reinforcement learning methods to implement Adaptive Video Streaming.

The code is based on the paper [Neural Adaptive Video Streaming with Pensieve (SIGCOMM '17)](http://web.mit.edu/pensieve/) 

We implement a PyTorch based neural method for pensieve, and you can find a original tensorflow version at https://github.com/hongzimao/pensieve

### Prerequisites

- Ubuntu 16.04, PyTorch1.1.0



### Training

- To train a new model, put training data in `pytorch_sim/cooked_traces` and testing data in `pytorch_sim/cooked_test_traces`, then in `pytoch_sim/` run `python get_video_sizes.py` and then run

```
python train.py
```

More details are to be found in `pytorch_sim/README.md`



### Testing

- To test the trained model in simulated environment, first copy over the model to `test/models` and modify the `NN_MODEL` field of `test/torch_rl_test.py` , and then in `test/` run `python get_video_sizes.py` and then run 

```
python torch_rl_test.py
```

Similar testing can be performed for buffer-based approach (`bb.py`), mpc (`mpc.py`) and offline-optimal (`dp.cc`) in simulations and also the original method `rl_no_training.py` . More details can be found in `test/README.md`.

