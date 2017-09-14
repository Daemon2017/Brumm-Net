Using TensorFlow backend.
Number of X and Y in train is the same. Work continues!
Number of X and Y in test is the same. Work continues!
Start training? (y or n): y
Building model...
Model ready!
Training...
Found 367 images belonging to 1 classes.
Found 367 images belonging to 1 classes.
Found 233 images belonging to 1 classes.
Found 233 images belonging to 1 classes.
Traceback (most recent call last):
  File "train.py", line 179, in <module>
    train()
  File "train.py", line 131, in train
    callbacks=[tbCallBack, WeightsSaver(model, 1)])
  File "/usr/local/lib/python2.7/dist-packages/keras/legacy/interfaces.py", line 87, in wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/keras/engine/training.py", line 1936, in fit_generator
    raise ValueError('When using a generator for validation data, '
ValueError: When using a generator for validation data, you must specify a value for `validation_steps`.
