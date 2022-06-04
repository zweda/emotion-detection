# Emotion detection in context
Source code for Text Analysis and Recognition course project.

## Environment setup
Source code provides `envirnoment.yml` for quick environment setup with `conda`. 
Position `console` in root project diroectory then run following commands:
    
    conda ${env-name} create -f environment.yml
    conda activate ${env-name}

Note that we are using `python 3.10` with `tensorflow 2.8` which will couse errors within `PyCharm IDE` but the program should run 
nonetheless. If you want to mitigate these IDE errors use as following:

    import tensorflow as tf
    ...

    tf.keras.layers.Embedding(...)
    
    ...