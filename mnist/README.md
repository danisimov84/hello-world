![mnist](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

# MNIST classifier. 

This article provides an example of a common task in ML. The dataset is the MNIST collection of images with numbers and known labels. The goal is to build a function that trains the algorithm to answer the drawn numbers on images. A convolutional neural network (Keras) is taken as a prediction algorithm. Provided code pulls dataset, trains the model, and estimates well-known losses (cross-entropy and accuracy).

1. Pull templates  
```
https://github.com/danisimov84/hello-world/tree/master/template/python3-ml
```
2. Create a service entity  
```
faas-cli new --lang python3-ml mnist --prefix="glowtools"  
Folder: python-service created.  
  ___                   _____           ____  
 / _ \ _ __   ___ _ __ |  ___|_ _  __ _/ ___|  
| | | | '_ \ / _ \ '_ \| |_ / _` |/ _` \___ \  
| |_| | |_) |  __/ | | |  _| (_| | (_| |___) |  
 \___/| .__/ \___|_| |_|_|  \__,_|\__,_|____/  
      |_|


Function created in folder: python-service  
Stack file written: mnist.yml  
```
3. Cat config 

cat mnist.yml
```
version: 1.0
provider:
  name: openfaas
  gateway: http://dev.kha.glow.tools
functions:
  mnist:
    lang: python3-ml
    handler: ./mnist
    image: danisimov84/mnist:latest
    environment:
         write_debug: true
         read_timeout: "1m"
         write_timeout: "1m"
         exec_timeout: "20m"
         batch_size: 128
```

4. Update the handle function  
cat mnist/handler.py 

```python
import json
import keras
import os
import warnings

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, InputLayer
from tensorflow.keras.datasets import mnist

warnings.filterwarnings("ignore")

def build_model(X_train, Y_train):
    model = Sequential()

    input_shape = list(X_train.shape[1:]) + [1]

    model.add(InputLayer(input_shape))
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())

    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.05))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.05))

    model.add(Dense(len(set(Y_train)), activation='softmax'))
    return model

def handle(req):
    print(req)
    try:
        args = json.loads(req)
    except Exception as e_message:
        print(e_message)
        args = {'epochs': 10, 'size': 1000}
    if not isinstance(args, dict):
        args = {'epochs': 10, 'size': 1000}
    size = int(args['size'])
    epochs = int(args['epochs'])

    if not 'descr' in args:
         args['descr'] = 'None'

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    (X_train, Y_train), (X_test, Y_test) = (X_train[:size], Y_train[:size]), (X_test[:size], Y_test[:size])

    model = build_model(X_train, Y_train)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

    batch_size = 128
    
    history = model.fit(X_train[:, :, :, None], Y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          validation_data=(X_test[:, :, :, None], Y_test))
    train_metric = model.evaluate(X_train[:, :, :, None], Y_train, verbose=0)
    test_metric = model.evaluate(X_test[:, :, :, None], Y_test, verbose=0)

    return json.dumps({'train': train_metric, 'test': test_metric, 'descr': args['descr']})

```

5. Build/Push/Deploy the service  
```
faas-cli up -f mnist.yml  

[0] > Building mnist.
Clearing temporary build folder: ./build/mnist/
Preparing: ./mnist/ build/mnist/function
Building: glowtools/mnist:latest with python3-ml template. Please wait..
Sending build context to Docker daemon  30.21kB
Step 1/18 : FROM openfaas/classic-watchdog:0.18.1 as watchdog
 ---> 94b5e0bef891
Step 2/18 : FROM glowtools/soccer:latest-2728ca6
 ---> 37929a9306bc
Step 3/18 : ARG ADDITIONAL_PACKAGE
 ---> Using cache
 ---> 10b278037908
Step 4/18 : COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
 ---> Using cache
 ---> b8acff615bd0
Step 5/18 : RUN chmod +x /usr/bin/fwatchdog
 ---> Using cache
 ---> 575a3bc6ee6d
Step 6/18 : WORKDIR /home/app/
 ---> Using cache
 ---> 1c0e6fc9f90a
Step 7/18 : COPY index.py           .
 ---> Using cache
 ---> 632560f30424
Step 8/18 : COPY requirements.txt   .
 ---> Using cache
 ---> df0e7b655308
Step 9/18 : WORKDIR /home/app/function/
 ---> Using cache
 ---> ad6b7c934964
Step 10/18 : COPY function/requirements.txt	.
 ---> Using cache
 ---> b00c957a5958
Step 11/18 : RUN pip install --upgrade pip
 ---> Using cache
 ---> 0f09d64ea578
Step 12/18 : RUN pip3 install --upgrade -r requirements.txt --target=/home/app/python
 ---> Using cache
 ---> 654f029a9871
Step 13/18 : WORKDIR /home/app/
 ---> Using cache
 ---> e92ac49357f9
Step 14/18 : COPY function           function
 ---> aeb6421ada75
Step 15/18 : ENV fprocess="python3 index.py"
 ---> Running in a4db20b55bfa
Removing intermediate container a4db20b55bfa
 ---> f6360625091f
Step 16/18 : EXPOSE 8080
 ---> Running in e4870229a758
Removing intermediate container e4870229a758
 ---> 516af0e002b2
Step 17/18 : HEALTHCHECK --interval=3s CMD [ -e /tmp/.lock ] || exit 1
 ---> Running in 056a9209e608
Removing intermediate container 056a9209e608
 ---> e4f061f2407f
Step 18/18 : CMD ["fwatchdog"]
 ---> Running in c5055df249a5
Removing intermediate container c5055df249a5
 ---> d4de0013a13b
Successfully built d4de0013a13b
Successfully tagged glowtools/mnist:latest
Image: glowtools/mnist:latest built.
[0] Building mnist done in 2.66s.
[0] Worker done.

Total build time: 2.66s

[0] Pushing mnist [glowtools/mnist:latest].
The push refers to repository [docker.io/glowtools/mnist]
32fa7892e2c4: Pushed 
3364389415db: Layer already exists 
a075645fb46f: Layer already exists 
dac9cf203e97: Layer already exists 
3fbfc980c918: Layer already exists 
a3b3a78b95a9: Layer already exists 
cdd87b2348e3: Layer already exists 
2acd0815f203: Layer already exists 
3e4fd673dbab: Layer already exists 
57ce1ec3bdd6: Layer already exists 
8691a5ff83e0: Layer already exists 
c323dd410be8: Layer already exists 
879bde267dbe: Layer already exists 
fcc6ec8ee1e0: Layer already exists 
2f986280fdef: Layer already exists 
366f440d5e25: Layer already exists 
7622c0970272: Layer already exists 
e52f1b803c45: Layer already exists 
befce55f84c6: Layer already exists 
d870b0f92c14: Layer already exists 
2c86cf4cd527: Layer already exists 
1cccc1f74e01: Layer already exists 
20119e4b0fc9: Layer already exists 
751ae3b79e0a: Layer already exists 
133ee43735a0: Layer already exists 
97c83918ca41: Layer already exists 
6b87768f66a4: Layer already exists 
808fd332a58a: Layer already exists 
b16af11cbf29: Layer already exists 
37b9a4b22186: Layer already exists 
e0b3afb09dc3: Layer already exists 
6c01b5a53aac: Layer already exists 
2c6ac8e5063e: Layer already exists 
cc967c529ced: Layer already exists 
latest: digest: sha256:f8f81a03057536267dbfe6f95a8fbb3501fc27cf5ce43fcff054ca42101403a6 size: 7657
[0] Pushing mnist [glowtools/mnist:latest] done.
[0] Worker done.

Deploying: mnist.
WARNING! Communication is not secure, please consider using HTTPS. Letsencrypt.org offers free SSL/TLS certificates.

Deployed. 202 Accepted.
URL: http://dev.kie.glow.tools/function/mnist
```
6. Invoke the service  
curl http://dev.kie.glow.tools/function/mnist -d '{"size": 10000, "epochs": 10}'  

```
{"train": [0.0058157178573310375, 0.9980000257492065], "test": [0.08711764216423035, 0.9797999858856201], "descr": "None"}

```

