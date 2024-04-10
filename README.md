# Rock Paper Scissor Image Classification


```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
tf.random.set_seed(42)
```

    /Users/kullya/josh/DSProjects/Rock-Paper-Scissors/venv-metal/lib/python3.10/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm



```python
data, info = tfds.load("rock_paper_scissors", with_info=True, as_supervised=True)
train_data = data["train"]
test_data = data["test"]
```

    2024-04-10 14:19:45.341551: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M1
    2024-04-10 14:19:45.341609: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
    2024-04-10 14:19:45.341619: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
    2024-04-10 14:19:45.342383: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
    2024-04-10 14:19:45.342487: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)


Let us visualise how does our images loook.! We will also know which image as what label


```python
# get some examples
fig = tfds.show_examples(train_data, info, rows=2, cols=6)
```

    2024-04-10 14:19:46.079777: W tensorflow/core/kernels/data/cache_dataset_ops.cc:858] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2024-04-10 14:19:46.151373: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence



    
![png](README_files/README_4_1.png)
    


This shows us the numeric values for labels.  
0 - Rock  
1 - Paper  
2 - Scissor  

---

Next step is to confirm the size of the dataset and make sure we have sufficient data available. We will take care of the following considerations.  
1. We have suffficient traininy and testing data.
2. What classes do we have.
3. Each class has equivalent amount of data and our dataset is not imbalanced. 
4. Understanding the size of each image.

Size of the dataset


```python
print("Train set size:", info.splits["train"].num_examples)
print("Test set size:", info.splits["test"].num_examples)
```

    Train set size: 2520
    Test set size: 372



```python
print("Classes:", info.features["label"].names)
```

    Classes: ['rock', 'paper', 'scissors']


Going with the 3rd consideration, we see how many data points we have per class.


```python
# count occurrences of each class
train_counts = [0, 0, 0]
test_counts = [0, 0, 0]
for image, label in train_data:
    train_counts[label] += 1
for image, label in test_data:
    test_counts[label] += 1

print("Train counts:", train_counts)
print("Test counts:", test_counts)
```

    2024-04-10 14:19:50.214223: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


    Train counts: [840, 840, 840]
    Test counts: [124, 124, 124]


    2024-04-10 14:19:50.461012: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


This seems to be an excellent dataset as we have equal data points for each class in the training(840) as well as the testing(124) data. Confirm the shapes of the images.


```python
input_shape = info.features["image"].shape
print("Input shape:", input_shape)
```

    Input shape: (300, 300, 3)


It should always be a good practice to have a validation set while training. This has the following purposes.
1. Prevent overfitting. We will implement multi metric training, early stopping and checkpointing in the future.
2. We cant use the testing data as validation data which would basically mean our model knows the images in the test set and it defeats the purpose of evaluating the performance. Our performance metrics will give all green signals on the basis of just mentioning what it already knows. 
3. Our early stopping and benchmarking will use validation set to identify where other performance has improved or not. So this validation set acts as a pseudo unknown data while training to tell us whether other performance is improving or not.  

So below we have split the data into training and testing. 

## Data Augmentation
Even though we said we have sufficient data, let us add some random variation. For that, we will
1. Flip the image randomly with a 0.5 percent chance right to left and top to bottom. So this way we can randomly get images like
- Horizontally flipped only
- Vertically flipped only
- both horizontally and vertically flipped
2. We will not replace this with the original data as we need both of them. So let us concat this to the original data. We will do this for the validation set as well to measure performance while trinaing. 


```python
train_data = train_data.shuffle(train_data.cardinality(), seed=42).cache()

val_size = 0.1
val_data = train_data.take(int(val_size * info.splits["train"].num_examples))
train_data = train_data.skip(int(val_size * info.splits["train"].num_examples))

# augment the data
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label

augmented_train_data = train_data.map(augment)
augmented_val_data = val_data.map(augment)
```

Now, we cant really map what was the original image and what was it augumented to. Reason being these images are shuffled and we havent cached them. Shuffling is very much necessary so that we dont have all the rocks together and so on. **take(1) picks one random element from the dataset, not the 0th element**. So when we do `take(1)` from respective datasets, they are completely random and not necessarily what is mapped to its augmentation.   

So to demonstrate on visualing these images, let us choose 10 images and augment them and mao them to their augumentation


```python
random_images = train_data.take(5)

# in the 1st row, print the original image and in the 2nd row, print the augmented image and set heigh of the image to 5
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, (image, label) in enumerate(random_images):
    axes[0, i].imshow(image)
    axes[0, i].set_title(f"Original\n{info.features['label'].int2str(label)}")
    axes[0, i].axis("off")
    axes[1, i].set_title("Agumented")
    axes[1, i].imshow(augment(image, label)[0])
    axes[1, i].axis("off")
```

    2024-04-10 14:19:51.222991: W tensorflow/core/kernels/data/cache_dataset_ops.cc:858] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2024-04-10 14:19:51.248047: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence



    
![png](README_files/README_16_1.png)
    



```python
# concatenate the augmented data with the original data
train_data = train_data.concatenate(augmented_train_data)
val_data = val_data.concatenate(augmented_val_data)
```


```python
# total size of the training and validation data
print("Train size:", train_data.cardinality().numpy())
print("Val size:", val_data.cardinality().numpy())

batch_size = 32

train_data = train_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
val_data = val_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

test_data = test_data.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

for img, label in train_data.take(1):
    print("Train data shape:", img.shape)
    print("Train label shape:", label.shape)
    
for img, label in val_data.take(1):
    print("Val data shape:", img.shape)
    print("Val label shape:", label.shape)
```

    Train size: 4536
    Val size: 504
    Train data shape: (32, 300, 300, 3)
    Train label shape: (32,)
    Val data shape: (32, 300, 300, 3)
    Val label shape: (32,)


    2024-04-10 14:19:51.914267: W tensorflow/core/kernels/data/cache_dataset_ops.cc:858] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2024-04-10 14:19:51.914433: W tensorflow/core/kernels/data/cache_dataset_ops.cc:858] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2024-04-10 14:19:51.914872: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
    2024-04-10 14:19:51.988909: W tensorflow/core/kernels/data/cache_dataset_ops.cc:858] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2024-04-10 14:19:51.989085: W tensorflow/core/kernels/data/cache_dataset_ops.cc:858] The calling iterator did not fully read the dataset being cached. In order to avoid unexpected truncation of the dataset, the partially cached contents of the dataset  will be discarded. This can happen if you have an input pipeline similar to `dataset.cache().take(k).repeat()`. You should use `dataset.take(k).cache().repeat()` instead.
    2024-04-10 14:19:51.989493: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


We originally had 2520 data points. We took 0.9 part of it for training which was 2268. Then we concatenated it with augumented images to double its size making it 4536. Likewise for the validation data. 

Then we batched our data. Using 32 was a trial and error choice. Since I am using M1 metal, which is the GPU for macos, I did not need significantly long time frame per batch and so I could run with a batch size of 32.

We store the `num_classes` in a variable to use it in the last layer.


```python
num_classes = info.features["label"].num_classes
```

## First model with hardcoded parameters

Even though I have hardcoded the values here, I had derived them from hyper parameter tuning. 

1. Using the input layer which does not compute anything other than properly templating the input shapes. 
2. Used a `Resizing` layer which later on became redundant after I added the `Input` layer. This layer was to ensure that all the images get converted to our actual shapes that we trained on. That is (300, 300).
3. We `Rescale` all the rgb values to decimals from 0-1 to avoid the problem of exploading gradients. Where our values grow larger and larger with time. Larger values will add unwanted biases for and also multiplying larger values takes more time.
4. We had a `RandomFlip` and `RandomRotation` for augumentation but this raised key challenges. 
  - I asked myself, these layers would remain in place inside the network in Production. When we ll have a test image, it will get randomly rotated and flipped. Why would we want that?
  - We cant save models with randomness. The model needs to know what is going to happen. So with additionally need to bundle this as a module and bind the config as to retain the custom variables and functions. 
4. We keep following `Conv2D` with `MaxPool`. This is as per what we read everywhere, to follow a `Conv2D` witha a `MaxPool` but I got very curious about this and thought of why not to have `Conv2D`, another `Conv2D` and then a `MaxPool`. This, logically should help us make more and better filters as the 2nd `Conv2D` has gotten the original values without any loss of information after a `MaxPool`.
5. We flatten the images and add a single `Dense` followed by another `Dense` layer with the exact number of classes we want. That is 3. We use `Softmax` activation function because we want probabilities that sum to 1. This ensures that our predictions are in a range and provide a clear interpretation.
6. For all other layers, we use `ReLU` for computation effieciency, addressing vanishing gradient and to offcourse add non linearity.

**We obviously ended up with an overfitted model as our data is very clean and dont really have significant noise. To tackle that, we used classic `Dropout` and `L2 Regularisation.`**. 
- `Dropout` percentage and `L2 Regularisation` parameters were derived from **Grid Search**.
- We use **Categorical Cross Entropy** for classification because our label belongs to only a single class and we need to penalise longer deviations. But we need to one hot encode our labels which we did not need to really. So we ended using `sparse_categorical_crossentropy` which is same as Categorical Cross Entropy without the need of one hot encoding.


```python
tf.keras.backend.clear_session()

firstModel = tf.keras.Sequential([
  # input
  layers.Input(input_shape),

  # resize and rescale
  layers.Resizing(input_shape[0], input_shape[1]),
  layers.Rescaling(1./255),

  # augmentations
  # layers.RandomFlip("horizontal_and_vertical", seed = 42),
  # layers.RandomRotation(0.2),

  # convolutions
  layers.Conv2D(32, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Flatten(),
  layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.Dense(num_classes, activation="softmax")
])

firstModel.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

firstModel.summary()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ resizing (<span style="color: #0087ff; text-decoration-color: #0087ff">Resizing</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">300</span>, <span style="color: #00af00; text-decoration-color: #00af00">300</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ rescaling (<span style="color: #0087ff; text-decoration-color: #0087ff">Rescaling</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">300</span>, <span style="color: #00af00; text-decoration-color: #00af00">300</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)    │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">298</span>, <span style="color: #00af00; text-decoration-color: #00af00">298</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │           <span style="color: #00af00; text-decoration-color: #00af00">896</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">149</span>, <span style="color: #00af00; text-decoration-color: #00af00">149</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)   │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">147</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)  │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">73</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)     │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">341056</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │    <span style="color: #00af00; text-decoration-color: #00af00">21,827,648</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)              │           <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,847,235</span> (83.34 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">21,847,235</span> (83.34 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



We fit the model below but end up with crucial observations while development.


```python
history = firstModel.fit(train_data, validation_data=val_data, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)], verbose = 1)
```

    Epoch 1/20


    2024-04-10 14:19:53.503174: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.


    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m91s[0m 610ms/step - accuracy: 0.6948 - loss: 3.7373 - val_accuracy: 0.9425 - val_loss: 1.1614
    Epoch 2/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m79s[0m 553ms/step - accuracy: 0.9540 - loss: 0.8790 - val_accuracy: 0.9663 - val_loss: 0.8830
    Epoch 3/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m79s[0m 555ms/step - accuracy: 0.9656 - loss: 0.7220 - val_accuracy: 0.9643 - val_loss: 1.0694
    Epoch 4/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m77s[0m 544ms/step - accuracy: 0.9674 - loss: 0.7568 - val_accuracy: 0.9563 - val_loss: 1.0154


At times, the early stopping triggers but we still had a vveerryy low performance. But we know that the model has randomisation due to the Dropout layer. (Opps, did I mention before that we cant save a model with randomisation but Dropout is actually random? Right! But Dropout layer goes inactive during evaluation and prediction so the network by itself has unknowngily handled that randomisation). With that in mind, I knew that the model can actually perform differently if I rerun it again. 

So without recompiling, I just ran `fit` but learnt an excellent lesson! The model resumes with its existing weights, i.e. from the point where the early stopping was triggered. So this Early Stopping can practically raise false alarms!! I have handled this later. 


```python
firstModel.evaluate(test_data)
```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 173ms/step - accuracy: 0.5284 - loss: 2.1894





    [2.261378049850464, 0.5215053558349609]



This is not favorable. 50% accuracy is same as just flipping a coin and mapping it to something. We need to do better. 


But, let us plot the performance over time and understand if our model was actually improving over time. We can do it by manually inspecting the logs above but images are easier to interpret!

### Plot the history


```python
def plotHistory(history):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(10, 3))
  plt.subplot(1, 2, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()
```


```python
plotHistory(history)
```


    
![png](README_files/README_33_0.png)
    


Performance did improve actually so let us just backup this model.


```python
# save the firstModel
tf.saved_model.save(firstModel, "models/firstModel")
```

    INFO:tensorflow:Assets written to: models/firstModel/assets


    INFO:tensorflow:Assets written to: models/firstModel/assets


> Early Stopping can raise false alarms.

An option is to use multiple parameters to evaluate model performance instead of just the accuracy. So let us use accuracy and F1 score. We use F1 because we have to reduce False Positives and False Negatives with equal importance. Else, we could have focused on F0.5 or F2 score.  

Since TF doesnt have an inbuilt way to monitor multiple parameters, let us build a custom EarlyStopping class. For this, we will need to one hot encode our labels. But, cant we just use Sparse Categorical Cross Entropy. Not here really because we have softmax in the last layer which outputs probabilities per class. To calculate `F1 Score`, we will recieve this class probabilites as predictions and we will have single integer labels as our true labels. To tackle this, we will encode the data and have a consistent format.


```python
from tensorflow.keras import backend as K
import numpy as np

encoded_train_data = train_data.map(lambda imgs, labels: (imgs, tf.one_hot(labels, num_classes)))
encoded_val_data = val_data.map(lambda imgs, labels: (imgs, tf.one_hot(labels, num_classes)))
encoded_test_data = test_data.map(lambda imgs, labels: (imgs, tf.one_hot(labels, num_classes)))
```


```python
for img, label in encoded_train_data.take(1):
    print("Train data shape:", img.shape)
    print("Train label shape:", label.shape)
```

    Train data shape: (32, 300, 300, 3)
    Train label shape: (32, 3)


    2024-04-10 14:25:22.186480: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


We see that the labels have changed the shape from `(32,)` to `(32, 3)` with 32 being the batch size. New shape has the one hot encoded labels and so the shape of 3 for each label.

We also write custom function for `F1 Score` and add a `MultiMetricEarlyStopping` which is extended from `tf.keras.callbacks.Callback`. To tackle the false alarms, we make sure that both `F1 Score` and the `Accuracy` has dropped to trigger early stopping.

Very carefully see how we have decorated the `f1` function with `@keras.saving.register_keras_serializable(package="my_package", name="f1")`. This is needed for saving the model and loading it again. use the same decorator and function while loading the model as well.


```python
import keras

@keras.saving.register_keras_serializable(package="my_package", name="f1")
def f1(y_true, y_pred):
    # Calculate precision and recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    # Calculate precision and recall
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    
    # Calculate F1-score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    
    return f1

class MultiMetricEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, metrics_to_monitor, patience=0):
        super().__init__()
        self.metrics_to_monitor = metrics_to_monitor
        self.patience = patience
        self.best_values = {metric: -1 for metric in metrics_to_monitor}
        self.wait = 0

    
    def on_epoch_end(self, epoch, logs=None):
        # Check improvement on all monitored metrics
        mets = [(logs.get(metric), self.best_values[metric]) for metric in self.metrics_to_monitor]
        reduced = [logs.get(metric) < self.best_values[metric] for metric in self.metrics_to_monitor]
        
        if not all(reduced):
            self.best_values = {metric: logs.get(metric) for metric in self.metrics_to_monitor}
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
```


```python
# clone model
secondModel = tf.keras.Sequential([
  # input
  layers.Input(input_shape),

  # resize and rescale
  layers.Resizing(input_shape[0], input_shape[1]),
  layers.Rescaling(1./255),

  # convolutions
  layers.Conv2D(32, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Dropout(0.3, seed=42),
  layers.Flatten(),
  layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.Dense(num_classes, activation="softmax")
])

secondModel.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1])
multiMetricEarlyStopping = MultiMetricEarlyStopping(metrics_to_monitor=['val_accuracy', 'val_f1'], patience=2)
```

Our model architecture has remained the same but notice we added `f1` in the `metrics` while compiling the model. With this, we will directly have loss, accuracy and the f1 score inside `on_epoch_end`. This gives more modularity. Else, code looks bad if we measure this metric in the extended Callback class.


```python
secondHistory = secondModel.fit(encoded_train_data, validation_data=encoded_val_data, epochs=20, callbacks=[multiMetricEarlyStopping], verbose = 1)
```

    Epoch 1/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m78s[0m 530ms/step - accuracy: 0.6625 - f1: 0.6021 - loss: 4.4405 - val_accuracy: 0.9405 - val_f1: 0.9404 - val_loss: 1.0443
    Epoch 2/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 513ms/step - accuracy: 0.9495 - f1: 0.9489 - loss: 0.8438 - val_accuracy: 0.9603 - val_f1: 0.9617 - val_loss: 0.8731
    Epoch 3/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m73s[0m 511ms/step - accuracy: 0.9665 - f1: 0.9669 - loss: 0.6877 - val_accuracy: 0.9603 - val_f1: 0.9606 - val_loss: 0.8318
    Epoch 4/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 519ms/step - accuracy: 0.9616 - f1: 0.9612 - loss: 0.6904 - val_accuracy: 0.9702 - val_f1: 0.9717 - val_loss: 0.6544
    Epoch 5/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 524ms/step - accuracy: 0.9755 - f1: 0.9741 - loss: 0.5614 - val_accuracy: 0.9683 - val_f1: 0.9654 - val_loss: 0.7181
    Epoch 6/20
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m74s[0m 520ms/step - accuracy: 0.9803 - f1: 0.9809 - loss: 0.5650 - val_accuracy: 0.9683 - val_f1: 0.9686 - val_loss: 0.6414


Notice how we also have the f1 score logged with each epoch. 


```python
plotHistory(secondHistory)
```


    
![png](README_files/README_45_0.png)
    


Accuracy plot is not smooth but it doesnt necessarily be smooth


```python
secondModel.evaluate(encoded_test_data)
```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 163ms/step - accuracy: 0.6593 - f1: 0.6546 - loss: 1.5417





    [1.6222985982894897, 0.6263440847396851, 0.612413227558136]




```python
# get the labels from model3
mod2Preds = secondModel.predict(encoded_test_data)
mod2PredLabels = np.argmax(mod2Preds, axis=1)

true_labels = []
for img, label in test_data:
    true_labels.extend(label.numpy())

true_labels = np.array(true_labels)

# build the confusion matrix
conf_matrix = tf.math.confusion_matrix(true_labels, mod2PredLabels)
print(conf_matrix)
```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 129ms/step
    tf.Tensor(
    [[102   1  21]
     [  5  45  74]
     [ 19  19  86]], shape=(3, 3), dtype=int32)


    2024-04-10 14:32:51.315383: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


This has an alarming number of false positives and false negatives.


```python
# save the second model
tf.saved_model.save(secondModel, "models/secondModel")
```

    INFO:tensorflow:Assets written to: models/secondModel/assets


    INFO:tensorflow:Assets written to: models/secondModel/assets


**But early stopping saves the final weights which are not really the best. Let us use checkpointing which only updates on better and better performance.**

Does that make Early Stopping redundant? Ofcourse. But I am blessed with genuninely good computation power and time availability.

But we also saw that the model is still overfitting. Knowing that the validation performance is much higher but test performance is significantly low.


```python
from tensorflow.keras.callbacks import ModelCheckpoint
# clone model
model3 = tf.keras.Sequential([
  # input
  layers.Input(input_shape),

  # resize and rescale
  # layers.Resizing(input_shape[0], input_shape[1]),
  layers.Rescaling(1./255),

  # convolutions
  layers.Conv2D(32, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Dropout(0.25, seed=42),
  layers.Flatten(),
  layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.Dropout(0.3, seed=42),
  layers.Dense(num_classes, activation="softmax")
])

model3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1])

checkpoint_callback = ModelCheckpoint(filepath='models/best_model_checkpoint.keras',
                                      save_best_only=True,  # Only save the best model
                                      save_weights_only=False,  # Save the entire model
                                      monitor='val_f1',  # Monitor validation F1 score
                                      mode='max',  # Save the model when validation F1 score improves
                                      verbose=1)  # Print verbose messages

history3 = model3.fit(encoded_train_data, validation_data=encoded_val_data, epochs=3, callbacks=[checkpoint_callback], verbose = 1)
```

    Epoch 1/3
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 586ms/step - accuracy: 0.6131 - f1: 0.5166 - loss: 3.4193
    Epoch 1: val_f1 improved from -inf to 0.86340, saving model to models/best_model_checkpoint.keras
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m89s[0m 611ms/step - accuracy: 0.6138 - f1: 0.5173 - loss: 3.4094 - val_accuracy: 0.9048 - val_f1: 0.8634 - val_loss: 1.1364
    Epoch 2/3
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 594ms/step - accuracy: 0.8876 - f1: 0.8811 - loss: 0.9467
    Epoch 2: val_f1 improved from 0.86340 to 0.93829, saving model to models/best_model_checkpoint.keras
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m87s[0m 616ms/step - accuracy: 0.8875 - f1: 0.8810 - loss: 0.9469 - val_accuracy: 0.9365 - val_f1: 0.9383 - val_loss: 0.9350
    Epoch 3/3
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 613ms/step - accuracy: 0.9275 - f1: 0.9240 - loss: 0.7896
    Epoch 3: val_f1 improved from 0.93829 to 0.96022, saving model to models/best_model_checkpoint.keras
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m90s[0m 634ms/step - accuracy: 0.9274 - f1: 0.9238 - loss: 0.7900 - val_accuracy: 0.9643 - val_f1: 0.9602 - val_loss: 0.7790


This very well mentions when the model had better performance and does not backup the model with lower perfomance. So we dont bother with newer epochs performing worse. But, our final weights are not necessarily the best weights so we will load back the model and evaluate on that.


```python
bestModel = tf.keras.models.load_model('models/best_model_checkpoint.keras')
bestModel.evaluate(encoded_test_data)
```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 157ms/step - accuracy: 0.6767 - f1: 0.6388 - loss: 1.3354





    [1.3580933809280396, 0.6586021780967712, 0.6305719017982483]



Woah!! this is great! We will also confirm the confusion matrix.


```python
# get the labels from model3
pred_labels = bestModel.predict(encoded_test_data)
pred_labels = np.argmax(pred_labels, axis=1)

true_labels = []
for img, label in test_data:
    true_labels.extend(label.numpy())

true_labels = np.array(true_labels)

# build the confusion matrix
conf_matrix = tf.math.confusion_matrix(true_labels, pred_labels)
print(conf_matrix)

```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 148ms/step
    tf.Tensor(
    [[109   0  15]
     [  4  68  52]
     [ 35  21  68]], shape=(3, 3), dtype=int32)


    2024-04-10 15:04:34.941819: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


This has a significantly lower level of false positives and false negatives! Now that we have better understanding of how to improve performance, let us now use deeper networks which can help us to identify more filters and give us more points to identify the classification labels. Being blessed with M1 chip, we ll add 2 more pairs of Conv2D and MaxPool.


```python
deeper = tf.keras.Sequential([
  # input
  layers.Input(input_shape),

  # resize and rescale
  # layers.Resizing(input_shape[0], input_shape[1]),
  layers.Rescaling(1./255),

  # convolutions
  layers.Conv2D(32, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.MaxPooling2D(),
  layers.Dropout(0.4, seed=42),
  layers.Flatten(),
  layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  layers.Dropout(0.5, seed=42),
  layers.Dense(num_classes, activation="softmax")
])

checkpoint_callback = ModelCheckpoint(filepath='models/deeper_model_checkpoint.keras',
                                      save_best_only=True, 
                                      save_weights_only=False, 
                                      monitor='val_f1', 
                                      mode='max', 
                                      verbose=1) 


deeper.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", f1])

history4 = deeper.fit(encoded_train_data, validation_data=encoded_val_data, epochs=3, callbacks=[checkpoint_callback], verbose = 1)
```

    Epoch 1/3
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 436ms/step - accuracy: 0.4172 - f1: 0.1333 - loss: 2.0269
    Epoch 1: val_f1 improved from -inf to 0.81968, saving model to models/deeper_model_checkpoint.keras
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m68s[0m 455ms/step - accuracy: 0.4178 - f1: 0.1344 - loss: 2.0233 - val_accuracy: 0.8433 - val_f1: 0.8197 - val_loss: 0.7206
    Epoch 2/3
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 442ms/step - accuracy: 0.8136 - f1: 0.8076 - loss: 0.7117
    Epoch 2: val_f1 improved from 0.81968 to 0.98587, saving model to models/deeper_model_checkpoint.keras
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m65s[0m 457ms/step - accuracy: 0.8135 - f1: 0.8076 - loss: 0.7118 - val_accuracy: 0.9901 - val_f1: 0.9859 - val_loss: 0.3656
    Epoch 3/3
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 437ms/step - accuracy: 0.8870 - f1: 0.8899 - loss: 0.5086
    Epoch 3: val_f1 did not improve from 0.98587
    [1m142/142[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m64s[0m 452ms/step - accuracy: 0.8869 - f1: 0.8897 - loss: 0.5089 - val_accuracy: 0.9901 - val_f1: 0.9826 - val_loss: 0.3141



```python
deeperModel = tf.keras.models.load_model('models/deeper_model_checkpoint.keras')
deeperModel.evaluate(encoded_test_data)
```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 151ms/step - accuracy: 0.7809 - f1: 0.7891 - loss: 0.6612





    [0.6744673252105713, 0.774193525314331, 0.7840778231620789]




```python
pred_labels = deeperModel.predict(encoded_test_data)
pred_labels = np.argmax(pred_labels, axis=1)

true_labels = []
for img, label in test_data:
    true_labels.extend(label.numpy())

true_labels = np.array(true_labels)

# build the confusion matrix
conf_matrix = tf.math.confusion_matrix(true_labels, pred_labels)
print(conf_matrix)

```

    [1m12/12[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 133ms/step
    tf.Tensor(
    [[112   0  12]
     [  0  90  34]
     [  9  29  86]], shape=(3, 3), dtype=int32)


    2024-04-10 14:39:57.714276: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


This has even better performance and lesser false positives and negatives. Values have shifted towards the diagonal in the confusion matrix. 


```python
# visual and predict on some test images
test_sample = test_data.take(1)
test_sample = test_sample.unbatch().batch(10)
predictedLabels = deeperModel.predict(test_sample)
predictedLabels = np.argmax(predictedLabels, axis=1)

# get 10 images and their labels from test_sample
test_viz = test_sample.unbatch()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (image, label) in enumerate(test_viz):
    axes[i%2, i%5].imshow(image)
    axes[i%2, i%5].set_title(f"True: {info.features['label'].int2str(label.numpy())}\nPredicted: {info.features['label'].int2str(predictedLabels[i])}")
    axes[i%2, i%5].axis("off")
```

          1/Unknown [1m0s[0m 25ms/step[1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 29ms/step


    2024-04-10 16:31:18.145254: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
    2024-04-10 16:31:18.222722: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence



    
![png](README_files/README_62_2.png)
    




**We now git push and go home early.!** 

We already documented our steps well so we now export this to a markdown format to put it as readme. 


```python
!jupyter nbconvert --to markdown "rock-paper-scissor.ipynb" --output README.md --TemplateExporter.exclude_input=False
```

    [NbConvertApp] Converting notebook rock-paper-scissor.ipynb to markdown
    [NbConvertApp] Support files will be in README_files/
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Writing 43617 bytes to README.md



```python

```
