# Product_Category_Classification
A CNN built for classifying images of product to their categories (Fashion - Artifacts - .....)

## Steps for building the model:
### 1. Collect Data
   Collect images for the five categories (Accessories, Artifacts, Fashion, Home, Stationary).
   Collect almost 100 image for each category
### 2. Preprocess Data
   After collecting the data, we needed to normalize all the pixels values to be value betwwen [0,1] as it fosters stability in the optimization process, promoting faster convergence during gradient-based training.
### 3. Build the model architecture
   We build the CNN needed for the model as follows:
   ```
   model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(256,256,3)),
    tf.keras.layers.Conv2D(64, kernel_size=(6, 6), padding='valid', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
   ```
### 4. Train the model
   Then, train the model for 50 Epochs until we reach a validation acuuracy of
   
### 5. Evaluate the model
   When we try the model on test data, it gives an acurracy of
   The Confusion Matrix of the model:
   
   The Precision, Recall and F1 score are as follows:
   
