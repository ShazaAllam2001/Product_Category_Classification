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
   ```
      Model: "sequential_17"
   _________________________________________________________________
    Layer (type)                Output Shape              Param #   
   =================================================================
    conv2d_46 (Conv2D)          (None, 251, 251, 64)      6976      
                                                                    
    batch_normalization_56 (Ba  (None, 251, 251, 64)      256       
    tchNormalization)                                               
                                                                    
    max_pooling2d_46 (MaxPooli  (None, 125, 125, 64)      0         
    ng2D)                                                           
                                                                    
    conv2d_47 (Conv2D)          (None, 123, 123, 16)      9232      
                                                                    
    batch_normalization_57 (Ba  (None, 123, 123, 16)      64        
    tchNormalization)                                               
                                                                    
    max_pooling2d_47 (MaxPooli  (None, 61, 61, 16)        0         
    ng2D)                                                           
                                                                    
    flatten_30 (Flatten)        (None, 59536)             0         
                                                                    
    dense_51 (Dense)            (None, 64)                3810368   
                                                                    
    batch_normalization_58 (Ba  (None, 64)                256       
    tchNormalization)                                               
                                                                    
    flatten_31 (Flatten)        (None, 64)                0         
                                                                    
    dense_52 (Dense)            (None, 16)                1040      
                                                                    
    batch_normalization_59 (Ba  (None, 16)                64        
    tchNormalization)                                               
                                                                    
    dense_53 (Dense)            (None, 5)                 85        
                                                                    
   =================================================================
   Total params: 3828341 (14.60 MB)
   Trainable params: 3828021 (14.60 MB)
   Non-trainable params: 320 (1.25 KB)
   _________________________________________________________________
   ```

### 4. Train the model
   Then, train the model for 50 Epochs until we reach a validation acuuracy of
   
### 5. Evaluate the model
   When we try the model on test data, it gives an acurracy of
   The Confusion Matrix of the model:
   
   The Precision, Recall and F1 score are as follows:
   
