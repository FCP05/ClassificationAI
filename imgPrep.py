import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

###Polanco, Samuel ###
#Images Prep/Generate Dataset Function:
#-------------------------------------------------------
#Deletes any "JFIF" images, creates dataset, data 
#augmentation is activated and applied, and a prefetching
#setting is activated.
def prepGenerateData():
    imagesSkipped = 0
    print("Looking for Corrupt Images...")
    for folderName in ("Bird","Car","Cat","Dog"):
        folderPath = os.path.join("Images", folderName)
        for fname in os.listdir(folderPath):
            fpath = os.path.join(folderPath, fname)
            try:
                with open(fpath, "rb") as fobj:
                    aJfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                imagesSkipped += 1
                continue
    
            if not aJfif:
                imagesSkipped += 1
                # Delete corrupted Images
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"Error deleting {fpath}: {e}")
    print("Deleted %d Images" % imagesSkipped)
    
    
    image_size = (56, 56)
    batch_size = 75
    print("Making Batches...")
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        "Images",
        validation_split=0.2,
        subset="both",
        seed=42,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img), tf.one_hot(label, depth=4)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, label: (img, tf.one_hot(label, depth=4)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    print("Batches Made")
    
    return train_ds, val_ds