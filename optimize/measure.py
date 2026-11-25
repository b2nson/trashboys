import os
import time
import numpy as np

def accuracy(predict, X, y):
    correct = 0

    for image, label in zip(X, y):
        probs = predict(image)
        pred_label = np.argmax(probs)
        if pred_label == label:
            correct += 1

    return correct / len(y)

def model_size(path):
    return os.path.getsize(path)

def inference_time(predict, X, n_warmup=5):
    # Warm-up
    for i in range(n_warmup):
        predict(X[i])

    start_time = time.perf_counter()
    for image in X:
        predict(image)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_image = total_time / len(X)
    return avg_time_per_image
