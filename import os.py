import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score

# ============== Device Configuration ==============

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"TensorFlow is using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"TensorFlow GPU setup error: {e}")
    else:
        print("No TensorFlow GPU detected. Using CPU.")
    return gpus

def get_torch_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch running on: {device}")
    return device

# ============== Data Utilities ==============

def parse_itm_file(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            img_name, text, label = line.strip().split("\t")
            q_and_a = text.split("?")
            question = q_and_a[0].strip() + '?'
            answer = q_and_a[1].strip() if len(q_and_a) > 1 else ""
            records.append({
                'image': img_name,
                'question': question,
                'answer': answer,
                'label': 1 if label == "match" else 0
            })
    return pd.DataFrame(records)

def parse_grouped_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            img_name, question, answers_str = line.strip().split("\t")
            answers = answers_str.split(" | ")
            data.append({
                'image': img_name,
                'question': question,
                'answers': answers,
                'correct_answer': answers[0]
            })
    return data

# ============== Image Processing ==============

def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def init_resnet(device):
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_image_features(image_path, model, transform, device):
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return None

# ============== Feature Assembly ==============

def prepare_features(df, img_dir, resnet_model, transform, ans_embeddings):
    X, y = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image'])
        if os.path.exists(img_path) and row['answer'] in ans_embeddings:
            img_feats = extract_image_features(img_path, resnet_model, transform, device)
            if img_feats is not None:
                combined_feats = np.concatenate([img_feats.flatten(), ans_embeddings[row['answer']]])
                X.append(combined_feats)
                y.append(row['label'])
    return np.array(X), np.array(y)

# ============== Model Setup and Training ==============

def create_dnn(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'AUC'])
    return model

def train_model(X_train, y_train, X_val, y_val, gpus):
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        model = create_dnn(X_train.shape[1])
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=16,
            epochs=40,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=1
        )
    return model, history

# ============== Evaluation Utilities ==============

def plot_training(history):
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig('training_curve.png')

def evaluate_model(model, X, y, dataset_name="Validation"):
    y_pred = model.predict(X)
    y_pred_binary = (y_pred > 0.5).astype(int)
    print(f"{dataset_name} Classification Report:")
    print(classification_report(y, y_pred_binary))
    
    # Calculate and print the F1 score
    f1 = f1_score(y, y_pred_binary)
    print(f"{dataset_name} F1 Score: {f1:.4f}")
    
    return f1

def compute_mrr(model, grouped_data, scaler, ans_embeddings, img_dir, resnet_model, transform, device):
    ranks = []
    for sample in grouped_data:
        img_path = os.path.join(img_dir, sample['image'])
        if not os.path.exists(img_path):
            continue

        img_feats = extract_image_features(img_path, resnet_model, transform, device)
        if img_feats is None:
            continue

        scores = []
        for ans in sample['answers']:
            if ans in ans_embeddings:
                combo = np.concatenate([img_feats.flatten(), ans_embeddings[ans]])
                scaled = scaler.transform([combo])
                pred_score = model.predict(scaled)[0][0]
                scores.append(pred_score)
            else:
                scores.append(-np.inf)
        
        correct_idx = sample['answers'].index(sample['correct_answer'])
        rank_order = np.argsort(-np.array(scores))
        rank = np.where(rank_order == correct_idx)[0][0] + 1
        ranks.append(1.0 / rank)
    
    return np.mean(ranks) if ranks else 0.0

# ============== Main Execution ==============

if __name__ == "__main__":
    # Setup
    gpus = setup_gpu()
    device = get_torch_device()
    transform = build_transform()
    resnet_model = init_resnet(device)

    # Paths
    base_path = "/workspaces/cmp9137-advanced-machine-learning/CMP9137 Advanced Machine Learning/ITM_Classifier-baselines"
    img_dir = os.path.join(base_path, "visual7w-images")

    # Load data
    train_df = parse_itm_file(os.path.join(base_path, "visual7w-text/v7w.TrainImages.itm.txt"))
    val_df = parse_itm_file(os.path.join(base_path, "visual7w-text/v7w.DevImages.itm.txt"))
    grouped_val = parse_grouped_data(os.path.join(base_path, "visual7w-text/v7w.DevImages.txt"))

    with open(os.path.join(base_path, "v7w.sentence_embeddings-gtr-t5-large.pkl"), "rb") as f:
        ans_embeddings = pickle.load(f)

    # Feature Extraction
    print("Generating training features...")
    X_train, y_train = prepare_features(train_df, img_dir, resnet_model, transform, ans_embeddings)
    print("Generating validation features...")
    X_val, y_val = prepare_features(val_df, img_dir, resnet_model, transform, ans_embeddings)
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

    # Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train Model
    start_time = time.time()
    model, history = train_model(X_train_scaled, y_train, X_val_scaled, y_val, gpus)
    training_duration = time.time() - start_time
    print(f"Training completed in {training_duration:.2f} seconds.")

    plot_training(history)
    evaluate_model(model, X_val_scaled, y_val)

    # Test Data Evaluation
    test_df = parse_itm_file(os.path.join(base_path, "visual7w-text/v7w.TestImages.itm.txt"))
    X_test, y_test = prepare_features(test_df, img_dir, resnet_model, transform, ans_embeddings)
    X_test_scaled = scaler.transform(X_test)
    
    start_infer = time.time()
    evaluate_model(model, X_test_scaled, y_test, dataset_name="Test")
    infer_duration = time.time() - start_infer
    print(f"Inference completed in {infer_duration:.2f} seconds.")

    # Mean Reciprocal Rank (MRR)
    mrr_score = compute_mrr(model, grouped_val, scaler, ans_embeddings, img_dir, resnet_model, transform, device)
    print(f"Validation Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
