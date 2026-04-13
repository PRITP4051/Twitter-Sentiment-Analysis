import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Embedding, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# ------------------ LOAD DATA ------------------
df = pd.read_csv("labeled_data.csv")

df.dropna(subset=['clean_text'], inplace=True)
df['clean_text'] = df['clean_text'].astype(str)
df = df[df['clean_text'].str.strip() != ""]
df = df.sample(frac=1).reset_index(drop=True)

X = df['clean_text']
y = df['label']

# ------------------ TOKENIZATION ------------------
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=80)

# ------------------ SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ------------------ CLASS WEIGHTS ------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# ------------------ EARLY STOPPING ------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=1,   # 🔥 more aggressive
    restore_best_weights=True
)

results = []

# =====================================================
# 🚀 1. TRAIN BEST MODEL (GRU - OPTIMIZED)
# =====================================================

print("\n🔥 Training GRU (Optimized Best Model)")

gru_model = Sequential()

gru_model.add(Embedding(10000, 64, input_length=80))

gru_model.add(GRU(
    16,
    dropout=0.5,
    recurrent_dropout=0.5,
    kernel_regularizer=l2(0.01)
))

gru_model.add(Dropout(0.6))

gru_model.add(Dense(3, activation='softmax'))

gru_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


gru_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# Predictions
train_pred = np.argmax(gru_model.predict(X_train), axis=1)
test_pred = np.argmax(gru_model.predict(X_test), axis=1)

# Metrics
results.append({
    "model": "GRU",
    "train_accuracy": accuracy_score(y_train, train_pred),
    "test_accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred, average='weighted'),
    "recall": recall_score(y_test, test_pred, average='weighted'),
    "f1_score": f1_score(y_test, test_pred, average='weighted')
})

# Save GRU model
gru_model.save("GRU_model.h5")

# =====================================================
# 🔁 2. TRAIN OTHER MODELS (LSTM + RNN)
# =====================================================

def build_model(model_type):
    model = Sequential()
    model.add(Embedding(10000, 64, input_length=80))

    if model_type == "LSTM":
        model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3))
    elif model_type == "RNN":
        model.add(SimpleRNN(32, dropout=0.3))

    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


for model_name in ["LSTM", "RNN"]:
    print(f"\n🚀 Training {model_name}")

    model = build_model(model_name)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=1
    )

    train_pred = np.argmax(model.predict(X_train), axis=1)
    test_pred = np.argmax(model.predict(X_test), axis=1)

    results.append({
        "model": model_name,
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, average='weighted'),
        "recall": recall_score(y_test, test_pred, average='weighted'),
        "f1_score": f1_score(y_test, test_pred, average='weighted')
    })

    model.save(f"{model_name}_model.h5")

# =====================================================
# 📊 SAVE RESULTS
# =====================================================

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)

print("\n🎉 ALL MODELS TRAINED + RESULTS SAVED")
print(results_df)