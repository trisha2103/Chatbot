# 🤖 Deep Learning Chatbot — NLP Intent Classification

> A conversational chatbot built with **TensorFlow/Keras** that classifies user messages into intents using an **Embedding + GlobalAveragePooling** neural network — trained on a custom `intents.json` dataset and saved for deployment.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-NLP-red?style=flat-square&logo=keras)
![NLP](https://img.shields.io/badge/NLP-Intent%20Classification-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 📌 Overview

This project builds an **intent-based chatbot** from scratch using deep learning. User messages are tokenized, embedded, and classified into intent categories. The bot then picks a random response from the matched intent's response pool — creating natural, varied replies.

The trained model, tokenizer, and label encoder are all **saved as files**, making the chatbot ready to deploy in any Python environment.

---

## 💬 How the Chatbot Works

```
User types a message
        ↓
Tokenize & pad the input sequence
        ↓
Feed into trained Embedding + DNN model
        ↓
Predict the intent tag (e.g. "greeting", "goodbye")
        ↓
Pick a random response from that intent's response list
        ↓
Bot replies with color-coded output
```

---

## 🗂️ Project Structure

```
Chatbot/
├── chatbot_demo_dl.ipynb       # Training + chat loop notebook
├── intents.json                # Custom intent dataset (patterns & responses)
├── saved_model.pb              # Saved TensorFlow model
├── keras_metadata.pb           # Keras model metadata
├── variables.data-00000-of-00001
├── variables.index
├── tokenizer.pickle            # Fitted tokenizer (saved)
├── label_encoder.pickle        # Fitted label encoder (saved)
├── LICENSE
└── README.md
```

---

## 🧠 Model Architecture

| Layer | Details |
|---|---|
| `Embedding` | vocab_size=1000, embedding_dim=16, input_length=20 |
| `GlobalAveragePooling1D` | Averages token embeddings into a fixed vector |
| `Dense` | 16 units · ReLU |
| `Dense` | 16 units · ReLU |
| `Dense` (output) | `num_classes` units · Softmax |

```python
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(16, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

## 📖 How It Works

### Step 1 — Load the Intents Dataset

```python
with open('intents.json') as file:
    data = json.load(file)
```

The `intents.json` file contains intent tags, training patterns, and possible responses:

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "Hey there"],
      "responses": ["Hello!", "Hi there!", "Hey! How can I help?"]
    },
    ...
  ]
}
```

### Step 2 — Tokenize & Encode

```python
# Tokenize patterns
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)

# Encode intent labels
lbl_encoder = LabelEncoder()
lbl_encoder.fit(training_labels)
training_labels = lbl_encoder.transform(training_labels)
```

### Step 3 — Train for 500 Epochs

```python
history = model.fit(padded_sequences, np.array(training_labels), epochs=500)
```

### Step 4 — Save Model & Artifacts

```python
model.save("chatbot_model")

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
```

### Step 5 — Run the Chatbot

```python
def chat():
    model = keras.models.load_model('chatbot_model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    while True:
        inp = input("User: ")
        if inp.lower() == "quit":
            break

        result = model.predict(
            pad_sequences(tokenizer.texts_to_sequences([inp]),
                          truncating='post', maxlen=20))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print("ChatBot:", np.random.choice(i['responses']))

chat()
```

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Vocab size | 1,000 |
| Embedding dim | 16 |
| Max sequence length | 20 |
| OOV token | `<OOV>` |
| Optimizer | Adam |
| Loss | Sparse Categorical Crossentropy |
| Epochs | 500 |

---

## 🚀 Features

- ✅ **Custom intent dataset** via `intents.json` — fully customizable
- ✅ **Embedding layer** for learned word representations
- ✅ **GlobalAveragePooling** for lightweight sequence encoding
- ✅ **Color-coded terminal output** via `colorama` (blue for user, green for bot)
- ✅ Model, tokenizer & label encoder all **saved for deployment**
- ✅ Random response selection for natural, varied replies

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| 🧠 `TensorFlow / Keras` | Model building & training |
| 📝 `Tokenizer` | Text tokenization |
| 🏷️ `LabelEncoder` | Intent label encoding |
| 🎨 `colorama` | Color-coded terminal chat |
| 💾 `pickle` | Save tokenizer & label encoder |
| 📄 `json` | Load intents dataset |

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/trisha2103/Chatbot.git
cd Chatbot

# 2. Install dependencies
pip install tensorflow scikit-learn colorama numpy

# 3. Add your intents.json file (or use the existing one)

# 4. Launch the notebook
jupyter notebook chatbot_demo_dl.ipynb
```

---

## 💡 Customizing the Chatbot

To add your own intents, edit `intents.json`:

```json
{
  "intents": [
    {
      "tag": "your_intent",
      "patterns": ["example input 1", "example input 2"],
      "responses": ["Bot reply 1", "Bot reply 2"]
    }
  ]
}
```

Then re-run the training cells — the model will automatically adapt to your new intents.

---


## 👩‍💻 Author

**Trisha** — [@trisha2103](https://github.com/trisha2103)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using TensorFlow, Keras & NLP</p>
