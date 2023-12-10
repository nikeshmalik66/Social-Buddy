#Importing Libraries
from flask import Flask, request, jsonify, render_template
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_sqlalchemy import SQLAlchemy
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
import re

app = Flask(__name__)

# --> Loading for Emotion Detection
model_path = "emotion_text_classifier"
tokenizer_path = "emotion_text_classifier"

loaded_model = RobertaForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)

# Set device for all
device = torch.device("cpu")
loaded_model.to(device)

# Ensure the model is in evaluation mode
loaded_model.eval()

# --> Loading for Hate Speech Model
hatespeech_model_path = "Hate-speech-detection"
hatespeech_tokenizer_path = "Hate-speech-detection"

loaded_hatespeech_model = AutoModelForSequenceClassification.from_pretrained(hatespeech_model_path)
loaded_hatespeech_tokenizer = AutoTokenizer.from_pretrained(hatespeech_tokenizer_path)

loaded_hatespeech_model.to(device)

# Ensure the model is in evaluation mode
loaded_hatespeech_model.eval()

# --> Loading for Sarcasm Detection
model_sarcasm_path = "sarcasm-detection"
tokenizer_sarcasm_path= "sarcasm-detection"

loaded_sarcasm_model = AutoModelForSequenceClassification.from_pretrained(model_sarcasm_path)
loaded_sarcasm_tokenizer = AutoTokenizer.from_pretrained(tokenizer_sarcasm_path)

# loaded_sarcasm_model = BertForSequenceClassification.from_pretrained(model_sarcasm_path)
# loaded_sarcasm_tokenizer = BertTokenizer.from_pretrained(tokenizer_sarcasm_path)

loaded_sarcasm_model.to(device)
loaded_sarcasm_model.eval()

# --> Loading for Slang Detection
model_slang_path = "slang_model_all"
tokenizer_slang_path= "slang_model_all"

loaded_slang_model = BertForSequenceClassification.from_pretrained(model_slang_path)
loaded_slang_tokenizer = BertTokenizer.from_pretrained(tokenizer_slang_path)

loaded_slang_model.to(device)
loaded_slang_model.eval()

# PREDICTION FUNCTIONS
# --> prediction function for emotion detection
emotion_dict = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

def predict_emotion(text, model, tokenizer):
    # Ensure the model is in evaluation mode and on the CPU
    model.eval()
    model.to(device)

    # Tokenize the user input
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    # Move input tensors to the CPU
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Get the model's logits
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()

    # Find the emotion with the highest score
    predicted_label = torch.argmax(logits, dim=1).item()

    return {
        "emotion": emotion_dict[predicted_label],
        "probabilities": probabilities
    }

# --> prediction function for hatespeech detection
label_dict = {
    0: "Not Hate Speech",
    1: "Hate Speech"
}

def predict_hate_speech(text, model, tokenizer):
    model.eval()
    model.to(device)

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    probabilities = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    predicted_label = torch.argmax(logits, dim=1).item()

    return {
        "label": label_dict[predicted_label],
        "probabilities": probabilities
    }

# --> prediction function for sarcasm detection
label_dict_sarcasm = {
    0: "Not Sarcasm",
    1: "Sarcasm"
}

def predict_sarcasm_speech(text, model, tokenizer):
    model.eval()
    model.to(device)

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    probabilities = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    predicted_label = torch.argmax(logits, dim=1).item()

    return {
        "sarcasm_label": label_dict_sarcasm[predicted_label],
        "probabilities": probabilities
    }

# --> prediction function for sarcasm detection

label_dict_slang = {
    0: "Not Slang",
    1: "Slang"
}

def detect_slang(text, model, tokenizer):
    model.eval()
    model.to(device)

    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    predicted_label = torch.argmax(logits, dim=1).item()

    # Return the prediction
    return {
        "slang_label": label_dict_slang[predicted_label],
        "probabilities": probs
    }


# --> Login and Registration Logic

# Secret key for session management and flash messages
app.secret_key = "some_secret_key"

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


#app routes and api endpoints

#for emotion
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    
    result = predict_emotion(text, loaded_model, loaded_tokenizer)
    
    return jsonify(result)

@app.route("/emotion")
@login_required
def emotion():
    return render_template("emotion.html", user=current_user)

#for hatespeech
@app.route("/predict_hatespeech", methods=["POST"])
def predict_hate():
    data = request.get_json()
    text = data["text"]
    
    result = predict_hate_speech(text, loaded_hatespeech_model, loaded_hatespeech_tokenizer)
    
    return jsonify(result)

@app.route("/hatespeech")
@login_required
def hatespeech():
    return render_template("hatespeech.html")

#for sarcasm
@app.route("/predict_sarcasm", methods=["POST"])
def predict_sarcasm():
    data = request.get_json()
    text = data["text"]
    
    result = predict_sarcasm_speech(text, loaded_sarcasm_model, loaded_sarcasm_tokenizer)
    
    return jsonify(result)

@app.route("/sarcasm")
@login_required
def sarcasm():
    return render_template("sarcasm.html")

# for slang
@app.route("/detect_slang", methods=["POST"])
def predict_slang():
    data = request.get_json()
    text = data["text"]
    
    result = detect_slang(text, loaded_slang_model, loaded_slang_tokenizer)
    
    return jsonify(result)

@app.route("/slang")
@login_required
def slang():
    return render_template("slang.html")

#for new slang detector
# Loading CSV file into a DataFrame
slang_df = pd.read_csv("/media/ayush/Ayush's Volume/Documents/Major Project Frontend/Frontend_Backend/slang_model_all/updated_slang_words.csv")  

# Converting the DataFrame into a dictionary
slang_dict = pd.Series(slang_df.slang_meaning.values, index=slang_df.slang.str.lower()).to_dict()

def detect_slang_new(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    slang_words = {word: slang_dict[word] for word in words if word in slang_dict}
    is_slang_sentence = "Slang" if len(slang_words) > 0 else "Not Slang"
    
    return  {
        "is_slang_sentence": is_slang_sentence,
        "slang_words": slang_words
    }
#api endpoint for slang
@app.route('/slang_detect', methods=['POST'])
def detect_slang_route():
    data = request.get_json()
    text = data["text"]

    result = detect_slang_new(text)
    return jsonify(result)

@app.route("/slang_new")
@login_required
def slang_new():
    return render_template("slang_new.html")


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Check if username or email already exists
        existing_user = User.query.filter_by(username=username).first()
        # existing_email = User.query.filter_by(email=email).first()
        
        if existing_user:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        # if existing_email:
        #     flash('Email already registered!', 'danger')
        #     return redirect(url_for('register'))

        # Add the user to the database
        user = User(username=username, email=email, password=password)  # In production, use hashed passwords
        db.session.add(user)
        db.session.commit()

        flash('Successfully registered!', 'success')
        return redirect(url_for('login'))
    
    return render_template('registration.html')

    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user exists and password matches
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:  # In production, compare hashed passwords
            login_user(user)
            flash('Successfully logged in!', 'success')
            return redirect(url_for('socialbuddy'))  # Redirect to desired page after login
        else:
            flash('Invalid credentials!', 'danger')
            return redirect(url_for('login'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Successfully logged out!', 'success')
    return redirect(url_for('login'))

@app.route("/socialbuddy")
@login_required
def socialbuddy():
    return render_template("mainpage.html", user=current_user)

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Creates the database table
    app.run(debug=True)
