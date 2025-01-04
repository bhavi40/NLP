from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

model = T5ForConditionalGeneration.from_pretrained("model/")
tokenizer = T5Tokenizer.from_pretrained("model/")
model.eval()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class TranslationHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    input_text = db.Column(db.Text, nullable=False)
    output_text = db.Column(db.Text, nullable=False)

# Routes
@app.route('/')
def home():
    return redirect(url_for('translate'))

@app.route('/translate', methods=['GET', 'POST'])
def translate():
    input_text = ""
    output_text = ""
    if request.method == 'POST':
        input_text = request.form.get('text')
        if input_text:
            inputs = tokenizer.encode(
                "translate Shakespeare to Modern English: " + input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            )
            outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if 'user' in session:
            user = User.query.filter_by(username=session['user']).first()
            history_entry = TranslationHistory(user_id=user.id, input_text=input_text, output_text=output_text)
            db.session.add(history_entry)
            db.session.commit()

    return render_template('Index.html', input_text=input_text, output_text=output_text)

@app.route('/history')
def history():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = User.query.filter_by(username=session['user']).first()
    user_history = TranslationHistory.query.filter_by(user_id=user.id).all()
    return render_template('History.html', history=user_history)

@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user'] = user.username
            return redirect(url_for('translate'))
        else:
            return render_template('Login.html', error="Invalid credentials!")
    return render_template('Login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('Register.html', error="User already exists!")
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('Register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
