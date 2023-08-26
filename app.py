from flask import Flask, render_template, request, send_file
from az_tts import generate_voice
import os
import time

app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    print('rendered')
    return render_template('index.html')

@app.route('/', methods=['POST', 'GET'])
def get_text():
    text = request.form.get('inputText')
    audio_path = generate_voice(text)
    print(text)
    # time.sleep(15)
    return render_template('index.html', audio_path=audio_path, your_text = text)


@app.route('/audio_temp/<path:filename>')
def get_audio(filename):
    audio_path = f"audio_temp/{filename}"
    return send_file(audio_path)

if __name__ == '__main__':
    app.run(debug=True, port=8001)
