from flask import Flask, request, jsonify
import subprocess
import os
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    source_lang = request.form.get('sourceLang')
    target_lang = request.form.get('targetLang')
    if not (source_lang and target_lang):
        return jsonify({'error': 'sourceLang and targetLang are required'})

    # if not os.path.exists('downloads'):
    #     os.makedirs('downloads')
    file.save(os.path.join('input.wav'))

    data = {'sourceLang': source_lang, 'targetLang': target_lang}
    with open('config.json', 'w') as json_file:
        json.dump(data, json_file)

    audio_data = {'sourceLang': source_lang, 'targetLang': target_lang, 'filename': 'downloads/input.wav'}
    with open('audio.json', 'w') as audio_file:
        json.dump(audio_data, audio_file)

    subprocess.Popen(['python', 'appv2.py'])
    return jsonify({'message': 'File uploaded successfully and main.py executed'})


@app.route('/result', methods=['GET'])
def get_result():
    with open('s2tt_output.txt', 'r') as file:
        result_data = file.read()

    lines = result_data.split('\n')
    numbers = [int(line.split()[0]) for line in lines if line]
    output_text = None
    for line in lines:
        if line.startswith('100 '):
            output_text = line.split(' ', 1)[1]
            break

    response_data = {'numbers': numbers}
    if output_text:
        response_data['output_text'] = output_text

    return jsonify(response_data)


@app.route('/config', methods=['GET'])
def get_state():
    return jsonify({'message': 'All okay'})


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=5000)
