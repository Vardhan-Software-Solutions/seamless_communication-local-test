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

    with open('s2tt_output.txt', 'w') as output_file:
        output_file.write('')
        
    subprocess.Popen(['python', 'appv2.py'])
    return jsonify({'message': 'File uploaded successfully and main.py executed'})


@app.route('/result-size', methods=['GET'])
def get_result():
    file_path = 's2tt_output.txt'
    try:
        file_size = os.path.getsize(file_path)
        return jsonify({'file_size': file_size})
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
        
# @app.route('/result1', methods=['GET'])
# def get_result():
#     with open('s2tt_output.txt', 'r') as file:
#         result_data = file.read()
#  return jsonify(result_data)

@app.route('/result', methods=['GET'])
def get_result():
    with open('s2tt_output.txt', 'r') as file:
        result_data = file.read()
    return jsonify(result_data)
        
@app.route('/config', methods=['GET'])
def get_state():
    return jsonify({'message': 'All okay'})


if __name__ == '__main__':
    app.run(host='0.0.0.0',  port=5000)
