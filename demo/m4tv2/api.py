from flask import Flask, request, jsonify
import subprocess
import os
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the file part is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    source_lang = request.form.get('sourceLang')
    target_lang = request.form.get('targetLang')
    
    # Check if both sourceLang and targetLang are provided
    if not (source_lang and target_lang):
        return jsonify({'error': 'sourceLang and targetLang are required'})

    # List of allowed source languages
    allowed_languages = ['pashto', 'nepali', 'punjabi', 'urdu']
    if source_lang not in allowed_languages:
        return jsonify({'error': 'Source language not recognized. Kindly contact the administrator.'})

    # Save the file
    file.save('input.wav')

    # Save config data
    data = {'sourceLang': source_lang, 'targetLang': target_lang}
    with open('config.json', 'w') as json_file:
        json.dump(data, json_file)

    # Save audio data
    audio_data = {'sourceLang': source_lang, 'targetLang': target_lang, 'filename': 'input.wav'}
    with open('audio.json', 'w') as audio_file:
        json.dump(audio_data, audio_file)

    os.environ['SOURCE_FILE_CONTENT'] = "s2tt_output_2.txt"
    os.environ['SOURCE_FILE_NAME'] = 'input.wav'
    os.environ['SOURCE_LANG'] = source_lang
    os.environ['TARGET_LANG'] = source_lang
    
    # Run subprocess
    subprocess.run(['python', 'appv2.py'])


    os.environ['SOURCE_FILE_CONTENT'] = "s2tt_output.txt"
    os.environ['SOURCE_FILE_NAME'] = 'input.wav'
    os.environ['SOURCE_LANG'] = source_lang
    os.environ['TARGET_LANG'] = target_lang
    
    # Run subprocess
    subprocess.run(['python', 'appv2.py'])


    os.environ.pop('SOURCE_FILE_NAME', None)
    os.environ.pop('SOURCE_LANG', None)
    os.environ.pop('TARGET_LANG', None)
    
    # Return success response
    return jsonify({'message': 'File uploaded successfully and main.py executed'})


@app.route('/result', methods=['GET'])
def get_result():
    with open('s2tt_output.txt', 'r') as file:
        result_data = file.read()

    return result_data


@app.route('/result-self', methods=['GET'])
def get_result():
    with open('s2tt_output_2.txt', 'r') as file:
        result_data = file.read()

    return result_data


@app.route('/config', methods=['GET'])
def get_state():
    return jsonify({'message': 'All okay'})

OUTPUT_FILE = "s2tt_output.txt"

@app.route('/result-size', methods=['GET'])
def get_result_size():
    try:
        file_size = os.path.getsize(OUTPUT_FILE)
        return jsonify({'message': 'All okay', 'file_size': file_size})
    except OSError as e:
        return jsonify({'message': 'Error reading file', 'error': str(e)}), 500
    
@app.route('/clear-file', methods=['POST'])
def clear_file():
    try:
        open("s2tt_output_2.txt", 'w').close()
        open("s2tt_output.txt", 'w').close()
        return jsonify({'message': 'File cleared successfully'})
    except OSError as e:
        return jsonify({'message': 'Error clearing file', 'error': str(e)}), 500


if __name__ == '__main__':
    print('server start------')
    app.run(debug=True,host='0.0.0.0', port=5000)


# import os
# import json
# import subprocess
# from io import BytesIO
# from wsgiref.util import setup_testing_defaults
# from urllib.parse import parse_qs
# from waitress import serve

# def application(environ, start_response):
#     setup_testing_defaults(environ)
#     path = environ.get('PATH_INFO', '').lstrip('/')
#     method = environ.get('REQUEST_METHOD')

#     # if path == 'upload' and method == 'POST':
#     #     try:
#     #         # Read and parse form data
#     #         content_length = int(environ.get('CONTENT_LENGTH', 0))
#     #         body = environ['wsgi.input'].read(content_length)
#     #         form_data = parse_qs(body.decode('utf-8'))

#     #         # Check if file part exists
#     #         if b'file' not in form_data:
#     #             start_response('400 Bad Request', [('Content-Type', 'application/json')])
#     #             return [json.dumps({'error': 'No file part'}).encode('utf-8')]

#     #         # Simulate saving the file
#     #         file = form_data[b'file'][0]
#     #         file_path = 'input.wav'
#     #         with open(file_path, 'wb') as f:
#     #             f.write(file.encode('utf-8'))

#     #         # Read languages
#     #         source_lang = form_data.get(b'sourceLang', [None])[0]
#     #         target_lang = form_data.get(b'targetLang', [None])[0]
#     #         if not (source_lang and target_lang):
#     #             start_response('400 Bad Request', [('Content-Type', 'application/json')])
#     #             return [json.dumps({'error': 'sourceLang and targetLang are required'}).encode('utf-8')]

#     #         # Save config
#     #         data = {'sourceLang': source_lang.decode('utf-8'), 'targetLang': target_lang.decode('utf-8')}
#     #         with open('config.json', 'w') as json_file:
#     #             json.dump(data, json_file)

#     #         # Save audio data
#     #         audio_data = {'sourceLang': source_lang.decode('utf-8'), 'targetLang': target_lang.decode('utf-8'), 'filename': 'input.wav'}
#     #         with open('audio.json', 'w') as audio_file:
#     #             json.dump(audio_data, audio_file)

#     #         # Run subprocess
#     #         subprocess.run(['python', 'app.py'])

#     #         # Return success response
#     #         start_response('200 OK', [('Content-Type', 'application/json')])
#     #         return [json.dumps({'message': 'File uploaded successfully and main.py executed'}).encode('utf-8')]

#     #     except Exception as e:
#     #         start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
#     #         return [json.dumps({'error': str(e)}).encode('utf-8')]
#     if path == 'upload' and method == 'POST':
#         try:
#             # Read and parse form data
#             content_length = int(environ.get('CONTENT_LENGTH', 0))
#             body = environ['wsgi.input'].read(content_length)
#             form_data = parse_qs(body.decode('utf-8'))

#             # Check if file part exists
#             if b'file' not in form_data:
#                 start_response('400 Bad Request', [('Content-Type', 'application/json')])
#                 return [json.dumps({'error': 'No file part'}).encode('utf-8')]

#             # Simulate saving the file
#             file = form_data[b'file'][0]
#             file_path = 'input.wav'
#             with open(file_path, 'wb') as f:
#                 f.write(file.encode('utf-8'))

#             # Read languages
#             source_lang = form_data.get(b'sourceLang', [None])[0]
#             target_lang = form_data.get(b'targetLang', [None])[0]
#             if not (source_lang and target_lang):
#                 start_response('400 Bad Request', [('Content-Type', 'application/json')])
#                 return [json.dumps({'error': 'sourceLang and targetLang are required'}).encode('utf-8')]

#             # Check if the source language is recognized
#             allowed_languages = ['pashto', 'nepali', 'punjabi', 'urdu']
#             if source_lang.decode('utf-8') not in allowed_languages:
#                 start_response('400 Bad Request', [('Content-Type', 'application/json')])
#                 return [json.dumps({'error': 'Source language not recognized. Kindly contact the administrator.'}).encode('utf-8')]

#             # Save config
#             data = {'sourceLang': source_lang.decode('utf-8'), 'targetLang': target_lang.decode('utf-8')}
#             with open('config.json', 'w') as json_file:
#                 json.dump(data, json_file)

#             # Save audio data
#             audio_data = {'sourceLang': source_lang.decode('utf-8'), 'targetLang': target_lang.decode('utf-8'), 'filename': 'input.wav'}
#             with open('audio.json', 'w') as audio_file:
#                 json.dump(audio_data, audio_file)

#             # Run subprocess
#             subprocess.run(['python', 'app.py'])

#             # Return success response
#             start_response('200 OK', [('Content-Type', 'application/json')])
#             return [json.dumps({'message': 'File uploaded successfully and main.py executed'}).encode('utf-8')]

#         except Exception as e:
#             start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
#             return [json.dumps({'error': str(e)}).encode('utf-8')]


#     elif path == 'result' and method == 'GET':
#         try:
#             with open('s2tt_output.txt', 'r') as file:
#                 result_data = file.read()

#             start_response('200 OK', [('Content-Type', 'text/plain')])
#             return [result_data.encode('utf-8')]

#         except Exception as e:
#             start_response('500 Internal Server Error', [('Content-Type', 'application/json')])
#             return [json.dumps({'error': str(e)}).encode('utf-8')]

#     elif path == 'config' and method == 'GET':
#         start_response('200 OK', [('Content-Type', 'application/json')])
#         return [json.dumps({'message': 'API RUNNING YOU CAN CONNECT YOUR REACTJS SERVER'}).encode('utf-8')]

#     else:
#         start_response('404 Not Found', [('Content-Type', 'application/json')])
#         return [json.dumps({'error': 'Not Found'}).encode('utf-8')]


# if __name__ == '__main__':
#     print("Starting server... at 5000")
#     serve(application, host='0.0.0.0', port=5000)

