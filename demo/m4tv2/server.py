from flask import Flask, jsonify, request, send_file
import subprocess
import os

app = Flask(__name__)

# Endpoint to run the command "python app.py"
@app.route('/run-app', methods=['POST'])
def run_app():
    try:
        # Run the command
        result = subprocess.run(['python', 'appv2.py'], capture_output=True, text=True, check=True)
        return jsonify({"status": "success", "output": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "error": e.stderr})

# Endpoint to return the content of a specified file
@app.route('/read-file', methods=['GET'])
def read_file():
    # Retrieve the file path from the request args
    file_path = request.args.get('file_path')
    
    # Check if the file path was provided and exists
    if not file_path or not os.path.exists(file_path):
        return jsonify({"status": "error", "message": "File path not provided or file does not exist"})
    
    # Return the file content as a response
    try:
        return send_file(file_path, as_attachment=False)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
