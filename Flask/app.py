from flask import Flask, jsonify
import soundfile as sf
import os
import json

app = Flask(__name__)

@app.route("/")
def index():
    return 'Server running for Flask application'

data_joint = None
sample_rate_joint = None
data_imu = None
sample_rate_imu = None
current_joint_index = 0

def load_joint_data():

    global data_joint, sample_rate_joint 
    # Change file path accordingly
    path = 'joint_data.wav'
    if not os.path.exists(path):
            return jsonify({'error': 'File not found with this path'}), 404
    
    data_joint, sample_rate_joint = sf.read(path)

# Loading joint data
load_joint_data()    

@app.route('/get_joint_data', methods=['GET'])
def get_joint_data():
    try:
        # Necessary because numpy arrays are not serializable
        data_list = data_joint.tolist()

        response = {
            'sample_rate': sample_rate_joint,
            'data_joint': data_list 
        }
        json_str = json.dumps(response)
        json_obj = jsonify(response)
        with open('joint_data.json', 'w') as file:
            file.write(json_str)

        return json_obj
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# Returns a single point and increments global index to get ready for the next point
@app.route('/get_next_joint_point', methods=['GET'])
def get_next_joint_point():

    global current_joint_index
    if current_joint_index >= len(data_joint):
        return jsonify({"error": "All data has been returned"}), 404
    
    next_point = data_joint[current_joint_index]
    response = {
        'index': current_joint_index,
        'point': next_point
    }
    current_joint_index += 1
    return jsonify(response)

@app.route('/get_imu_data', methods=['GET'])
def get_imu_data():
    try:
        # Change file path accordingly
        path = 'imu_data.wav'

        if not os.path.exists(path):
            return jsonify({'error': 'File not found with this path'}), 404

        data_imu, sample_rate_imu = sf.read(path)

        # Necessary because numpy arrays are not serializable
        data_list = data_imu.tolist()

        response = {
            'sample_rate': sample_rate_imu,
            'data_imu': data_list 
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)