# These two imports is for the mock data for the live chart
import math
from random import random
from time import time
# mock data ends here
from flask import Flask, jsonify, make_response, request
import soundfile as sf
import os
from models.arm_model import ArmModel
import json
from flask_cors import CORS
import datetime
import pandas as pd 


app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return 'Server running for Flask application'

data_joint = None
sample_rate_joint = None
data_imu = None
sample_rate_imu = None
current_joint_index = 0
current_imu_index = 0

def load_joint_data():

    global data_joint, sample_rate_joint 
    # Change file path accordingly
    path = 'joint_data.wav'
    if not os.path.exists(path):
            return jsonify({'error': 'File not found with this path'}), 404
    
    data_joint, sample_rate_joint = sf.read(path)

def load_imu_data():

    global data_imu, sample_rate_imu 
    # Change file path accordingly
    path = 'imu_data.wav'
    if not os.path.exists(path):
            return jsonify({'error': 'File not found with this path'}), 404
    
    data_imu, sample_rate_imu = sf.read(path)


# Loading joint data
load_joint_data()    
load_imu_data()

@app.route('/get_joint_data', methods=['GET'])
def get_joint_data():
    try:
        # Necessary because numpy arrays are not serializable
        data_list = data_joint.tolist()

        response = {
            'sample_rate': sample_rate_joint,
            'data_joint': data_list 
        }

        json_obj = jsonify(response)

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
        # Necessary because numpy arrays are not serializable
        data_list = data_imu.tolist()
        data_list_length = len(data_list)
        print(data_list_length)

        response = {
            'sample_rate': sample_rate_imu,
            'data_imu': data_list 
        }

        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Returns a single point and increments global index to get ready for the next point
@app.route('/get_next_imu_point', methods=['GET'])
def get_next_imu_point():

    global current_imu_index
    if current_imu_index >= len(data_imu):
        return jsonify({"error": "All data has been returned"}), 404
    
    next_point = data_imu[current_imu_index]
    response = {
        'index': current_imu_index,
        'point': next_point
    }
    current_imu_index += 1
    return jsonify(response)

@app.route('/get_arm_data', methods=['GET'])
def get_arm_data():
     file = 'P001_T001_armSwing_fast_combined.xlsx'
     arm_model = ArmModel(file)
     data = arm_model.get_data_from_file()
     return jsonify(data)

# file = 'P001_T001_armSwing_fast_combined.xlsx'
# arm_model = ArmModel(file)
@app.route('/get_single_point', methods=['GET'])
def get_single_point():
    idx = int(request.args.get('index', 0))

    data_point = arm_model.get_data_point(idx)
    # idx =+ 1
    if data_joint is not None:
         return jsonify(data_point)
    else:
         return jsonify({'error': 'Index out of range'}), 400
    
# Mock endpoint for live data generator
@app.route('/data', methods=["GET", "POST"])
def data():
    #  mock data
    data = [time() * 1000, random()*100]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

@app.route('/large_data', methods=["GET"])
def large_data():
    n = 64000
    arr = []
    current_year = datetime.datetime.utcnow().year
    start_date = datetime.datetime(current_year, 1, 1)
    i = 0
    x = start_date - datetime.timedelta(hours=n)
    for i in range(n):
        if (i % 100 == 0): 
            a = 2 * random()
        
        if (i % 1000 == 0): 
            b = 2 * random()
        
        if (i % 10000 == 0): 
            c = 2 * random()
        
        if (i % 50000 == 0): 
            spike = 10
        else: 
            spike = 0

        x = x + datetime.timedelta(hours=1)

        x = x + datetime.timedelta(hours=1)
        arr.append([
            x.timestamp() * 1000,  # Convert to milliseconds
            2 * math.sin(i / 100) + a + b + c + spike + random()
        ])

    print("array length: ", len(arr))
    response = jsonify(arr)
    return response

@app.route('/sensor1_emg', methods=['GET'])
def sensor1_emg():
    n = 29975
    df = pd.read_csv('sensor1_emg.csv')

    if 'time' not in df.columns or 'IM EMG1' not in df.columns:
        return jsonify({'error': 'Required columns'}), 400
    
    data = df[['time', 'IM EMG1']].head(n)
    data = df.values.tolist()
    return jsonify(data)

@app.route('/joint_angle_pred', methods=['GET'])
def joint_angle_pred():
    df = pd.read_csv('joint_angle_pred.csv')
    n = df.size

    if 'time' not in df.columns or 'elbow_flex_r_pred' not in df.columns:
        return jsonify({'error': 'Required columns'}), 400
    
    data = df[['time', 'elbow_flex_r_pred']].head(n)
    data = df.values.tolist()
    return jsonify(data)


          

if __name__ == '__main__':
    app.run(port=5000, debug=True)