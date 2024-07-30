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

df = pd.read_csv('joint_angle_pred.csv')


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
# load_joint_data()    
# load_imu_data()

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

# df = pd.read_csv('emg_data.csv')
# n = df['time'].size
@app.route('/sensor1_emg', methods=['GET'])
def sensor1_emg():
    n = 29975
    df = pd.read_csv('sensor1_emg.csv')

    if 'time' not in df.columns or 'IM EMG1' not in df.columns:
        return jsonify({'error': 'Required columns'}), 400
    
    data = df[['time', 'IM EMG1']].head(n)
    # print("data: ", data)
    data = df.values.tolist()
    print("size of df: ", n)
    # print(df.head())
    return jsonify(data)

@app.route('/emg_data', methods=['GET'])
def emg_data():
    df = pd.read_csv('emg_data.csv')
    if not set(['time', 'IM EMG1', 'IM EMG2', 'IM EMG3', 'IM EMG4', 'IM EMG5', 'IM EMG6']).issubset(df.columns):
        return jsonify({'error': 'Column name not found'}), 400
    data = df[['time', 'IM EMG1', 'IM EMG2', 'IM EMG3', 'IM EMG4', 'IM EMG5', 'IM EMG6']].to_dict(orient='list')
    return jsonify(data)

@app.route('/imu_data', methods=['GET'])
def imu_data():
    df = pd.read_csv('imu_data.csv')
    if not set(['time', 'ACCX1', 'ACCY1', 'ACCZ1', 'GYROX1', 'GYROY1', 'GYROZ1', 'ACCX2', 'ACCY2', 'ACCZ2', 'GYROX2', 'GYROY2', 'GYROZ2', 'ACCX3', 'ACCY3', 'ACCZ3', 'GYROX3', 'GYROY3', 'GYROZ3', 'ACCX4', 'ACCY4', 'ACCZ4', 'GYROX4', 'GYROY4', 'GYROZ4', 'ACCX5', 'ACCY5', 'ACCZ5', 'GYROX5', 'GYROY5', 'GYROZ5', 'ACCX6', 'ACCY6', 'ACCZ6', 'GYROX6', 'GYROY6', 'GYROZ6']):
        return jsonify({'Error': 'Column name not found from imu data'}), 400
    
    data = df[['time', 'ACCX1', 'ACCY1', 'ACCZ1', 'GYROX1', 'GYROY1', 'GYROZ1', 'ACCX2', 'ACCY2', 'ACCZ2', 'GYROX2', 'GYROY2', 'GYROZ2', 'ACCX3', 'ACCY3', 'ACCZ3', 'GYROX3', 'GYROY3', 'GYROZ3', 'ACCX4', 'ACCY4', 'ACCZ4', 'GYROX4', 'GYROY4', 'GYROZ4', 'ACCX5', 'ACCY5', 'ACCZ5', 'GYROX5', 'GYROY5', 'GYROZ5', 'ACCX6', 'ACCY6', 'ACCZ6', 'GYROX6', 'GYROY6', 'GYROZ6']].to_dict(orient='list')
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


pred_data_idx = 0
@app.route('/pred_data', methods=['GET'])
def pred_data():
    df = pd.read_csv('joint_angle_pred.csv')
    global pred_data_idx
    n = len(df)
    if pred_data_idx >= n:
        pred_data_idx = 0
        return jsonify({'Backend error': 'No more data available'}), 400

    # if 'time_pred' not in df.columns or 'elbow_flex_r_pred' not in df.columns or 'elbow_flex_r' in df.columns:
    #     return jsonify({'Backend error': 'Column name not found in csv file'}), 400
    
    row = df.iloc[pred_data_idx]
    result = {
        'time': row['time'],
        'elbow_flex_r': row['elbow_flex_r'],
        'elbow_flex_r_pred': row['elbow_flex_r_pred']
    }
    pred_data_idx += 1
    return jsonify(result), 200

# NEW ENDPOINT FOR DATA FROM JUST ONE SENSOR
pred_data_idx_one = 0
@app.route('/pred_data_one', methods=['GET'])
def pred_data_one():
    df = pd.read_csv('joint_angle_pred_one.csv')
    global pred_data_idx_one
    n = len(df)
    if pred_data_idx_one >= n:
        pred_data_idx_one = 0
        return jsonify({'Backend error': 'No more data available'}), 400

    # if 'time_pred' not in df.columns or 'elbow_flex_r_pred' not in df.columns or 'elbow_flex_r' in df.columns:
    #     return jsonify({'Backend error': 'Column name not found in csv file'}), 400
    
    row = df.iloc[pred_data_idx_one]
    result = {
        'time': row['time'],
        'elbow_flex_r': row['elbow_flex_r'],
        'elbow_flex_r_pred': row['elbow_flex_r_pred']
    }
    pred_data_idx_one += 1
    return jsonify(result), 200
# NEW ENDPOINT FOR DATA FROM JUST ONE SENSOR


if __name__ == '__main__':
    app.run(port=5000, debug=True)