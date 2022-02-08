#import packages
import os
import pickle5 as pickle
import pandas as pd
from flask import Flask, jsonify, request
import json
from sklearn.neighbors import NearestNeighbors
import xgboost
import shap

# Load 
path = os.path.join('model', 'results_lgbm.pickle')
with open(path, 'rb') as file:
    model_obj = pickle.load(file)

model = model_obj[1]
thresh = 0.5

df = pd.read_csv(os.path.join('data', 'df_red.csv'))
y_train_df = df.pop('TARGET')
X = pd.read_csv(os.path.join('data', 'X_valid_red.csv'))
y_train = pd.read_csv(os.path.join('data', 'y_valid_red.csv'))

y_train.columns

X_shap = X.drop(columns=['SK_ID_CURR']).copy(deep=True)
y_shap = y_train.drop(columns=['SK_ID_CURR']).copy(deep=True)
y_train.columns

###############################################################
# initiate Flask app
app = Flask(__name__)

# API greeting message
@app.route("/")
def index():
    return ("API for Home Credit Default Risk Prediction, created by Khalil Henchi"), 200

# Get client's ID list
@app.route('/get_id_list/')
def get_id_list():
    temp = sorted(X['SK_ID_CURR'].values)
    temp_int = [int(i) for i in temp]
    id_list = json.loads(json.dumps(temp_int))
    return jsonify({'status': 'ok',
    		        'id_list': id_list}), 200
# Get client score 
@app.route('/get_score/')
def get_score():
    id = int(request.args.get('id'))
    temp_df = X[X['SK_ID_CURR']==id]
    temp_df.drop(columns=['SK_ID_CURR'], inplace=True)
    proba = model.predict_proba(temp_df)[:,1][0]
    score = model.predict(temp_df)
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': int(id),
                    'score': float(score),
    		        'proba': float(proba),
                    'thresh': float(thresh)}), 200

# Get client informations 
@app.route('/get_information_descriptive/')
def get_information_descriptive():
    id = int(request.args.get('id'))
    temp_df = df[df['SK_ID_CURR']==id]
    X_df_json = temp_df.to_json()
    return jsonify({'status': 'ok',
    				'df': X_df_json,}), 200

@app.route('/get_data/')
def get_data():
    df_json = X.to_json()
    y_train_json = y_train.to_json()
    return jsonify({'status': 'ok',
    				'X': df_json,
    				'y_train': y_train_json}), 200

# Get cliet's similar neighbors 
@app.route('/get_neighbors/')
def get_neighbors():
    id = int(request.args.get('id'))
    n_neighbors = int(request.args.get('n_neighbors'))
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    temp_df = X[X['SK_ID_CURR']==id]
    index = neigh.kneighbors(X=temp_df,
                           n_neighbors=n_neighbors,
                           return_distance=False)                
    X_neigh_df = X.iloc[index[0], :]
    y_neigh = y_train.iloc[index[0]]
    X_neigh_json = X_neigh_df.to_json()
    y_neigh_json = y_neigh.to_json()
    return jsonify({'status': 'ok',
    				'X_neigh': X_neigh_json,
    				'y_neigh': y_neigh_json}), 200

# Get model feature importance 
@app.route('/get_feature_importance/')
def get_feature_importance():
    features_importances = pd.Series(model.feature_importances_,
                                     index=X.drop(columns=['SK_ID_CURR'])
                                     .columns).sort_values(ascending=False).to_json()
    return jsonify({'status': 'ok',
    		        'features_importances': features_importances}), 200

# Get Shap Values                     
@app.route('/get_shap_values/')
def get_shap_values():
    model_clf = xgboost.XGBClassifier().fit(X_shap, y_shap)
    explainer = shap.TreeExplainer(model_clf)
    shap_values = explainer.shap_values(X_shap)
    shap_values_json =  json.loads(json.dumps(shap_values.tolist()))
    expected_value_json = json.loads(json.dumps(explainer.expected_value.tolist()))
    return jsonify({'status': 'ok',
                    'shap_values': shap_values_json,
                  #  'expected_value_json': expected_value_json}
                  ), 200

# main function 
if __name__ == "__main__":

    app.run(debug=True)
