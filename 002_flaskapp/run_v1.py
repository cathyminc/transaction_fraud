from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd
from Transformer import Frequency_Transformer_Single, Feq_Transformer_Multi

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open('frequency_encoder.pkl', 'rb') as f:
    frequency_encoder = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    form_data = {}
    
    if request.method == "POST":
    
        Sender_Id = request.form["Sender_Id"]
        Time_step = request.form["Time_step"]
        USD_amount = request.form["USD_amount"]  # Convert to float
        Sender_Sector = request.form["Sender_Sector"]
        Bene_Account = request.form["Bene_Account"]
        
        timestamp_object = pd.Timestamp(Time_step)   
        dayofyear = timestamp_object.dayofyear
        SDAYPair = Sender_Id + '-' + str(dayofyear)
         
        data= {'SDAYPair': SDAYPair, 
             'USD_amount': USD_amount, 
             'Sender_Sector': Sender_Sector, 
             'Bene_Account': Bene_Account
             }
            
        df = pd.DataFrame([data])
        
        data_transform = frequency_encoder.transform(df)
        
        pred = model.predict_proba(data_transform[['SDAYPair_ave', 'USD_amount_ave', 'Sender_Sector_ave', 'Bene_Account_ave']])[0][1]
        
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
