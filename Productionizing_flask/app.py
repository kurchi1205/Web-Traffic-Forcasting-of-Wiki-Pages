from flask import Flask ,render_template,request
import final_pred as f
import numpy as np
app= Flask(__name__,template_folder="Template")
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    page = request.args.get('name')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    pagehits=f.final(page,start_date,end_date)
    return render_template('index.html', table=pagehits.to_html(classes='table-responsive'))
if __name__ == "__main__":
    app.run(debug=True)