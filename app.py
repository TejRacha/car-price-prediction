from flask import Flask, render_template,jsonify,request
import pandas as pd
import pickle
app =Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        make = request.form.get('make')
        model = request.form.get('model')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel')
        hp = request.form.get('hp')
        cylinders = request.form.get('cylinders')
        transmission = request.form.get('transmission')
        wheels = request.form.get('wheels')
        doors = request.form.get('doors')
        size = request.form.get('size')
        style = request.form.get('style')
        highway = request.form.get('highway')
        city = request.form.get('city')
        popularity = request.form.get('popularity')
        df=pd.read_json('new.json')
        make_encode = df['Make_encode'][df['Make']==make].values[0]
        model_encode = df['Model_encode'][df['Model']==model].values[0]
        eft_encode = df['Engine Fuel Type_encode'][df['Engine Fuel Type']==fuel_type].values[0]
        dw_encode = df['Driven_Wheels_encode'][df['Driven_Wheels']==wheels].values[0]
        tt_encode = df['Transmission Type_encode'][df['Transmission Type']==transmission].values[0]
        vs_encode = df['Vehicle Style_encode'][df['Vehicle Style']==style].values[0]
        vsz_encode = df['Vehicle Size_encode'][df['Vehicle Size']==size].values[0]
        with open('model.pkl', 'rb') as model_file:
            mlmodel = pickle.load(model_file)    
        predict=mlmodel.predict([[int(year),float(hp),float(cylinders),float(doors),float(highway),float(city),float(popularity),make_encode,model_encode,eft_encode,tt_encode,dw_encode,vsz_encode,vs_encode]])
        print(predict)
        print(make,model,year,fuel_type,hp,cylinders,transmission,wheels,doors,size,style,highway,city,popularity)
        return jsonify({'Predicted Result':f'result:{predict}'})
    else:       
        return render_template('predict.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',port = '5050')
 