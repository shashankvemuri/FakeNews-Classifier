from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/',methods=['POST'])
def predict():
	filename='./model/DT_classifier.sav'
	model = pickle.load(open(filename, 'rb'))

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = model.predict(data)
		print (my_prediction)
	
	return render_template('result.html', prediction = my_prediction, message=message)

if __name__ == '__main__':
	app.run(debug=True)