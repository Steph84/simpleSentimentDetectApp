from flask import Flask, render_template, request, flash
import pickle
from nlp_processing import natural_language_processing


app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"
model_file_name = "models/mod_bernoulli.h5"

@app.route("/message")
def index():
	flash("Copier le contenu du tweet ici :")
	return render_template("index.html")

@app.route("/detect", methods=['POST', 'GET'])
def detect():
	res = process_tweet(str(request.form['name_input']))
	flash("Le sentiment de la phrase '" + str(request.form['name_input']) + "' est " + res)
	return render_template("index.html")

def process_tweet(text_tweet):
	res = "Aucune idée..."

	# NLP processing
	clean_text = natural_language_processing(text_tweet)

	# load the model and detect
	loaded_model = pickle.load(open(model_file_name, 'rb'))
	temp_res = loaded_model.predict(clean_text)

	# compute resultat

	print(temp_res)
	if temp_res[0] > 0.5:
		res = "positif"
	else:
	    res = "négatif"
	return res