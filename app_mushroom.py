from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
app = Flask(__name__)
model = pickle.load(open('pycharm_mushrooms_dec_tree.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('mushroom.html')


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Capshape=request.form['Capshape']
        Capsurface = request.form['Capsurface']
        Gillattachment = request.form['Gillattachment']
        Gillsize = request.form['Gillsize']
        Population=request.form['Population']
        Habitat = request.form['Habitat']

        #One-Hot Encoding Cap-shape
        shapes=['b','c','f','k','s']
        shape_enc = [0.0, 0.0, 0.0, 0.0, 0.0]
        if Capshape in shapes:
            shape_ind=shapes.index(Capshape)
            shape_enc[shape_ind]=1.0

        # One-Hot Encoding Cap-surface
        surfaces=['f','g','s']
        surface_enc=[0.0,0.0,0.0]
        if Capsurface in surfaces:
            surface_ind = surfaces.index(Capsurface)
            surface_enc[surface_ind] = 1.0

        # One-Hot Encoding GillAttachment
        if (Gillattachment == 'a'):
            Gill_attachment_enc = [1.0]
        else:
            Gill_attachment_enc = [0.0]

        # One-Hot Encoding GillSize
        if (Gillsize == 'b'):
            Gill_size_enc = [1.0]
        else:
            Gill_size_enc = [0.0]

        #One-Hot Encoding Population
        populations=['a','c','n','s','v']
        pop_enc=[0.0,0.0,0.0,0.0,0.0]
        if Population in populations:
            pop_ind = populations.index(Population)
            pop_enc[pop_ind] = 1.0

        # One-Hot Encoding Habitat
        habitations = ['d', 'g', 'l', 'm', 'p','u']
        habitat_enc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if Habitat in habitations:
            habit_ind=habitations.index(Habitat)
            habitat_enc[habit_ind]=1.0
        #making of test data so that model can consume
        test_data = []
        for i in [shape_enc, surface_enc, Gill_attachment_enc, Gill_size_enc, pop_enc, habitat_enc]:
            test_data.extend(i)
        test=np.array(test_data).reshape(1,-1) # as the test data has 1 row & 21 columns
        prediction=model.predict(test)
        print('Prediction: ',prediction, 'Type: ',type(prediction))
        if prediction:
            print('Printing mushroom as Poisonous')
            return render_template('mushroom.html',prediction_text="This Mushroom is Poisonous!!")
        else:
            print('Printing mushroom as Edible')
            return render_template('mushroom.html',prediction_text="This Mushroom is Edible & Safe to Consume!!")
    else:
        return render_template('mushroom.html')

if __name__=="__main__":
    app.run(debug=True)