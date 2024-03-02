import pickle
from flask import Flask, request, jsonify, render_template
from src.pipeline.prediction_pipeline import CustomData,PredictPipline

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=int(request.form.get('Gender')),
            Age =int(request.form.get('Age')),
            shoulder =int(request.form.get('ShoulderWidth')),
            chest =int(request.form.get('ChestWidth ')),
            waist=int(request.form.get('Waist ')),
            hips =int(request.form.get('Hips ')),
            shoulder_to_waist =int(request.form.get('ShoulderToWaist ')),
        )


        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = predict_pipline.predict(final_data)
        result = pred

        if result == 1:
            return render_template("Result.html", final_result="V-shape")
        elif result == 2:
            return render_template("Result.html", final_result="Rectangular")
        elif result == 3:
            return render_template("Result.html", final_result="Hourglass")
        elif result == 4:
            return render_template("Result.html", final_result="Pear")
        elif result == 5:
            return render_template("Result.html", final_result="Triangle")


if __name__ == "__main__":
    app.run()