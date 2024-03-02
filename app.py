import pickle
from flask import Flask, request, jsonify, render_template
from src.pipelines.prediction_pipeline import CustomData,PredictPipline

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            Gender=int(request.form.get('Gender')),
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

        if result == "V-shape":
            return render_template("Results.html", final_result="Fitted Blazers and Jackets, V-neck Shirts and Sweaters, Layering")
        elif result == "Rectangular":
            return render_template("Results.html", final_result="Peplum Tops and Dresses, Belted Dresses and Tops,Wrap Dresses and Tops")
        elif result == "Hourglass":
            return render_template("Results.html", final_result="Bodycon Skirts and Tops")
        elif result == "Pear":
            return render_template("Results.html", final_result="Dark Wash Bottoms")
        elif result == "Triangle":
            return render_template("Results.html", final_result="Statement Tops")


if __name__ == "__main__":
    app.run(debug=True)