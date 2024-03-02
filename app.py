import os
from flask import Flask, request, jsonify, render_template
from src.pipelines.prediction_pipeline import CustomData,PredictPipline

app = Flask(__name__)

picFolder = os.path.join('static', 'pics')

app.config['UPLOAD_FOLDER'] = picFolder

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

        imageList1 = os.listdir('static/vshape')
        imagelist1 = ['vshape/1.jpeg','vshape/2.png','vshape/3.jpeg','vshape/4.jpg']
        
        imageList2 = os.listdir('static/Rectangular')
        imagelist2 = ['Rectangular/1.jpg','Rectangular/2.jpg','Rectangular/3.jpg','Rectangular/4.png']

        imageList3 = os.listdir('static/Hourglass')
        imagelist3 = ['Hourglass/1.png','Hourglass/2.jpg','Hourglass/3.jpg','Hourglass/4.png']

        imageList4 = os.listdir('static/Pear')
        imagelist4 = ['Pear/1.jpg','Pear/2.png','Pear/3.png','Pear/4.jpg']

        imageList5 = os.listdir('static/Triangle')
        imagelist5 = ['Triangle/1.png','Triangle/2.png','Triangle/3.jpg','Triangle/4.png']

        if result == "V-shape":
            return render_template("Results1.html",imagelist=imagelist1)
        elif result == "Rectangular":
            return render_template("Results2.html", imagelist=imagelist2)
        elif result == "Hourglass":
            return render_template("Results3.html", imagelist=imagelist3)
        elif result == "Pear":
            return render_template("Results4.html",imagelist=imagelist4)
        elif result == "Triangle":
            return render_template("Results5.html", imagelist=imagelist5)


if __name__ == "__main__":
    app.run(debug=True)