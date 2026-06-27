from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
with open("dropout_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")
@app.route("/predict", methods=["POST"])
def predict():

    try:
        age = float(request.form["age"])
        region = int(request.form["region"])
        exam_season = int(request.form["exam_season"])
        courses_enrolled = float(request.form["courses_enrolled"])
        completed_assignments = float(request.form["completed_assignments"])
        completion_rate = float(request.form["completion_rate"])
        login_frequency = float(request.form["login_frequency"])
        last_activity_days_ago = float(request.form["last_activity_days_ago"])
        forum_posts_count = float(request.form["forum_posts_count"])
        dropout_score = float(request.form["dropout_score"])
        features = np.array([[
            age,
            region,
            exam_season,
            courses_enrolled,
            completed_assignments,
            completion_rate,
            login_frequency,
            last_activity_days_ago,
            forum_posts_count,
            dropout_score
        ]])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        prediction_dict = {
    0: {
        "status": "Active",
        "message": "The student is currently active. Continue regular academic monitoring.",
        "color": "#28A745",
        "badge_class": "status-active"
    },

    1: {
        "status": "At Risk",
        "message": "The student is at risk of dropping out. Academic counselling and additional support are recommended.",
        "color": "#FFC107",
        "badge_class": "status-risk"
    },

    2: {
        "status": "Dropout",
        "message": "The student has a high probability of dropping out. Immediate intervention is recommended.",
        "color": "#DC3545",
        "badge_class": "status-dropout"
    }

}
        result = prediction_dict.get(
            prediction,
            {
                "status": "Unknown",
                "message": "Unable to determine the student's academic status.",
                "color": "#000000"
            }
        )
        return render_template(
            "result.html",
            prediction=result["status"],
            message=result["message"],
            color=result["color"],
            badge_class=result["badge_class"]
        )
    except Exception as e:
     return render_template(
        "result.html",
        prediction="Error",
        message=f"An error occurred: {str(e)}",
        color="red",
        badge_class="status-dropout"
    )
if __name__ == "__main__":
    app.run(debug=True)