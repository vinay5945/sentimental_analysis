# app.py

from flask import Flask, render_template, request

# Import your sentiment analysis function from your Python script
from analyze import analyze_sentiment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        review = request.form['review']
        # Call your sentiment analysis function
        data = [review]
        sentiment_result = analyze_sentiment(data)
        return render_template('index.html', result=sentiment_result)

if __name__ == '__main__':
    app.run(debug=False)
