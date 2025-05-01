from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    return "Test endpoint working"

if __name__ == '__main__':
    app.run(port=5003)
