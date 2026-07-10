# from flask import Flask
# from flask.templating import render_template

# app = Flask(__name__, static_url_path='/static')


# @app.route('/')
# def index():
#     return render_template('index.html', name='home')


# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, render_template, request

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    path = request.args.get('p', 'home')  # Default to 'home' if no query parameter is provided

    if path == 'home':
        return render_template('index.html')  # Ensure that home.html exists in templates folder
    
    else:
        return render_template('home.html', name='home')
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    app.run(debug=True)

