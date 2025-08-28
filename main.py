from flask import Flask, request, render_template, jsonify,flash,redirect

main = Flask(__name__)


@main.route('/')
def index():
    return render_template('home.html')

# from flask import flash, redirect, render_template

@main.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            return redirect('/index')
        else:
            flash('Invalid credentials!')
            return redirect('/login')
    return render_template('login.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # NOTE: In production, store this data in a database and hash passwords.
        flash('Registration successful! Please login.')
        return redirect('/login')
    return render_template('register.html')