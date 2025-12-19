GPA NeuralNet / GPA_Predictor.py 
by saatwik
A client-side React application that mimics Python ML libraries (NumPy, Scikit-Learn) to predict GPA trends and cluster subject performance.

 Overview

This project is a React-based academic dashboard designed for students. It moves beyond simple GPA calculation by implementing lightweight machine learning algorithms directly in the browser (no Python backend required).

It features a Terminal

 Key Features

MiniNumPy Engine: A custom JavaScript implementation of vector operations (dot product, mean, std dev, matrix math).

Client-Side OLS Regression: Predicts future semester GPA based on historical trends using Ordinary Least Squares.

K-Means Clustering: Unsupervised learning algorithm ($k=3$) to group subjects into "Strong", "Average", and "Weak" performance clusters.

Terminal UI: Data entry and logs styled as a developer console.

 Tech Stack

Framework: React (Vite)

Styling: Tailwind CSS

Icons: Lucide React

Math: Custom MiniNumPy and MiniSklearn classes.

Installation & Setup

Clone the repository

git clone [https://github.com/saatu10/gpa-predictor.git](https://github.com/yourusername/gpa-predictor.git)
cd gpa-predictor


Install dependencies

npm install
npm install lucide-react


Run the development server

npm run dev


How it Works

The app runs entirely in the browser. When you input course data:

MiniLinearRegression.fit(X, y) calculates the slope and intercept of your semester GPAs.

MiniKMeans.fit(grades) iterates through centroids to cluster your courses.

Visualizations are rendered using pure HTML/CSS based on the computed stats.

 