from django.shortcuts import render
from django.http import JsonResponse
from learning.nmf import NMF
import numpy as np
import json
from data import unicamp

# Create your views here.


def index(request):
    # Render the index page
    return render(request, 'index.html')


def stats(request):
    # Render the stats page

    # Load the dataset and table head
    u = unicamp.load()
    data = u.data.T
    head = ['Calculus 1', 'Calculus 2', 'Calculus 3', 'Calculus 4', 'Physics 1', 'Physics 2', 'Physics 3', 'Physics 4', 'Programming 1', 'Programming 2', 'Analytic Geometry', 'Linear Algebra']

    # Generate some statistics from the students in the dataset
    course_stats = []
    for count, row in enumerate(data):
        course = head[count]
        nonzero = np.count_nonzero(row)
        mean = round(np.mean(row[np.nonzero(row)]), 2)
        median = np.median(row[np.nonzero(row)])
        std = round(np.std(row[np.nonzero(row)]), 2)
        minv = np.amin(row[np.nonzero(row)])
        maxv = np.amax(row[np.nonzero(row)])
        row_list = [course, nonzero, mean, median, std, minv, maxv]
        course_stats.append(row_list)

    # Name those statistics
    stat_names = ["Course", "Nonzero", "Mean", "Median", "STD", "Min", "Max"]

    # Pass context variables to rendered page
    return render(request, 'stats.html', {"course_stats": course_stats, "stat_names": stat_names})


def example(request):
    # Render W and H page
    return render(request, 'example.html')


def math(request):
    # Render Desgin and Math page
    return render(request, 'math.html')


def fit_predict(request):
    # Server-side process of predicting tables

    # Get the POST data
    raw = json.loads(request.POST['data'])
    R = int(request.POST['R'])
    eC = float(request.POST['eC'])

    # Clean the inputs by converting the elements to floats and converting '?' into 0s
    matrix = []
    for row in raw:
        new_row = [x[1] for x in sorted(row.items(), key=lambda s: s[0]) if x[1].isnumeric() or x[1] == "?"]
        cleaned_row = []
        for item in new_row:
            if item == "?":
                item = float(0)
            else:
                item = float(item)
            cleaned_row.append(item)
        matrix.append(cleaned_row)

    # Convert to numpy array and fit
    matrix = np.array(matrix)
    model = NMF()
    model.fit(matrix, R, eC=eC, alg=1)

    # Return the Json response
    return JsonResponse({
        'data': model.get_V().tolist(),
        'W': model.W.tolist(),
        'H': model.H.tolist()
        })
