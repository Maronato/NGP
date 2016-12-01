from django.shortcuts import render
from django.http import JsonResponse
from learning.nmf import NMF
import numpy as np
import json

# Create your views here.


def index(request):
    return render(request, 'index.html')


def stats(request):
    return render(request, 'stats.html')


def info(request):
    return render(request, 'info.html')


def docs(request):
    return render(request, 'docs.html')


def example(request):
    return render(request, 'example.html')


def math(request):
    return render(request, 'math.html')


def fit_predict(request):
    raw = json.loads(request.POST['data'])
    R = int(request.POST['R'])
    eC = float(request.POST['eC'])
    matrix = []
    for row in raw:
        cleaned_row = [float(x[1]) for x in sorted(row.items(), key=lambda s: s[0]) if x[1].isnumeric()]
        matrix.append(cleaned_row)
    matrix = np.array(matrix)
    model = NMF()
    model.fit(matrix, R, eC=eC)
    return JsonResponse({
        'data': model.get_V().tolist(),
        'W': model.W.tolist(),
        'H': model.H.tolist()
        })
