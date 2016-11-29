from django.shortcuts import render, redirect
from django.http import JsonResponse
from learning.nmf import NMF
import numpy as np
from data import unicamp

# Create your views here.

def index(request):
    return render(request, 'index.html')

def stats(request):
    return render(request, 'stats.html')

def info(request):
    return render(request, 'info.html')

def docs(request):
    return render(request, 'docs.html')

def fit_predict(request):
    import json
    from operator import itemgetter
    raw = json.loads(request.POST['data'])
    R = int(request.POST['R'])
    eC = float(request.POST['eC'])
    matrix = []
    for row in raw:
        l = [float(x[1]) for x in sorted(row.items(), key=lambda s: s[0])]
        matrix.append(l)
    matrix = np.array(matrix)
    model = NMF()
    model.fit(matrix, R, eC=eC)
    return JsonResponse({
        'data': model.get_V().tolist(),
        'W': model.W.tolist(),
        'H': model.H.tolist()
        })
