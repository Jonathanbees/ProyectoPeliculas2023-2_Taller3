from django.shortcuts import render
from movie.models import Movie

from dotenv import load_dotenv
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import os

_ = load_dotenv('../openAI.env')
openai.api_key  = os.environ['openAI_api_key']

def recommendation(request):
    rec = request.GET.get('req')
    if rec:
        emb = get_embedding(rec, engine='text-embedding-ada-002')
        movies = list(Movie.objects.all())
        sim = []
        for movie in movies:
            emb_binary = np.array(movie.emb).tobytes()
            rec_emb = list(np.frombuffer(emb_binary))
            sim.append(cosine_similarity(emb, rec_emb))
        sim = np.array(sim)
        idx = np.argmax(sim)
        
        lista_movies_puntajes = []

        for i in range(len(sim)):
            if sim[i] > 0.825:
                lista_movies_puntajes.append({'movie':movies[i] , 'puntaje':sim[i]})

        lista_ordenada_movie_puntajes = sorted(lista_movies_puntajes, key=lambda x: x["puntaje"], reverse=True)

        lista_ordenada_movies = []
        for i in lista_ordenada_movie_puntajes:
            lista_ordenada_movies.append(i['movie'])

        return render(request, 'recommendation.html', {'movies': lista_ordenada_movies})


        '''
        recommended_movie = []

        for i in range(len(sim)):
            if sim[i] > 0.825:
                recommended_movie.append(movies[i])


        recommended_movie = sorted(recommended_movie, reverse=True)
        return render(request, 'recommendation.html', {'movies': recommended_movie})
        '''
        

    else:
        return render(request, 'recommendation.html')