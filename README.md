# Optimisation d'une formule de calcul par multiThreads & Vectorisation :
## Introduction
Nous examinerons les différentes méthodes et niveaux de parallélisation, à savoir la vectorisation et le multithreading. 

Nous débuterons par évaluer le temps nécessaire à l'exécution du calcul de la distance entre deux vecteurs, chacun composé de N éléments. Ensuite, nous chercherons à améliorer ce temps d'exécution en explorant les techniques de vectorisation, de multithreading, ainsi qu'en combinant les deux pour déterminer si cela nous permet d'obtenir un temps optimal. Enfin, nous comparerons ces approches aux méthodes précédentes.

Dans notre scénario, nous travaillerons avec des registres de 256 bits (SSE 256), ce qui signifie que nos vecteurs auront 8 éléments, chacun codé sur 32 bits. Nous tiendrons également compte du cas où le nombre d'éléments n'est pas un multiple de 8 et où les éléments ne sont pas alignés en mémoire.

Afin de faciliter la comparaison des temps d'exécution, nous maintiendrons le nombre d'éléments fixé à $1024^2$.
## Comment compiler & Exécuter : 
### Pour les gens ( chanceux ) qui sont sur la distribution Ubuntu ou Fedora : 
Vous pouvez utiliser gcc pour la compilation comme suit : 

```python
 gcc -o main main.c -lpthread -lm -mavx               # La compilation
./main                                                # l'exécution
```

### Pour les gens ( moins chanceux ) qui sont sur Windows :
Il vous faut installer un compilateur adéquat haha.

