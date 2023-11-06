#define _USE_MATH_DEFINES // Pour activer les constantes mathématiques, par exemple M_PI
#define _USE_AVX_DEFINES  // Pour activer les constantes et les fonctions AVX

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <pthread.h>
#include <sys/time.h>
#include <math.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <immintrin.h>

#define N (1024*1024)
#define NB_THREADS 4

/************* fonction qui gère le temps (pris dans le tp) ****************/
double now(){
    struct timeval t;
    double f_t;
    gettimeofday(&t, NULL);
    f_t = t.tv_usec;
    f_t = f_t / 1000000.0;
    f_t += t.tv_sec;
    return f_t;
}


/********  structure de contenant les données de chaque thread  **********/
typedef struct {
    float *U; // tableau
    float *V;
    int start;
    int end;
    double *partial_sum;
} ThreadData;

pthread_mutex_t lock; // variable globale

float U[N], V[N];
/*================================================ distance vectorielle : Sans verfication de N%8 ni Alignement  ===================*/
double vect_dist(float *U, float *V, int n) {
    __m256 mm_distance = _mm256_setzero_ps();  // variable qui va accumuler les sommes partielles
    double distance = 0.0;
    int numVectors = n / 8;  // Chaque vecteur AVX traite 8 éléments

    for (int i = 0; i < numVectors; i++) {
        // Charger 8 éléments de U et V dans des registres AVX
        __m256 u = _mm256_load_ps(U + (i * 8));
        __m256 v = _mm256_load_ps(V + (i * 8));
        // somme en une seule ligne pour éviter le stockage ( load & store ) des varaibles intermidiaires ( pénalisent la performance ).
        __m256 temp = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(u, u), _mm256_mul_ps(v, v)));
        mm_distance = _mm256_add_ps(mm_distance, temp);
    }
    // le calcul du résultat final
    for(int i = 0; i < 8; i++) {
        distance += mm_distance[i];
    }
    return distance;
}
/*====================================== Initialise le tableau avec une valeur aléatoire (pris dans un exemple du cours) ===================*/

void init(){
    for(unsigned int i = 0; i < N; i++){
        U[i] = (float)rand() / RAND_MAX / N*100;
        V[i] = (float)rand() / RAND_MAX / N*100;
    }
}

/******************************* DIST_SCALAIRE ***************************/
double dist(float *U, float *V, int n){
    double distance = 0.0;
    for(int i = 0; i < n; i++){
        distance += sqrt(U[i] * U[i] + V[i] * V[i]); // la formule de calcul de la distance donnée sur a feuille
    }
    return distance;
}

/*******************************    THREAD_DIST_SCALAIRE ***************************/
void *thread_scalaire(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    double sum = 0.0;
    for(int i = data->start; i <= data->end; i++) {
        sum += sqrt(data->U[i]*data->U[i] + data->V[i]*data->V[i]);
    }
    pthread_mutex_lock(&lock);
    *(data->partial_sum) += sum;
    pthread_mutex_unlock(&lock);
    pthread_exit(NULL);
}
/************************************************************************************/

/*******************************    THREAD_DIST_VECTORIEL ***************************/
void *thread_dist_vect(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    double sum = 0.0;
    int n = (data->end-data->start)+1 ;
    if(n<8) {
        for(int i = data->start; i <= data->end; i++) {
            sum += sqrt(data->U[i]*data->U[i] + data->V[i]*data->V[i]);
        }
        pthread_mutex_lock(&lock);
        *(data->partial_sum) += sum;
        pthread_mutex_unlock(&lock);
        pthread_exit(NULL);
    }else{
        int nn = n-n%8;
        __m256 mm_distance = _mm256_setzero_ps();

        for(int i = 0; i < nn; i += 8) {
            __m256 mm_u = _mm256_loadu_ps(&U[data->start + i]);
            __m256 mm_v = _mm256_loadu_ps(&V[data->start + i]);
            __m256 temp = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(mm_u, mm_u), _mm256_mul_ps(mm_v, mm_v)));
            mm_distance = _mm256_add_ps(mm_distance, temp);
        }

        float distance_values[8];
        _mm256_storeu_ps(distance_values, mm_distance);
        double distance = 0.0;

        for(int i = 0; i < 8; i++) {
            distance += distance_values[i];
        }

        for(int i = nn; i < n; i++) {
            distance += sqrt(U[i] * U[i] + V[i] * V[i]);
        }

        pthread_mutex_lock(&lock);
        *(data->partial_sum) += distance;
        pthread_mutex_unlock(&lock);
        pthread_exit(NULL);
    }

}

/************************** distance vectorielle generale *********************************/
double vect_dist_gle(float *U, float *V, int n) {
    double distance = 0.0;
    __m256 mm_distance = _mm256_setzero_ps();
    int n_r = n-n%8;

    for(int i = 0; i < n_r; i += 8) {
        __m256 UU = _mm256_loadu_ps(&U[i]);
        __m256 VV = _mm256_loadu_ps(&V[i]);
        __m256 temp = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(UU, UU), _mm256_mul_ps(VV, VV)));
        mm_distance = _mm256_add_ps(mm_distance, temp);
    }
    // somme totale
    for(int i = 0; i < 8; i++) {
        distance += mm_distance[i];
    }
    //  le reste des élément < 8
    for(int i = n_r; i < n; i++) {
        distance += sqrt(U[i] * U[i] + V[i] * V[i]);
    }
    return distance;
}


/*****************************  Distance multithreadé au choix  ************************************/
double distPar(float *U, float *V, int n, int nb_threads, int mode) {
    double result = 0.0;
    if(mode == 1) {
        pthread_t threads[nb_threads];
        ThreadData thread_data[nb_threads];
        double partial_sums[nb_threads];
        // initialisation du tableau des sommes partielles
        for(int i = 0; i < nb_threads; i++) {
            partial_sums[i] = 0.0;
        }
        pthread_mutex_init(&lock, NULL);
        int block_size = n / nb_threads; // nombre de block pour chaque thread
        for(int i = 0; i < nb_threads; i++) {
            thread_data[i].U = U;
            thread_data[i].V = V;
            thread_data[i].start = i * block_size;
            thread_data[i].end = (i == nb_threads - 1) ? n - 1 : (i + 1) * block_size - 1;
            thread_data[i].partial_sum = &partial_sums[i];
            pthread_create(&threads[i], NULL, thread_dist_vect, (void *)&thread_data[i]);
        }
        for(int i = 0; i < nb_threads; i++) {
            pthread_join(threads[i], NULL);
            result += partial_sums[i];
        }
        pthread_mutex_destroy(&lock);
    }
    else if(mode == 0) {
        pthread_t threads[nb_threads];
        ThreadData thread_data[nb_threads];
        double partial_sums[nb_threads];
        for(int i = 0; i < nb_threads; i++) {
            partial_sums[i] = 0.0;
        }
        pthread_mutex_init(&lock, NULL);
        int block_size = n / nb_threads;
        for(int i = 0; i < nb_threads; i++) {
            thread_data[i].U = U;
            thread_data[i].V = V;
            thread_data[i].start = i * block_size;
            thread_data[i].end = (i == nb_threads - 1) ? n - 1 : (i + 1) * block_size - 1;
            thread_data[i].partial_sum = &partial_sums[i];
            pthread_create(&threads[i], NULL, thread_scalaire, (void *)&thread_data[i]);
        }
        for(int i = 0; i < nb_threads; i++) {
            pthread_join(threads[i], NULL);
            result += partial_sums[i];
        }
        pthread_mutex_destroy(&lock);
    }
    return result;
}

int main() {

    init();
    double t;

    printf("\n Exécution pour :  %d threads | %d élément total |  ==> %d vectors de 8 éléments | %d éléments restant.\n",NB_THREADS,N,N/8,N%8);
    printf("-------------------------------------------------------------------------------------------------------------\n");
    // Calcul de la distance avec la fonction dist
    t = now();
    double result_dist = dist(U, V, N);
    t = now()-t;
    printf("Distance (dist):                                    %f  |  temps :     %f\n", result_dist,t);
    printf("-------------------------------------------------------------------------------------\n");

    // Calcul de la distance vectorielle avec la fonction qui ne prend pas en considération l'alignement et la division par
    t = now();
    result_dist = vect_dist(U, V, N);
    t = now()-t;
    printf("Distance (dist_vect: sans align sans N%%8!=0?):      %f  |  temps :     %f\n", result_dist,t);
    printf("-------------------------------------------------------------------------------------\n");


    // Calcul de la distance vectorielle générale prenant en considération les cas où N%8!=0 et l'alignement
    t = now();
    result_dist = vect_dist_gle(U, V, N);
    t = now()-t;
    printf("Distance (dist_vect_gle :  avec align & N%%8!=0?):   %f  |  temps :     %f\n", result_dist,t);
    printf("-------------------------------------------------------------------------------------\n");


    // Calcul de la distance multithreadée  scalaire ( mode 0 )

    t = now();
    result_dist = distPar(U, V, N, NB_THREADS, 0);
    t = now()-t;
    printf("Distance (dist_sclaire mode 0):                     %f  |  temps :     %f\n", result_dist,t);
    printf("-------------------------------------------------------------------------------------\n");

    // Calcul de la distance multithreadée  vectorielle  ( mode 0 )
    t = now();
    result_dist = distPar(U, V, N, NB_THREADS, 1);
    t = now()-t;
    printf("Distance (dist_vect_thread mode 1):                 %f  |  temps :     %f\n\n", result_dist,t);
    printf("-------------------------------------------------------------------------------------\n");

    return 0;
}