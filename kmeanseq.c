#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_ITER 200
#define MAX_DIM 5
#define N 4000
#define K 10

typedef struct {
    double *data;
    int label;
} DataPoint;

double euclidean_distance(double *a, double *b, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

void initialize_centroids(DataPoint *data, int n, double **centroids, int k, int dim) {
    centroids[0] = data[rand() % n].data;
    for (int i = 1; i < k; i++) {
        centroids[i] = data[rand() % n].data; // Randomly choose centroids
    }
}

int kmeans(DataPoint *data, int n, int k, int dim, int *labels) {
    double **centroids = (double **)malloc(k * sizeof(double *));
    initialize_centroids(data, n, centroids, k, dim);

    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        int changed = 0;

        for (int i = 0; i < n; i++) {
            double min_dist = INFINITY;
            int best_label = -1;
            for (int j = 0; j < k; j++) {
                double dist = euclidean_distance(data[i].data, centroids[j], dim);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_label = j;
                }
            }
            if (labels[i] != best_label) {
                labels[i] = best_label;
                changed = 1;
            }
        }

        // Update centroids
        for (int j = 0; j < k; j++) {
            double *new_centroid = (double *)calloc(dim, sizeof(double));
            int count = 0;

            for (int i = 0; i < n; i++) {
                if (labels[i] == j) {
                    for (int d = 0; d < dim; d++) {
                        new_centroid[d] += data[i].data[d];
                    }
                    count++;
                }
            }

            if (count > 0) {
                for (int d = 0; d < dim; d++) {
                    new_centroid[d] /= count;
                }
                centroids[j] = new_centroid;
            } else {
                free(new_centroid);
            }
        }

        if (!changed) {
            break; // Stop if no points changed clusters
        }
    }

    free(centroids);
    return iter;
}

void load_data(DataPoint **data, int *n, int *dim) {
    *n = N;  // Set number of data points
    *dim = MAX_DIM;  // Set number of dimensions
    *data = (DataPoint *)malloc(*n * sizeof(DataPoint));
    for (int i = 0; i < *n; i++) {
        (*data)[i].data = (double *)malloc(*dim * sizeof(double));
        for (int j = 0; j < *dim; j++) {
            (*data)[i].data[j] = rand() % 100;  // Random data points
        }
        (*data)[i].label = -1; // Initialize label
    }
}

void free_data(DataPoint *data, int n) {
    for (int i = 0; i < n; i++) {
        free(data[i].data);
    }
    free(data);
}

int main() {
    DataPoint *data;
    int n, dim;
    int *labels = (int *)malloc(N * sizeof(int));

    srand(time(NULL));
    load_data(&data, &n, &dim);

    clock_t start = clock();
    int iterations = kmeans(data, n, K, dim, labels);
    clock_t end = clock();
    
    printf("K-means completed in %d iterations\n", iterations);
    printf("Sequential execution time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Cleanup
    free(labels);
    free_data(data, n);
    
    return 0;
}
