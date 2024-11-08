#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MAX_ITER 100
#define MAX_DIM 10

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
    // Randomly select the first centroid
    centroids[0] = data[rand() % n].data;

    for (int i = 1; i < k; i++) {
        double total_dist = 0.0;
        double *distances = malloc(n * sizeof(double));

        // Compute distances from the nearest centroid
        for (int j = 0; j < n; j++) {
            double min_dist = INFINITY;
            for (int m = 0; m < i; m++) {
                double dist = euclidean_distance(data[j].data, centroids[m], dim);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            distances[j] = min_dist;
            total_dist += min_dist;
        }

        // Select the next centroid based on a weighted probability
        double r = ((double)rand() / (RAND_MAX)) * total_dist;
        double cumulative = 0.0;

        for (int j = 0; j < n; j++) {
            cumulative += distances[j];
            if (cumulative >= r) {
                centroids[i] = data[j].data;
                break;
            }
        }

        free(distances);
    }
}

int kmeans(DataPoint *data, int n, int k, int dim, int *labels) {
    double **centroids = malloc(k * sizeof(double *));
    initialize_centroids(data, n, centroids, k, dim);

    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        int changed = 0;

        #pragma omp parallel for
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
            double *new_centroid = calloc(dim, sizeof(double));
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

    for (int j = 0; j < k; j++) {
        free(centroids[j]);
    }
    free(centroids);
    return iter;
}

void load_data(const char *filename, DataPoint **data, int *n, int *dim) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("File opening failed");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d", n, dim);
    *data = malloc(*n * sizeof(DataPoint));
    for (int i = 0; i < *n; i++) {
        (*data)[i].data = malloc(*dim * sizeof(double));
        for (int j = 0; j < *dim; j++) {
            fscanf(file, "%lf", &(*data)[i].data[j]);
        }
        (*data)[i].label = -1; // Initialize label
    }

    fclose(file);
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
    int k = 5; // Number of clusters
    int *labels;

    srand(time(NULL));
    load_data("data.txt", &data, &n, &dim);
    labels = malloc(n * sizeof(int));
    
    clock_t start = clock();
    int iterations = kmeans(data, n, k, dim, labels);
    clock_t end = clock();
    
    printf("K-means completed in %d iterations\n", iterations);
    printf("Parallel execution time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // Cleanup
    free(labels);
    free_data(data, n);
    
    return 0;
}


