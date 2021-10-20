

#include <iostream>
#include <random>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <chrono>
using namespace std;

#define VECTOR_SIZE 256

int print_usage_and_exit()
{
    cout << "[USAGE] ./a.out [num_of_vectors(default = 256)]" << endl;
    return 0;
}

void create_data_set(unsigned int n_vecs,
                     vector<float> d_vecs[],
                     vector<float>& q_vec)
{
    std::default_random_engine engine;
    std::uniform_real_distribution<float> dist(10, 25); // range 1 - 5
    srand(time(0));

    // create random normalized data vectors
    for (unsigned int i=0; i < n_vecs; i++)
    {
        float norm = 0.0;
        for (unsigned int j=0; j < VECTOR_SIZE; j++)
        {
            float element = dist(engine) * ((rand() % 2) ? 1 : -1);
            d_vecs[i].push_back(element);
            norm += element * element;
        }
        // normalize
        for (unsigned int j=0; j < VECTOR_SIZE; j++)
        {
            d_vecs[i][j] = d_vecs[i][j] / sqrt(norm);
        }
    }

    // create a random normalized query vector
    float norm = 0.0;
    for (unsigned int k=0; k<VECTOR_SIZE; k++)
    {
        float element = dist(engine) * ((rand() % 2) ? 1 : -1);
        q_vec.push_back(element);
        norm += element * element;

    }
    // normalize
    for (unsigned int k=0; k<VECTOR_SIZE; k++)
    {
        q_vec[k] = q_vec[k] / sqrt(norm);
    }
}

unsigned int compute_cosine_similarity(unsigned int n_vecs,
                                       vector<float> d_vecs[],
                                       vector<float>& q_vec)
{
    // computing the cosine similarity values for each vector with the query vector
    // return the index of the vector close enough with the query vector
    float cosine_value = 0.0;
    unsigned index = 0;
    #pragma omp parallel for
    for (unsigned int i=0; i < n_vecs; i++)
    {
        float dot_product = 0.0;
        for (unsigned int j=0; j < VECTOR_SIZE; j++)
        {
            dot_product += q_vec[j] * d_vecs[i][j];
        }

        if (dot_product > 0 && cosine_value < dot_product)
        {
            cosine_value = dot_product;
            index = i;
        }
    }
    cout << "[" << index <<  "] : max cosine value: " << cosine_value << endl;
    return index;
}

int main(int argc, char* argv[])
{
    if (argc == 2 && argv[1] == "-h")
    {
        print_usage_and_exit();
    }

    // default values
    unsigned int num_vectors = VECTOR_SIZE;

    if (argc > 2)
    {
        print_usage_and_exit();
    }
    else if (argc == 2)
    {
        num_vectors = atoi(argv[1]);
    }

    vector<float>* data_vectors = new vector<float>[num_vectors];
    vector<float> query_vector;

    create_data_set(num_vectors,
                    data_vectors,
                    query_vector);

    unsigned int vector_index = compute_cosine_similarity(num_vectors,
                                                          data_vectors,
                                                          query_vector);
    return 0;
}
