#include <iostream>
#include <cstdlib>
#include <ctime>

int main(){
    srand(time(NULL));
    int n;
    std::cin >> n;
    double *data = new double[n*n];
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            data[j + i * n] = rand();
        }
    }
    std::cout << n << '\n';
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            std::cout << data[j + i * n] << ' ';
        }
        std::cout << '\n';
    }
}