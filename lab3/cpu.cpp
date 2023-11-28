#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>

using namespace std;

typedef struct uchar4{
    int x;
    int y;
    int z;
    int w;
} uchar4;


typedef struct int2{
    int x;
    int y;
} int2;

double avg[32][3];
double cov[32][3][3];
double inverse_cov[32][3][3];
double det[32];

double calc_jc(uchar4 pixel, int sample_id)
{
    double jc = 0.0;
    double diff[3], transposed[3];
    for (int i = 0; i < 3; ++i) 
    {
        diff[i] = 0.0;
        transposed[i] = 0.0;
    }
    diff[0] = pixel.x - avg[sample_id][0];
    diff[1] = pixel.y - avg[sample_id][1];
    diff[2] = pixel.z - avg[sample_id][2];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
            transposed[i] += -diff[j] * inverse_cov[sample_id][j][i];
        jc += transposed[i] * diff[i];
    }
    jc -= std::log(std::abs(det[sample_id]));
    return jc;
}

int classify(uchar4 pixel, int nc)
{
    int num_class = 0;
    double max_elem = calc_jc(pixel, num_class);
    for (int i = 1; i < nc; ++i)
    {
        double elem = calc_jc(pixel, i);
        if (elem > max_elem)
        {
            max_elem = elem;
            num_class = i;
        }
    }
    return num_class;
}

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    string input_file, output_file;
    vector<vector<int2>> samples_coords;
    uchar4 *data;
    int w, h, nc;
    cin >> input_file >> output_file >> nc;
    samples_coords.resize(nc);

    ifstream fd0(input_file, std::ios::in | std::ios::binary);
    if (fd0.is_open())
    {
        fd0.read((char *)&w, sizeof(w));
        fd0.read((char *)&h, sizeof(h));
        data = new uchar4[w * h];
        fd0.read((char *)data, w * h * sizeof(data[0]));
        fd0.close();
    }
    else
        return 1;

    for (int i = 0; i < nc; ++i)
    {
        int np;
        std::cin >> np;
        samples_coords[i].resize(np);
        for (int j = 0; j < np; ++j)
            std::cin >> samples_coords[i][j].x >> samples_coords[i][j].y;
    }

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 3; ++j)
            avg[i][j] = 0.0;

    for (int i = 0; i < nc; ++i)
    {
        int np = samples_coords[i].size();
        for (int j = 0; j < np; ++j)
        {
            int x = samples_coords[i][j].x;
            int y = samples_coords[i][j].y;
            uchar4 curr_pixel = data[x + y * w];
            avg[i][0] += curr_pixel.x;
            avg[i][1] += curr_pixel.y;
            avg[i][2] += curr_pixel.z;
        }
        for (int k = 0; k < 3; ++k)
            avg[i][k] /= np;
    }

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                cov[i][j][k] = 0.0;

    for (int i = 0; i < nc; ++i)
    {
        int np =  samples_coords[i].size();
        for (int j = 0; j < np; ++j)
        {
            double diff[3];
            int x = samples_coords[i][j].x;
            int y = samples_coords[i][j].y;
            uchar4 curr_pixel = data[x + y * w];
            diff[0] = curr_pixel.x - avg[i][0];
            diff[1] = curr_pixel.y - avg[i][1];
            diff[2] = curr_pixel.z - avg[i][2];

            for (int k = 0; k < 3; ++k)
                for (int m = 0; m < 3; ++m)
                    cov[i][k][m] += diff[k] * diff[m];
        }

        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                cov[i][j][k] /= np - 1;
    }

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                inverse_cov[i][j][k] = 0.0;

    for (int i = 0; i < nc; ++i)
    {
        double d = 0;
        for (int j = 0; j < 3; ++j)
            d += cov[i][0][j] * (cov[i][1][(j + 1) % 3] * cov[i][2][(j + 2) % 3] - cov[i][1][(j + 2) % 3] * cov[i][2][(j + 1) % 3]);
        inverse_cov[i][0][0] = (cov[i][1][1] * cov[i][2][2] - cov[i][2][1] * cov[i][1][2]) / d;
        inverse_cov[i][0][1] = (cov[i][0][2] * cov[i][2][1] - cov[i][0][1] * cov[i][2][2]) / d;
        inverse_cov[i][0][2] = (cov[i][0][1] * cov[i][1][2] - cov[i][0][2] * cov[i][1][1]) / d;
        inverse_cov[i][1][0] = (cov[i][1][2] * cov[i][2][0] - cov[i][1][0] * cov[i][2][2]) / d;
        inverse_cov[i][1][1] = (cov[i][0][0] * cov[i][2][2] - cov[i][0][2] * cov[i][2][0]) / d;
        inverse_cov[i][1][2] = (cov[i][1][0] * cov[i][0][2] - cov[i][0][0] * cov[i][1][2]) / d;
        inverse_cov[i][2][0] = (cov[i][1][0] * cov[i][2][1] - cov[i][2][0] * cov[i][1][1]) / d;
        inverse_cov[i][2][1] = (cov[i][2][0] * cov[i][0][1] - cov[i][0][0] * cov[i][2][1]) / d;
        inverse_cov[i][2][2] = (cov[i][0][0] * cov[i][1][1] - cov[i][1][0] * cov[i][0][1]) / d;
        det[i] = d;
    }

    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < w * h; ++i){
        data[i].w = classify(data[i], nc);
    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    ofstream fd1(output_file, std::ios::out | std::ios::binary);
    if (fd1.is_open())
    {
        fd1.write((char *)&w, sizeof(w));
        fd1.write((char *)&h, sizeof(h));
        fd1.write((char *)data, w * h * sizeof(data[0]));
        fd1.close();
    }
    else
        return 2;
    
    return 0;
}