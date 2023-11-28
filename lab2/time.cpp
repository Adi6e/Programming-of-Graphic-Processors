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

pair<int, int> find_coords(int x, int y, int w, int h){
    int y_new = max(min(y, h), 1);
    int x_new = max(min(x, w), 1);
    return make_pair(y_new, x_new);
}

int main()
{
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    string input_file, output_file;
    uchar4 *data, *out;
    int w, h;
    cin >> input_file >> output_file;

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

    out = new uchar4[w * h];

    auto t1 = high_resolution_clock::now();

    for (int y = 0; y < h; ++y){
        for (int x = 0; x < w; ++x){
            pair<int, int> first = find_coords(x, y, w, h);
            pair<int, int> second = find_coords(x + 1, y + 1, w, h);
            pair<int, int> third = find_coords(x + 1, y, w, h);
            pair<int, int> fourth = find_coords(x, y + 1, w, h);
            uchar4 p1 = data[first.first * w + first.second];
            uchar4 p2 = data[second.first * w + second.second];
            uchar4 p3 = data[third.first * w + third.second];
            uchar4 p4 = data[fourth.first * w + fourth.second];
            float Y1 = 0.299 * p1.x + 0.587 * p1.y + 0.114 * p1.z;
			float Y2 = 0.299 * p2.x + 0.587 * p2.y + 0.114 * p2.z;
			float Y3 = 0.299 * p3.x + 0.587 * p3.y + 0.114 * p3.z;
			float Y4 = 0.299 * p4.x + 0.587 * p4.y + 0.114 * p4.z;
			float gx = Y2 - Y1;
			float gy = Y4 - Y3;
            int g = min((float)255, sqrt(gx * gx + gy * gy));
            out[y * w + x].x = g;
            out[y * w + x].y = g;
            out[y * w + x].z = g;
            out[y * w + x].w = p1.x;
        }
    }

    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";

    ofstream fd1(output_file, std::ios::out | std::ios::binary);
    if (fd1.is_open())
    {
        fd1.write((char *)&w, sizeof(w));
        fd1.write((char *)&h, sizeof(h));
        fd1.write((char *)out, w * h * sizeof(out[0]));
        fd1.close();
    }
    else
        return 2;
    
    return 0;
}