#include <iostream>
#include <cmath>

using namespace std;

#define CSC(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "ERROR is %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(0); \
    } \
} while(0)

// вспомогательные структуры
class Vector{
    public:
        double x;
        double y;
        double z;

    __host__ __device__ Vector() {}
    __host__ __device__ Vector(double new_x, double new_y, double new_z){
        this->x = new_x;
        this->y = new_y;
        this->z = new_z;
    }
};

__host__ __device__ Vector operator+(Vector vec1, Vector vec2){
    Vector summ;
    summ.x = vec1.x + vec2.x;
    summ.y = vec1.y + vec2.y;
    summ.z = vec1.z + vec2.z;
    return summ;
}

__host__ __device__ Vector operator-(Vector vec1, Vector vec2){
    Vector diff;
    diff.x = vec1.x - vec2.x;
    diff.y = vec1.y - vec2.y;
    diff.z = vec1.z - vec2.z;
    return diff;
}

__host__ __device__ Vector operator*(Vector vec1, double vec2){
    Vector mult;
    mult.x = vec1.x * vec2;
    mult.y = vec1.y * vec2;
    mult.z = vec1.z * vec2;
    return mult;
}


class polygon{
    public:
        Vector a;
        Vector b;
        Vector c;
        uchar4 clr;

    __host__ __device__ polygon() {}
    __host__ __device__ polygon(Vector new_a, Vector new_b, Vector new_c, uchar4 new_color){
        this->a = new_a;
        this->b = new_b;
        this->c = new_c;
        this->clr = new_color;
    }
};

__host__ __device__ double make_dot(Vector vec1, Vector vec2){
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__host__ __device__ Vector make_prod(Vector vec1, Vector vec2){
    Vector prod;
    prod.x = vec1.y * vec2.z - vec1.z * vec2.y;
    prod.y = vec1.z * vec2.x - vec1.x * vec2.z;
    prod.z = vec1.x * vec2.y - vec1.y * vec2.x;
    return prod;
}

__host__ __device__ Vector normalize(Vector v){
    double l = sqrt(make_dot(v, v));
    Vector norm;
    norm.x = v.x / l;
    norm.y = v.y / l;
    norm.z = v.z / l;
    return norm;
}

__host__ __device__ Vector mult(Vector vec1, Vector vec2, Vector vec3, Vector v){
    Vector mult;
    mult.x = vec1.x * v.x + vec2.x * v.y + vec3.x * v.z;
    mult.y = vec1.y * v.x + vec2.y * v.y + vec3.y * v.z;
    mult.z = vec1.z * v.x + vec2.z * v.y + vec3.z * v.z;
    return mult;
}

__host__ __device__ uchar4 create_ray(polygon* polygs_arr, int polygs_amnt, Vector place, Vector orient, Vector light_plc, uchar4 light_clr){
    int min_idx = polygs_amnt;
    double min_vec2_dot;
    for (int polyg_ind = 0; polyg_ind < polygs_amnt; ++polyg_ind) {
        Vector vec1 = polygs_arr[polyg_ind].b - polygs_arr[polyg_ind].a;
        Vector vec2 = polygs_arr[polyg_ind].c - polygs_arr[polyg_ind].a;
        Vector prod_res = make_prod(orient, vec2);
        double dot_res = make_dot(prod_res, vec1);
        if (fabs(dot_res) < 1e-10)
            continue;
        Vector diff = place - polygs_arr[polyg_ind].a;
        double diff_dot = make_dot(prod_res, diff) * (1 / dot_res);
        if (!(diff_dot >= 0.0 && diff_dot <= 1.0))
            continue;
        Vector diff_prod = make_prod(diff, vec1);
        double orient_dot = make_dot(diff_prod, orient) * (1 / dot_res);
        if (!(orient_dot >= 0.0 && (orient_dot + diff_dot) <= 1.0))
            continue;
        double vec2_dot = make_dot(diff_prod, vec2) / dot_res; 
        if (!(vec2_dot >= 0.0))
            continue;
        if (min_idx == polygs_amnt || min_vec2_dot >= vec2_dot){
            min_idx = polyg_ind;
            min_vec2_dot = vec2_dot;
        }
    }
    if (min_idx == polygs_amnt)
        return make_uchar4(0, 0, 0, 255);

    Vector new_place = orient * min_vec2_dot + place;
    Vector new_orient = light_plc - new_place;
    double dist = sqrt(make_dot(new_orient, new_orient));
    new_orient = normalize(new_orient);
    for (int polyg_ind = 0; polyg_ind < polygs_amnt; ++polyg_ind) {
        Vector vec1 = polygs_arr[polyg_ind].b - polygs_arr[polyg_ind].a;
        Vector vec2 = polygs_arr[polyg_ind].c - polygs_arr[polyg_ind].a;
        Vector prod_res = make_prod(new_orient, vec2);
        double dot_res = make_dot(prod_res, vec1);
        if (fabs(dot_res) < 1e-10)
            continue;
        Vector diff = new_place - polygs_arr[polyg_ind].a;
        double diff_dot = make_dot(prod_res, diff) / dot_res;
        if (!(diff_dot >= 0.0 && diff_dot <= 1.0))
            continue;
        Vector diff_prod = make_prod(diff, vec1);
        double new_orient_dot = make_dot(diff_prod, new_orient) / dot_res;
        if (!(new_orient_dot >= 0.0 && (new_orient_dot + diff_dot) <= 1.0))
            continue;
        double vec2_dot = make_dot(diff_prod, vec2) / dot_res; 
        if (vec2_dot > 0.0 && dist >= vec2_dot && polyg_ind != min_idx)
            return make_uchar4(0, 0, 0, 255);
    }

    unsigned char res_x = polygs_arr[min_idx].clr.x * light_clr.x;
    unsigned char res_y = polygs_arr[min_idx].clr.y * light_clr.y;
    unsigned char res_z = polygs_arr[min_idx].clr.z * light_clr.z;
    return make_uchar4(res_x, res_y, res_z, 255);
}

__host__ __device__ void cpu_render(Vector cam_place, Vector cam_view, int w, int h, double ang, uchar4* img, Vector light_plc, uchar4 light_clr, polygon* polygs_arr, int polygs_amnt){
    Vector old_basis;
    old_basis.x = 0.0;
    old_basis.y = 0.0;
    old_basis.z = 1.0;
    double new_w = 2.0 / (w - 1.0);
    double new_h = 2.0 / (h - 1.0);
    double z = 1.0 / tan(ang * M_PI / 360.0);
    Vector basis_z = normalize(cam_view - cam_place);
    Vector basis_x = normalize(make_prod(basis_z, old_basis));
    Vector basis_y = normalize(make_prod(basis_x, basis_z));

    for (int dw = 0; dw < w; ++dw) {
        for (int dh = 0; dh < h; ++dh) {
            Vector vec = Vector(new_w * dw - 1.0, (-1.0 + new_h * dh) * h / w, z);
            Vector orient = mult(basis_x, basis_y, basis_z, vec);
            img[(h - dh - 1) * w + dw] = create_ray(polygs_arr, polygs_amnt, cam_place, normalize(orient), light_plc, light_clr);
        }
    }
}

__global__ void kernel_render(polygon* polygs_arr, int polygs_amnt, uchar4* img, Vector light_plc, uchar4 light_clr, Vector cam_place, Vector cam_view, int w, int h, double ang){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    Vector old_basis;
    old_basis.x = 0.0;
    old_basis.y = 0.0;
    old_basis.z = 1.0;
    double new_w = 2.0 / (w - 1.0);
    double new_h = 2.0 / (h - 1.0);
    double z = 1.0 / tan(ang * M_PI / 360.0);
    Vector basis_z = normalize(cam_view - cam_place);
    Vector basis_x = normalize(make_prod(basis_z, old_basis));
    Vector basis_y = normalize(make_prod(basis_x, basis_z));

    for (int ix = idx; ix < w; ix += offsetx){
        for (int jy = idy; jy < h; jy += offsety){
            Vector v = Vector(new_w * ix - 1.0, (-1.0 + new_h * jy) * h / w, z);
            Vector orient = mult(basis_x, basis_y, basis_z, v);
            img[(h - jy - 1) * w + ix] = create_ray(polygs_arr, polygs_amnt, cam_place, normalize(orient), light_plc, light_clr);
        }
    }
}

__host__ __device__ void cpu_make_ssaa(int rays_num, uchar4* img, uchar4* ssaa_img, int w, int h){
    for (int dw = 0; dw < w; ++dw){
        for (int dy = 0; dy < h; ++dy){
            uint4 pxl_vals = make_uint4(0, 0, 0, 0);
            for (int x = 0; x < rays_num; ++x){
                for (int y = 0; y < rays_num; ++y){
                    uchar4 curr_pixel = img[w * rays_num * (rays_num * dy + y) + (rays_num * dw + x)];
                    pxl_vals.x += curr_pixel.x;
                    pxl_vals.y += curr_pixel.y;
                    pxl_vals.z += curr_pixel.z;
                }
            }
            int pixel_rays = rays_num * rays_num;
            ssaa_img[dy * w + dw] = make_uchar4(pxl_vals.x / pixel_rays, pxl_vals.y / pixel_rays, pxl_vals.z / pixel_rays, 255);
        }
    }
}

__global__ void kernel_ssaa(int rays_num, uchar4* img, uchar4* ssaa_img, int w, int h){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int ix = idx; ix < w; ix += offsetx){
        for (int jy = idy; jy < h; jy += offsety){
            uint4 pxl_vals = make_uint4(0, 0, 0, 0);
            for (int x = 0; x < rays_num; ++x) {
                for (int y = 0; y < rays_num; ++y){
                    uchar4 curr_pixel = img[w * rays_num * (rays_num * jy + y) + (rays_num * ix + x)];
                    pxl_vals.x += curr_pixel.x;
                    pxl_vals.y += curr_pixel.y;
                    pxl_vals.z += curr_pixel.z;
                }
            }
            int pixel_rays = rays_num * rays_num;
            ssaa_img[jy * w + ix] = make_uchar4(pxl_vals.x / pixel_rays, pxl_vals.y / pixel_rays, pxl_vals.z / pixel_rays, 255);
        }
    }
}

void make_scene(polygon* polygs_arr, int index_start, uchar4 clr, Vector vec1, Vector vec2, Vector vec3, Vector vec4){
    polygs_arr[index_start] = polygon(vec1, vec2, vec3, clr);
    polygs_arr[index_start + 1] = polygon(vec1, vec3, vec4, clr);
}

void make_hexahedron(polygon* polygs_arr, int index_start, uchar4 clr, Vector center, double radius){
    double a = 2 * radius / sqrt(3);
    Vector first_v = Vector(center.x - a / 2, center.y - a / 2, center.z - a / 2);
    Vector vertices[] = {
        Vector(first_v.x, first_v.y, first_v.z),
        Vector(first_v.x, first_v.y + a, first_v.z),
        Vector(first_v.x + a, first_v.y + a, first_v.z),
        Vector(first_v.x + a, first_v.y, first_v.z),
        Vector(first_v.x, first_v.y, first_v.z + a),
        Vector(first_v.x, first_v.y + a, first_v.z + a),
        Vector(first_v.x + a, first_v.y + a, first_v.z + a),
        Vector(first_v.x + a, first_v.y, first_v.z + a)
    };

    polygs_arr[index_start] = polygon(vertices[0], vertices[1], vertices[2], clr);
    polygs_arr[index_start + 1] = polygon(vertices[2], vertices[3], vertices[0], clr);
    polygs_arr[index_start + 2] = polygon(vertices[6], vertices[7], vertices[3], clr);
    polygs_arr[index_start + 3] = polygon(vertices[3], vertices[2], vertices[6], clr);
    polygs_arr[index_start + 4] = polygon(vertices[2], vertices[1], vertices[5], clr);
    polygs_arr[index_start + 5] = polygon(vertices[5], vertices[6], vertices[2], clr);
    polygs_arr[index_start + 6] = polygon(vertices[4], vertices[5], vertices[1], clr);
    polygs_arr[index_start + 7] = polygon(vertices[1], vertices[0], vertices[4], clr);
    polygs_arr[index_start + 8] = polygon(vertices[3], vertices[7], vertices[4], clr);
    polygs_arr[index_start + 9] = polygon(vertices[4], vertices[0], vertices[3], clr);
    polygs_arr[index_start + 10] = polygon(vertices[6], vertices[5], vertices[4], clr);
    polygs_arr[index_start + 11] = polygon(vertices[4], vertices[7], vertices[6], clr);
}

void make_dodecahedron(polygon* polygs_arr, int index_start, uchar4 clr, Vector center, double radius){
    double a = (1 + sqrt(5)) / 2;
    double b = 1 / a;
    Vector vertices[] = {
        Vector(-b, 0, a),
        Vector(b, 0, a), 
        Vector(-1, 1, 1), 
        Vector(1, 1, 1), 
        Vector(1, -1, 1), 
        Vector(-1, -1, 1), 
        Vector(0, -a, b), 
        Vector(0, a, b), 
        Vector(-a, -b, 0), 
        Vector(-a, b, 0), 
        Vector(a, b, 0), 
        Vector(a, -b, 0), 
        Vector(0, -a, -b), 
        Vector(0, a, -b), 
        Vector(1, 1, -1), 
        Vector(1, -1, -1), 
        Vector(-1, -1, -1), 
        Vector(-1, 1, -1), 
        Vector(b, 0, -a), 
        Vector(-b, 0, -a)
    };

    for (auto& v: vertices){
        v.x = v.x * radius / sqrt(3) + center.x;
        v.y = v.y * radius / sqrt(3) + center.y;
        v.z = v.z * radius / sqrt(3) + center.z;
    }

    polygs_arr[index_start] = polygon(vertices[4], vertices[0], vertices[6], clr);
    polygs_arr[index_start + 1] = polygon(vertices[0], vertices[5], vertices[6], clr);
    polygs_arr[index_start + 2] = polygon(vertices[0], vertices[4], vertices[1], clr);
    polygs_arr[index_start + 3] = polygon(vertices[0], vertices[3], vertices[7], clr);
    polygs_arr[index_start + 4] = polygon(vertices[2], vertices[0], vertices[7], clr);
    polygs_arr[index_start + 5] = polygon(vertices[0], vertices[1], vertices[3], clr);
    polygs_arr[index_start + 6] = polygon(vertices[10], vertices[1], vertices[11], clr);
    polygs_arr[index_start + 7] = polygon(vertices[3], vertices[1], vertices[10], clr);
    polygs_arr[index_start + 8] = polygon(vertices[1], vertices[4], vertices[11], clr);
    polygs_arr[index_start + 9] = polygon(vertices[5], vertices[0], vertices[8], clr);
    polygs_arr[index_start + 10] = polygon(vertices[0], vertices[2], vertices[9], clr);
    polygs_arr[index_start + 11] = polygon(vertices[8], vertices[0], vertices[9], clr);
    polygs_arr[index_start + 12] = polygon(vertices[5], vertices[8], vertices[16], clr);
    polygs_arr[index_start + 13] = polygon(vertices[6], vertices[5], vertices[12], clr);
    polygs_arr[index_start + 14] = polygon(vertices[12], vertices[5], vertices[16], clr);
    polygs_arr[index_start + 15] = polygon(vertices[4], vertices[12], vertices[15], clr);
    polygs_arr[index_start + 16] = polygon(vertices[4], vertices[6], vertices[12], clr);
    polygs_arr[index_start + 17] = polygon(vertices[11], vertices[4], vertices[15], clr);
    polygs_arr[index_start + 18] = polygon(vertices[2], vertices[13], vertices[17], clr);
    polygs_arr[index_start + 19] = polygon(vertices[2], vertices[7], vertices[13], clr);
    polygs_arr[index_start + 20] = polygon(vertices[9], vertices[2], vertices[17], clr);
    polygs_arr[index_start + 21] = polygon(vertices[13], vertices[3], vertices[14], clr);
    polygs_arr[index_start + 22] = polygon(vertices[7], vertices[3], vertices[13], clr);
    polygs_arr[index_start + 23] = polygon(vertices[3], vertices[10], vertices[14], clr);
    polygs_arr[index_start + 24] = polygon(vertices[8], vertices[17], vertices[19], clr);
    polygs_arr[index_start + 25] = polygon(vertices[16], vertices[8], vertices[19], clr);
    polygs_arr[index_start + 26] = polygon(vertices[8], vertices[9], vertices[17], clr);
    polygs_arr[index_start + 27] = polygon(vertices[14], vertices[11], vertices[18], clr);
    polygs_arr[index_start + 28] = polygon(vertices[11], vertices[15], vertices[18], clr);
    polygs_arr[index_start + 29] = polygon(vertices[10], vertices[11], vertices[14], clr);
    polygs_arr[index_start + 30] = polygon(vertices[12], vertices[19], vertices[18], clr);
    polygs_arr[index_start + 31] = polygon(vertices[15], vertices[12], vertices[18], clr);
    polygs_arr[index_start + 32] = polygon(vertices[12], vertices[16], vertices[19], clr);
    polygs_arr[index_start + 33] = polygon(vertices[19], vertices[13], vertices[18], clr);
    polygs_arr[index_start + 34] = polygon(vertices[17], vertices[13], vertices[19], clr);
    polygs_arr[index_start + 35] = polygon(vertices[13], vertices[14], vertices[18], clr);
}

void make_icosahedron(polygon* polygs_arr, int index_start, uchar4 clr, Vector center, double radius){
    double a = (1.0 + sqrt(5.0)) / 2.0;
    Vector vertices[12] = {
        Vector(-1, a, 0),
        Vector(1, a, 0),
        Vector(-1, -a, 0),
        Vector(1, -a, 0),
        Vector(0, -1, a),
        Vector(0, 1, a),
        Vector(0, -1, -a),
        Vector(0, 1, -a),
        Vector(a, 0, -1),
        Vector(a, 0, 1),
        Vector(-a, 0, -1),
        Vector(-a, 0, 1)
    };
    for (auto &v:vertices){
        double n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
        v.x = v.x / n * radius + center.x;
        v.y = v.y / n * radius + center.y;
        v.z = v.z / n * radius + center.z;
    }
    polygs_arr[index_start] = polygon(vertices[0], vertices[11], vertices[5], clr);
    polygs_arr[index_start + 1] = polygon(vertices[0], vertices[5], vertices[1], clr);
    polygs_arr[index_start + 2] = polygon(vertices[0], vertices[1], vertices[7], clr);
    polygs_arr[index_start + 3] = polygon(vertices[0], vertices[7], vertices[10], clr);
    polygs_arr[index_start + 4] = polygon(vertices[0], vertices[10], vertices[11], clr);
    polygs_arr[index_start + 5] = polygon(vertices[1], vertices[5], vertices[9], clr);
    polygs_arr[index_start + 6] = polygon(vertices[5], vertices[11], vertices[4], clr);
    polygs_arr[index_start + 7] = polygon(vertices[11], vertices[10], vertices[2], clr);
    polygs_arr[index_start + 8] = polygon(vertices[10], vertices[7], vertices[6], clr);
    polygs_arr[index_start + 9] = polygon(vertices[7], vertices[1], vertices[8], clr);
    polygs_arr[index_start + 10] = polygon(vertices[3], vertices[9], vertices[4], clr);
    polygs_arr[index_start + 11] = polygon(vertices[3], vertices[4], vertices[2], clr);
    polygs_arr[index_start + 12] = polygon(vertices[3], vertices[2], vertices[6], clr);
    polygs_arr[index_start + 13] = polygon(vertices[3], vertices[6], vertices[8], clr);
    polygs_arr[index_start + 14] = polygon(vertices[3], vertices[8], vertices[9], clr);
    polygs_arr[index_start + 15] = polygon(vertices[4], vertices[9], vertices[5], clr);
    polygs_arr[index_start + 16] = polygon(vertices[2], vertices[4], vertices[11], clr);
    polygs_arr[index_start + 17] = polygon(vertices[6], vertices[2], vertices[10], clr);
    polygs_arr[index_start + 18] = polygon(vertices[8], vertices[6], vertices[7], clr);
    polygs_arr[index_start + 19] = polygon(vertices[9], vertices[8], vertices[1], clr);
}

void default_input(){
    cout << "20" << "\n";
    cout << "images/%d.data" << "\n";
    cout << "600 600 120" << "\n\n";

    cout << "7.0 3.0 0.0     2.0 1.0     2.0 6.0 1.0     0.0 0.0" << "\n";
    cout << "2.0 0.0 0.0     0.5 0.1     1.0 4.0 1.0     0.0 0.0" << "\n\n";

    cout << "3.0 3.0 0.5    0.3 0.55 0.0     1.0" << "\n";
    cout << "0.0 0.0 0.0     0.5 0.25 0.55     1.75" << "\n";
    cout << "-3.0 -3.0 0.0     0.0 0.7 0.7     1.5" << "\n\n";

    cout << "-5.0 -5.0 -1.0     -5.0 5.0 -1.0    5.0 5.0 -1.0    5.0 -5.0 -1.0   1.0 0.9 0.25" << "\n\n";

    cout << "-10.0 0.0 15.0     0.3 0.2 0.1" << "\n\n";

    cout << "4" << "\n";
}

int main(int argc, char **argv) {
    if (argc >= 2 && string(argv[1]) == "--default"){
        default_input();
        exit(0);
    }

    bool gpu_flag = true;
    if (argc >= 2 && string(argv[1]) == "--cpu")
        gpu_flag = false;

    int frames_num;
    char out_path[256];
    int w, h;
    double ang;
    double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc;
    double r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;
    double center1_x, center1_y, center1_z;
    double color1_x, color1_y, color1_z;
    double r1;
    double center2_x, center2_y, center2_z;
    double color2_x, color2_y, color2_z;
    double r2;
    double center3_x, center3_y, center3_z;
    double color3_x, color3_y, color3_z;
    double r3;
    double floor1_x, floor1_y, floor1_z, floor2_x, floor2_y, floor2_z;
    double floor3_x, floor3_y, floor3_z, floor4_x, floor4_y, floor4_z;
    double floor_color_x, floor_color_y, floor_color_z;
    double light_pos_x, light_pos_y, light_pos_z;
    double light_color_x, light_color_y, light_color_z;
    double rays_num;

    cin >> frames_num;
    cin >> out_path;
    cin >> w >> h >> ang;
    cin >> r0c >> z0c >> phi0c >> Arc >> Azc >> wrc >> wzc >> wphic >> prc >> pzc;
    cin >> r0n >> z0n >> phi0n >> Arn >> Azn >> wrn >> wzn >> wphin >> prn >> pzn;
    cin >> center1_x >> center1_y >> center1_z;
    cin >> color1_x >> color1_y >> color1_z;
    cin >> r1;
    cin >> center2_x >> center2_y >> center2_z;
    cin >> color2_x >> color2_y >> color2_z;
    cin >> r2;
    cin >> center3_x >> center3_y >> center3_z;
    cin >> color3_x >> color3_y >> color3_z;
    cin >> r3;
    cin >> floor1_x >> floor1_y >> floor1_z >> floor2_x >> floor2_y >> floor2_z;
    cin >> floor3_x >> floor3_y >> floor3_z >> floor4_x >> floor4_y >> floor4_z;
    cin >> floor_color_x >> floor_color_y >> floor_color_z;
    cin >> light_pos_x >> light_pos_y >> light_pos_z;
    cin >> light_color_x >> light_color_y >> light_color_z;
    cin >> rays_num;

    polygon polygs_arr[70];
    make_scene(
        polygs_arr, 0,
        make_uchar4(floor_color_x * 255, floor_color_y * 255, floor_color_z * 255, 255),
        Vector(floor1_x, floor1_y, floor1_z),
        Vector(floor2_x, floor2_y, floor2_z),
        Vector(floor3_x, floor3_y, floor3_z),
        Vector(floor4_x, floor4_y, floor4_z)
    );
    make_icosahedron(
        polygs_arr, 2,
        make_uchar4(color1_x * 255, color1_y * 255, color1_z * 255, 255),
        Vector(center1_x, center1_y, center1_z), r1
    );
    make_hexahedron(
        polygs_arr, 22,
        make_uchar4(color2_x * 255, color2_y * 255, color2_z * 255, 255),
        Vector(center2_x, center2_y, center2_z), r2
    );
    make_dodecahedron(
        polygs_arr, 34,
        make_uchar4(color3_x * 255, color3_y * 255, color3_z * 255, 255),
        Vector(center3_x, center3_y, center3_z), r3
    );
    Vector light_plc = Vector(light_pos_x, light_pos_y, light_pos_z);
    uchar4 light_clr = make_uchar4(light_color_x * 255, light_color_y * 255, light_color_z * 255, 255);

    uchar4 *img = (uchar4*)malloc(sizeof(uchar4) * w * h * rays_num * rays_num);
    uchar4 *ssaa_img = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4 *ker_data;
    uchar4 *ker_ssaa_data;
    polygon* ker_polygs;
    char buffer[256];
    if (gpu_flag){
        CSC(cudaMalloc(&ker_data, sizeof(uchar4) * w * h * rays_num * rays_num));
        CSC(cudaMalloc(&ker_ssaa_data, sizeof(uchar4) * w * h));
        CSC(cudaMalloc(&ker_polygs, sizeof(polygon) * 70));
        CSC(cudaMemcpy(ker_polygs, polygs_arr, sizeof(polygon) * 70, cudaMemcpyHostToDevice));
    }

    for (int frame = 0; frame < frames_num; ++frame){
        double t = 2 * M_PI * frame / frames_num;
        Vector cam_place, cam_view;
        double rc = r0c + Arc * sin(wrc * t + prc);
        double zc = z0c + Azc * sin(wzc * t + pzc);
        double phic = phi0c + wphic * t;
        double rn = r0n + Arn * sin(wrn * t + prn);
        double zn = z0n + Azn * sin(wzn * t + pzn);
        double phin = phi0n + wphin * t;
        cam_place.x = rc * cos(phic);
        cam_place.y = rc * sin(phic);
        cam_place.z = zc;
        cam_view.x = rn * cos(phin);
        cam_view.y = rn * sin(phin);
        cam_view.z = zn;
    
        cudaEvent_t start, stop;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start));

        if (gpu_flag){
            kernel_render<<<dim3(16, 16), dim3(16, 16)>>>(ker_polygs, 70, ker_data, light_plc, light_clr, cam_place, cam_view, w * rays_num, h * rays_num, ang);
            CSC(cudaGetLastError());
            kernel_ssaa<<<dim3(16, 16), dim3(16, 16)>>>(rays_num, ker_data, ker_ssaa_data, w, h);
            CSC(cudaGetLastError());
            CSC(cudaMemcpy(ssaa_img, ker_ssaa_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        } 
        else{
            cpu_render(cam_place, cam_view, w * rays_num, h * rays_num, ang, img, light_plc, light_clr, polygs_arr, 70);
            cpu_make_ssaa(rays_num, img, ssaa_img, w, h);
        }

        CSC(cudaEventRecord(stop));
        CSC(cudaEventSynchronize(stop));
        float time;
        CSC(cudaEventElapsedTime(&time, start, stop));
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));

        sprintf(buffer, out_path, frame);
        FILE *output = fopen(buffer, "w");
        if (output == NULL)
            return -1;
        fwrite(&w, sizeof(int), 1, output);
        fwrite(&h, sizeof(int), 1, output);
        fwrite(ssaa_img, sizeof(uchar4), w * h, output);
        fclose(output);
        cout << frame + 1 << "\t" << time << "\t" << w * h * rays_num * rays_num << endl;
    }
    free(img);
    free(ssaa_img);
    if (gpu_flag){
        CSC(cudaFree(ker_data));
        CSC(cudaFree(ker_ssaa_data));
    }
    return 0;
}