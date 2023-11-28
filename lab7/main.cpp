#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "mpi.h"

#define _i(i, j, k) (((k) + 1) * (block_y + 2) * (block_x + 2) + ((j) + 1) * (block_x + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * (grid_x * grid_y) + (j) * grid_x + (i))

int main(int argc, char *argv[]) {
    int ib, jb, kb;
    int grid_x, grid_y, grid_z, block_x, block_y, block_z;
    double hx, hy, hz;
    std::string out_line;
    double eps;
    double lx, ly, lz;
    double bc_d, bc_u, bc_l, bc_r, bc_f, bc_b;
    double u_0;
    int ProcNum, ProcRank;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (ProcRank == 0) {
        std::cin >> grid_x >> grid_y >> grid_z;
        std::cin >> block_x >> block_y >> block_z;
        std::cin >> out_line;
        std::cin >> eps;
        std::cin >> lx >> ly >> lz;
        std::cin >> bc_d >> bc_u >> bc_l >> bc_r >> bc_f >> bc_b;
        std::cin >> u_0;
    }
    MPI_Bcast(&grid_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&grid_z, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&block_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&block_z, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&bc_d, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_u, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&bc_l, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_r, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&bc_f, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&u_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    ib = ProcRank % (grid_x * grid_y) % grid_x;
    jb = ProcRank % (grid_x * grid_y) / grid_x;
    kb = ProcRank / (grid_x * grid_y);

    hx = lx / (grid_x * block_x);	
    hy = ly / (grid_y * block_y);
    hz = lz / (grid_z * block_z);

    double* data = (double*)malloc(sizeof(double) * (block_x + 2) * (block_y + 2) * (block_z + 2));	
    double* next = (double*)malloc(sizeof(double) * (block_x + 2) * (block_y + 2) * (block_z + 2));
    double* buff = (double*)malloc(sizeof(double) * block_x * block_y * block_z);
    int buf_size;
    MPI_Pack_size((block_x + 2) * (block_y + 2) * (block_z + 2), MPI_DOUBLE, MPI_COMM_WORLD, &buf_size);
    buf_size = 12 * (buf_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buf_size);
    MPI_Buffer_attach(buffer, buf_size);
    for (int i = 0; i < block_x; ++i) {
        for (int j = 0; j < block_y; ++j) {
            for (int k = 0; k < block_z; ++k) {
                data[_i(i, j, k)] = u_0;
            }
        }
    }
    
    double max_global;
    do {
        MPI_Barrier(MPI_COMM_WORLD);

        // ----------------------------
        if (ib + 1 < grid_x){
            for (int iy = 0; iy < block_y; ++iy)
                for (int jz = 0; jz < block_z; ++jz)
                    buff[iy * block_z + jz] = data[_i(block_x - 1, iy, jz)];

            MPI_Send(buff, block_y * block_z, MPI_DOUBLE, _ib(ib + 1, jb, kb), ProcRank, MPI_COMM_WORLD);
        }

        if (ib >= 1){
            MPI_Recv(buff, block_y * block_z, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
            for (int iy = 0; iy < block_y; ++iy)
                for (int jz = 0; jz < block_z; ++jz)
                    data[_i(-1, iy, jz)] = buff[iy * block_z + jz];
        } 
        else{
            for (int iy = 0; iy < block_y; ++iy)
                for (int yz = 0; yz < block_z; ++yz)
                    data[_i(-1, iy, yz)] = bc_l;
        }

        if (jb + 1 < grid_y){
            for (int ix = 0; ix < block_x; ++ix)
                for (int jz = 0; jz < block_z; ++jz)
                    buff[ix * block_z + jz] = data[_i(ix, block_y - 1, jz)];
                
            MPI_Send(buff, block_x * block_z, MPI_DOUBLE, _ib(ib, jb + 1, kb), ProcRank, MPI_COMM_WORLD);
        }

        if (jb >= 1){
            MPI_Recv(buff, block_x * block_z, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
            for (int ix = 0; ix < block_x; ++ix)
                for (int jz = 0; jz < block_z; ++jz)
                    data[_i(ix, -1, jz)] = buff[ix * block_z + jz];
        } 
        else{
            for (int ix = 0; ix < block_x; ++ix)
                for (int jz = 0; jz < block_z; ++jz)
                    data[_i(ix, -1, jz)] = bc_f;
        }

        if (kb + 1 < grid_z){
            for (int ix = 0; ix < block_x; ++ix)
                for (int jy = 0; jy < block_y; ++jy)
                    buff[ix * block_y + jy] = data[_i(ix, jy, block_z - 1)];

            MPI_Send(buff, block_x * block_y, MPI_DOUBLE, _ib(ib, jb, kb + 1), ProcRank, MPI_COMM_WORLD);
        }

        if (kb >= 1){
            MPI_Recv(buff, block_x * block_y, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
            for (int ix = 0; ix < block_x; ++ix)
                for (int jy = 0; jy < block_y; ++jy)
                    data[_i(ix, jy, -1)] = buff[ix * block_y + jy];
        } 
        else{
            for (int ix = 0; ix < block_x; ++ix)
                for (int jy = 0; jy < block_y; ++jy)
                    data[_i(ix, jy, -1)] = bc_d;
        }
        // ----------------------------
        if (ib - 1 >= 0){
            for (int iy = 0; iy < block_y; ++iy)
                for (int jz = 0; jz < block_z; ++jz)
                    buff[iy * block_z + jz] = data[_i(0, iy, jz)];

            MPI_Send(buff, block_y * block_z, MPI_DOUBLE, _ib(ib - 1, jb, kb), ProcRank, MPI_COMM_WORLD);
        }

        if (ib + 1 < grid_x){
            MPI_Recv(buff, block_y * block_z, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
            for (int iy = 0; iy < block_y; ++iy)
                for (int jz = 0; jz < block_z; ++jz) 
                    data[_i(block_x, iy, jz)] = buff[iy * block_z + jz];
        } 
        else{
            for (int iy = 0; iy < block_y; ++iy)
                for (int jz = 0; jz < block_z; ++jz)
                    data[_i(block_x, iy, jz)] = bc_r;
        }    

        if (jb - 1 >= 0){
            for (int ix = 0; ix < block_x; ++ix)
                for (int jz = 0; jz < block_z; ++jz)
                    buff[ix * block_z + jz] = data[_i(ix, 0, jz)];
  
            MPI_Send(buff, block_x * block_z, MPI_DOUBLE, _ib(ib, jb - 1, kb), ProcRank, MPI_COMM_WORLD);
        }

        if (jb + 1 < grid_y){
            MPI_Recv(buff, block_x * block_z, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
            for (int ix = 0; ix < block_x; ++ix)
                for (int jz = 0; jz < block_z; ++jz)
                    data[_i(ix, block_y, jz)] = buff[ix * block_z + jz];
        } 
        else{
            for (int ix = 0; ix < block_x; ++ix)
                for (int jz = 0; jz < block_z; ++jz)
                    data[_i(ix, block_y, jz)] = bc_b;
        }    

        if (kb - 1 >= 0){
            for (int ix = 0; ix < block_x; ++ix)
                for (int jy = 0; jy < block_y; ++jy)
                    buff[ix * block_y + jy] = data[_i(ix, jy, 0)];

            MPI_Send(buff, block_x * block_y, MPI_DOUBLE, _ib(ib, jb, kb - 1), ProcRank, MPI_COMM_WORLD);
        }
        
        if (kb + 1 < grid_z){
            MPI_Recv(buff, block_x * block_y, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
            for (int ix = 0; ix < block_x; ++ix)
                for (int jy = 0; jy < block_y; ++jy) 
                    data[_i(ix, jy, block_z)] = buff[ix * block_y + jy];
        } 
        else{
            for (int ix = 0; ix < block_x; ++ix)
                for (int jy = 0; jy < block_y; ++jy)
                    data[_i(ix, jy, block_z)] = bc_u;
        }

        // ----------------------------
        MPI_Barrier(MPI_COMM_WORLD);

        double curr_max = 0.0;
        for (int ix = 0; ix < block_x; ++ix){
            for (int jy = 0; jy < block_y; ++jy)
                for (int kz = 0; kz < block_z; ++kz){
                    double summ = 1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz);
                    next[_i(ix, jy, kz)] = ((data[_i(ix + 1, jy, kz)] + data[_i(ix - 1, jy, kz)]) / (hx * hx)
                        + (data[_i(ix, jy + 1, kz)] + data[_i(ix, jy - 1, kz)]) / (hy * hy)
                        + (data[_i(ix, jy, kz + 1)] + data[_i(ix, jy, kz - 1)]) / (hz * hz)) / (2 * summ);
                    double diff = fabs(next[_i(ix, jy, kz)] - data[_i(ix, jy, kz)]);
                    if (curr_max <= diff)
                        curr_max = diff;
                }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&curr_max, &max_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double *temp = data;
        data = next;
        next = temp;
    
    } while (eps < max_global);

    MPI_Barrier(MPI_COMM_WORLD);

    if (ProcRank != 0){
        for (int iz = 0; iz < block_z; ++iz){
            for (int jy = 0; jy < block_y; ++jy){
                for (int kx = 0; kx < block_x; ++kx)
                    buff[kx] = data[_i(kx, jy, iz)];
                MPI_Send(buff, block_x, MPI_DOUBLE, 0, iz * block_y + jy, MPI_COMM_WORLD);
            }
        }
    } 
    else{
        std::ofstream output_file(out_line);
        output_file << std::scientific << std::setprecision(6);
        for (int igz = 0; igz < grid_z; ++igz){
            for (int iz = 0; iz < block_z; ++iz){
                for (int jgy = 0; jgy < grid_y; ++jgy){
                    for (int jy = 0; jy < block_y; ++jy){
                        for (int igx = 0; igx < grid_x; ++igx){
                            if (_ib(igx, jgy, igz) == 0){
                                for (int ix = 0; ix < block_x; ++ix)
                                    buff[ix] = data[_i(ix, jy, iz)];
                            } 
                            else
                                MPI_Recv(buff, block_x, MPI_DOUBLE, _ib(igx, jgy, igz), iz * block_y + jy, MPI_COMM_WORLD, &status);

                            for (int ix = 0; ix < block_x; ++ix)
                                output_file << buff[ix] << " ";

                            if (igx + 1 == grid_x)
                                output_file << "\n";
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Buffer_detach(buffer, &buf_size);
    MPI_Finalize();
    free(data);
    free(next);
    free(buff);
    free(buffer);
    return 0;
}