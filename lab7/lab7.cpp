#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>

#include "mpi.h"

#define _i(i, j, k) (((k) + 1) * (yb + 2) * (xb + 2) + ((j) + 1) * (xb + 2) + (i) + 1)
#define _ib(i, j, k) ((k) * (xg * yg) + (j) * xg + (i))

int main(int argc, char *argv[]) {
    int numproc, id;
    int xg, yg, zg;
    int xb, yb, zb;
    std::string file;
    double eps;
    double lx, ly, lz;
    double bc_down, bc_up, bc_left, bc_right, bc_front, bc_back;
    double u0;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Barrier(MPI_COMM_WORLD);

    if (id == 0) {
        std::cin >> xg >> yg >> zg;
        std::cin >> xb >> yb >> zb;
        std::cin >> file;
        std::cin >> eps;
        std::cin >> lx >> ly >> lz;
        std::cin >> bc_down >> bc_up >> bc_left >> bc_right >> bc_front >> bc_back;
        std::cin >> u0;
    }

    MPI_Bcast(&xg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zg, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&zb, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&bc_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double hx = lx / (xg * xb);	
    double hy = ly / (yg * yb);
    double hz = lz / (zg * zb);

	double *temp;
    double* data = (double*)malloc(sizeof(double) * (xb + 2) * (yb + 2) * (zb + 2));	
    double* next = (double*)malloc(sizeof(double) * (xb + 2) * (yb + 2) * (zb + 2));
    double* buff = (double*)malloc(sizeof(double) * xb * yb * zb);

    int buffer_size;
    MPI_Pack_size((xb+2) * (yb+2) * (zb+2), MPI_DOUBLE, MPI_COMM_WORLD, &buffer_size);
    buffer_size = 12 * (buffer_size + MPI_BSEND_OVERHEAD);
    double* buffer = (double*)malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    for (int i = 0; i < xb; ++i) {
        for (int j = 0; j < yb; ++j) {
            for (int k = 0; k < zb; ++k) {
                data[_i(i, j, k)] = u0;
            }
        }
    }

    int ib = id % (xg * yg) % xg;
    int jb = id % (xg * yg) / xg;
    int kb = id / (xg * yg);

	double global_max;
	do {
        MPI_Barrier(MPI_COMM_WORLD);
		
        if (ib + 1 < xg) {
            for (int j = 0; j < yb; ++j) {
                for (int k = 0; k < zb; ++k) {
                    buff[j * zb + k] = data[_i(xb - 1, j, k)];
                }
            }
            MPI_Bsend(buff, yb * zb, MPI_DOUBLE, _ib(ib + 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (ib - 1 >= 0) {
            for (int j = 0; j < yb; ++j) {
                for (int k = 0; k < zb; ++k) {
                    buff[j * zb + k] = data[_i(0, j, k)];
                }
            }
            MPI_Bsend(buff, yb * zb, MPI_DOUBLE, _ib(ib - 1, jb, kb), id, MPI_COMM_WORLD);
        }

        if (jb + 1 < yg) {
            for (int i = 0; i < xb; ++i) {
                for (int k = 0; k < zb; ++k) {
                    buff[i * zb + k] = data[_i(i, yb - 1, k)];
                }
            }
            MPI_Bsend(buff, xb * zb, MPI_DOUBLE, _ib(ib, jb + 1, kb), id, MPI_COMM_WORLD);
        }

        if (jb - 1 >= 0) {
            for (int i = 0; i < xb; ++i) {
                for (int k = 0; k < zb; ++k) {
                    buff[i * zb + k] = data[_i(i, 0, k)];
                }
            }
            MPI_Bsend(buff, xb * zb, MPI_DOUBLE, _ib(ib, jb - 1, kb), id, MPI_COMM_WORLD);
        }

        if (kb + 1 < zg) {
            for (int i = 0; i < xb; ++i) {
                for (int j = 0; j < yb; ++j) {
                    buff[i * yb + j] = data[_i(i, j, zb - 1)];
                }
            }
            MPI_Bsend(buff, xb * yb, MPI_DOUBLE, _ib(ib, jb, kb + 1), id, MPI_COMM_WORLD);
        }

        if (kb - 1 >= 0) {
            for (int i = 0; i < xb; ++i) {
                for (int j = 0; j < yb; ++j) {
                    buff[i * yb + j] = data[_i(i, j, 0)];
                }
            }
            MPI_Bsend(buff, xb * yb, MPI_DOUBLE, _ib(ib, jb, kb - 1), id, MPI_COMM_WORLD);
        }

		//------------------------------------------------------------------------------------------------------------
		
        if (ib + 1 < xg) {
            MPI_Recv(buff, yb * zb, MPI_DOUBLE, _ib(ib + 1, jb, kb), _ib(ib + 1, jb, kb), MPI_COMM_WORLD, &status);
            for (int j = 0; j < yb; ++j) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(xb, j, k)] = buff[j * zb + k];
                }
            }
        } else {
            for (int j = 0; j < yb; ++j) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(xb, j, k)] = bc_right;
                }
            }
        }

        if (ib - 1 >= 0) {
            MPI_Recv(buff, yb * zb, MPI_DOUBLE, _ib(ib - 1, jb, kb), _ib(ib - 1, jb, kb), MPI_COMM_WORLD, &status);
            for (int j = 0; j < yb; ++j) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(-1, j, k)] = buff[j * zb + k];
                }
            }
        } else {
            for (int j = 0; j < yb; ++j) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(-1, j, k)] = bc_left;
                }
            }
        }

        if (jb + 1 < yg) {
            MPI_Recv(buff, xb * zb, MPI_DOUBLE, _ib(ib, jb + 1, kb), _ib(ib, jb + 1, kb), MPI_COMM_WORLD, &status);
            for (int i = 0; i < xb; ++i) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(i, yb, k)] = buff[i * zb + k];
                }
            }
        } else {
            for (int i = 0; i < xb; ++i) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(i, yb, k)] = bc_back;
                }
            }
        }

        if (jb - 1 >= 0) {
            MPI_Recv(buff, xb * zb, MPI_DOUBLE, _ib(ib, jb - 1, kb), _ib(ib, jb - 1, kb), MPI_COMM_WORLD, &status);
            for (int i = 0; i < xb; ++i) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(i, -1, k)] = buff[i * zb + k];
                }
            }
        } else {
            for (int i = 0; i < xb; ++i) {
                for (int k = 0; k < zb; ++k) {
                    data[_i(i, -1, k)] = bc_front;
                }
            }
        }

        if (kb + 1 < zg) {
            MPI_Recv(buff, xb * yb, MPI_DOUBLE, _ib(ib, jb, kb + 1), _ib(ib, jb, kb + 1), MPI_COMM_WORLD, &status);
            for (int i = 0; i < xb; ++i) {
                for (int j = 0; j < yb; ++j) {
                    data[_i(i, j, zb)] = buff[i * yb + j];
                }
            }
        } else {
            for (int i = 0; i < xb; ++i) {
                for (int j = 0; j < yb; ++j) {
                    data[_i(i, j, zb)] = bc_up;
                }
            }
        }

        if (kb - 1 >= 0) {
            MPI_Recv(buff, xb * yb, MPI_DOUBLE, _ib(ib, jb, kb - 1), _ib(ib, jb, kb - 1), MPI_COMM_WORLD, &status);
            for (int i = 0; i < xb; ++i) {
                for (int j = 0; j < yb; ++j) {
                    data[_i(i, j, -1)] = buff[i * yb + j];
                }
            }
        } else {
            for (int i = 0; i < xb; ++i) {
                for (int j = 0; j < yb; ++j) {
                    data[_i(i, j, -1)] = bc_down;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

		double max = 0;
		for (int i = 0; i < xb; i++)
        {
            for (int j = 0; j < yb; j++)
            {
                for (int k = 0; k < zb; k++)
                {
                    next[_i(i, j, k)] = ((data[_i(i + 1, j, k)] + data[_i(i - 1, j, k)]) / (hx * hx) +
                       (data[_i(i, j + 1, k)] + data[_i(i, j - 1, k)]) / (hy * hy) +
                        (data[_i(i, j, k + 1)] + data[_i(i, j, k - 1)]) / (hz * hz)) /
                        (2 * (1.0 / (hx * hx) + 1.0 / (hy * hy) + 1.0 / (hz * hz)));
                    max = std::max(max, fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
	 	MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        temp = data;
        data = next;
        next = temp;

    } while (global_max >= eps);
    
    MPI_Barrier(MPI_COMM_WORLD);

    std::ofstream output(file);
	output << std::setprecision(6);
    if (id != 0) {
        for (int k = 0; k < zb; ++k) {
            for (int j = 0; j < yb; ++j) {
                for (int i = 0; i < xb; ++i) {
                    buff[i] = data[_i(i, j, k)];
                }
                MPI_Send(buff, xb, MPI_DOUBLE, 0, k * yb + j, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int g3 = 0; g3 < zg; ++g3) {
            for (int b3 = 0; b3 < zb; ++b3) {
                for (int g2 = 0; g2 < yg; ++g2) {
                    for (int b2 = 0; b2 < yb; ++b2) {
                        for (int g1 = 0; g1 < xg; ++g1) {
                            
                            if (_ib(g1, g2, g3) == 0) {
                                for (int b1 = 0; b1 < xb; ++b1)
                                    buff[b1] = data[_i(b1, b2, b3)];
							}
                            else {
                                MPI_Recv(buff, xb, MPI_DOUBLE, _ib(g1, g2, g3), b3 * yb + b2, MPI_COMM_WORLD, &status);
							}

                            for (int i = 0; i < xb; ++i)
                                output << buff[i] << " ";
                            if (g1 + 1 == xg)
                                output << "\n";

                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Buffer_detach(buffer, &buffer_size);
    MPI_Finalize();

    free(buffer);
    free(data);
    free(buff);
    free(next);

    return 0;
}
