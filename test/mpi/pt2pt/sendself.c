/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2006 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpitest.h"
#include "dtpools.h"

/*
static char MTEST_Descrip[] = "Test of sending to self (with a preposted receive)";
*/



int main(int argc, char *argv[])
{
    int errs = 0, err;
    int rank, size;
    int count[2];
    int i, j, len;
    MPI_Aint sendcount, recvcount;
    MPI_Comm comm;
    MPI_Datatype sendtype, recvtype;
    MPI_Request req;
    DTP_t send_dtp, recv_dtp;
    char send_name[MPI_MAX_OBJECT_NAME] = { 0 };
    char recv_name[MPI_MAX_OBJECT_NAME] = { 0 };
    void *sendbuf, *recvbuf;

    MTest_Init(&argc, &argv);

    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

#ifndef USE_DTP_POOL_TYPE__STRUCT       /* set in 'test/mpi/structtypetest.txt' to split tests */
    MPI_Datatype basic_type;
    char type_name[MPI_MAX_OBJECT_NAME] = { 0 };

    err = MTestInitBasicPt2ptSignature(argc, argv, count, &basic_type);
    if (err)
        return MTestReturnValue(1);

    err = DTP_pool_create(basic_type, count[0], &send_dtp);
    if (err != DTP_SUCCESS) {
        MPI_Type_get_name(basic_type, type_name, &len);
        fprintf(stdout, "Error while creating send pool (%s,%d)\n", type_name, count[0]);
        fflush(stdout);
    }

    err = DTP_pool_create(basic_type, count[1], &recv_dtp);
    if (err != DTP_SUCCESS) {
        MPI_Type_get_name(basic_type, type_name, &len);
        fprintf(stdout, "Error while creating recv pool (%s,%d)\n", type_name, count[1]);
        fflush(stdout);
    }
#else
    MPI_Datatype *basic_types = NULL;
    int *basic_type_counts = NULL;
    int basic_type_num;

    err = MTestInitStructSignature(argc, argv, &basic_type_num, &basic_type_counts, &basic_types);
    if (err)
        return MTestReturnValue(1);

    err = DTP_pool_create_struct(basic_type_num, basic_types, basic_type_counts, &send_dtp);
    if (err != DTP_SUCCESS) {
        fprintf(stdout, "Error while creating struct pool\n");
        fflush(stdout);
    }

    err = DTP_pool_create_struct(basic_type_num, basic_types, basic_type_counts, &recv_dtp);
    if (err != DTP_SUCCESS) {
        fprintf(stdout, "Error while creating struct pool\n");
        fflush(stdout);
    }

    /* these are ignored */
    count[0] = 0;
    count[1] = 0;
#endif

    /* To improve reporting of problems about operations, we
     * change the error handler to errors return */
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    for (i = 0; i < send_dtp->DTP_num_objs; i++) {
        err = DTP_obj_create(send_dtp, i, 0, 1, count[0]);
        if (err != DTP_SUCCESS) {
            errs++;
        }

        sendcount = send_dtp->DTP_obj_array[i].DTP_obj_count;
        sendtype = send_dtp->DTP_obj_array[i].DTP_obj_type;
        sendbuf = send_dtp->DTP_obj_array[i].DTP_obj_buf;

        for (j = 0; j < recv_dtp->DTP_num_objs; j++) {
            err = DTP_obj_create(recv_dtp, j, 0, 0, 0);
            if (err != DTP_SUCCESS) {
                errs++;
            }

            recvcount = recv_dtp->DTP_obj_array[j].DTP_obj_count;
            recvtype = recv_dtp->DTP_obj_array[j].DTP_obj_type;
            recvbuf = recv_dtp->DTP_obj_array[j].DTP_obj_buf;

            err = MPI_Irecv(recvbuf, recvcount, recvtype, rank, 0, comm, &req);
            if (err) {
                errs++;
                if (errs < 10) {
                    MTestPrintError(err);
                }
            }

            err = MPI_Send(sendbuf, sendcount, sendtype, rank, 0, comm);
            if (err) {
                errs++;
                if (errs < 10) {
                    MTestPrintError(err);
                }
            }

            err = MPI_Wait(&req, MPI_STATUS_IGNORE);
            err = DTP_obj_buf_check(recv_dtp, j, 0, 1, count[0]);
            if (err != DTP_SUCCESS) {
                if (errs < 10) {
                    MPI_Type_get_name(sendtype, send_name, &len);
                    MPI_Type_get_name(recvtype, recv_name, &len);
                    fprintf(stdout,
                            "Data in target buffer did not match for destination datatype %s and source datatype %s, count = %d\n",
                            recv_name, send_name, count[0]);
                    fflush(stdout);
                }
                errs++;
            }

            err = MPI_Irecv(recvbuf, recvcount, recvtype, rank, 0, comm, &req);
            if (err) {
                errs++;
                if (errs < 10) {
                    MTestPrintError(err);
                }
            }

            err = MPI_Ssend(sendbuf, sendcount, sendtype, rank, 0, comm);
            if (err) {
                errs++;
                if (errs < 10) {
                    MTestPrintError(err);
                }
            }

            err = MPI_Wait(&req, MPI_STATUS_IGNORE);
            err = DTP_obj_buf_check(recv_dtp, j, 0, 1, count[0]);
            if (err != DTP_SUCCESS) {
                if (errs < 10) {
                    MPI_Type_get_name(sendtype, send_name, &len);
                    MPI_Type_get_name(recvtype, recv_name, &len);
                    fprintf(stdout,
                            "Data in target buffer did not match for destination datatype %s and source datatype %s, count = %d\n",
                            recv_name, send_name, count[0]);
                    fflush(stdout);
                }
                errs++;
            }

            err = MPI_Irecv(recvbuf, recvcount, recvtype, rank, 0, comm, &req);
            if (err) {
                errs++;
                if (errs < 10) {
                    MTestPrintError(err);
                }
            }

            err = MPI_Rsend(sendbuf, sendcount, sendtype, rank, 0, comm);
            if (err) {
                errs++;
                if (errs < 10) {
                    MTestPrintError(err);
                }
            }

            err = MPI_Wait(&req, MPI_STATUS_IGNORE);
            err = DTP_obj_buf_check(recv_dtp, j, 0, 1, count[0]);
            if (err != DTP_SUCCESS) {
                if (errs < 10) {
                    MPI_Type_get_name(sendtype, send_name, &len);
                    MPI_Type_get_name(recvtype, recv_name, &len);
                    fprintf(stdout,
                            "Data in target buffer did not match for destination datatype %s and source datatype %s, count = %d\n",
                            recv_name, send_name, count[0]);
                    fflush(stdout);
                }
                errs++;
            }
            DTP_obj_free(recv_dtp, j);
        }
        DTP_obj_free(send_dtp, i);
    }

    DTP_pool_free(send_dtp);
    DTP_pool_free(recv_dtp);

#ifdef USE_DTP_POOL_TYPE__STRUCT
    /* cleanup array if any */
    if (basic_types) {
        free(basic_types);
    }
    if (basic_type_counts) {
        free(basic_type_counts);
    }
#endif

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
