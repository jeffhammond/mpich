/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"

#if defined(__GLIBC__)
#include <features.h>
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 33)
#include <malloc.h>
#define HAVE_MALLINFO2 1
#endif
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_REPS 10

static long long current_heap_bytes(void)
{
#if defined(HAVE_MALLINFO2)
    struct mallinfo2 mi = mallinfo2();
    return (long long) mi.uordblks;
#else
    return -1;
#endif
}

static long long proc_status_kb(const char *name)
{
#if defined(__linux__)
    FILE *fp = fopen("/proc/self/status", "r");
    char line[256];
    size_t name_len = strlen(name);

    if (!fp)
        return -1;

    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, name, name_len) == 0 && line[name_len] == ':') {
            char *p = line + name_len + 1;
            long long value;

            while (*p == ' ' || *p == '\t')
                p++;
            value = strtoll(p, NULL, 10);
            fclose(fp);
            return value;
        }
    }

    fclose(fp);
#endif
    return -1;
}

static int get_arg_int(int argc, char **argv, const char *name, int default_value)
{
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0)
            return atoi(argv[i + 1]);
    }
    return default_value;
}

static int get_key(int active_rank, int subcomm_size)
{
    int offset = active_rank % subcomm_size;

    return subcomm_size - 1 - offset;
}

static void run_split_case(MPI_Comm comm, int subcomm_size, int reps,
                           int rank, double *time_sec,
                           long long *heap_delta, long long *rss_delta, long long *hwm_delta)
{
    MPI_Comm newcomm;
    int color = rank / subcomm_size;
    int key = get_key(rank, subcomm_size);
    long long heap0, heap1, rss0, rss1, hwm0, hwm1;
    long long local_heap_delta, local_rss_delta, local_hwm_delta;
    double local_total = 0.0;

    MPI_Barrier(comm);
    heap0 = current_heap_bytes();
    rss0 = proc_status_kb("VmRSS");
    hwm0 = proc_status_kb("VmHWM");
    MPI_Comm_split(comm, color, key, &newcomm);
    heap1 = current_heap_bytes();
    rss1 = proc_status_kb("VmRSS");
    hwm1 = proc_status_kb("VmHWM");
    MPI_Comm_free(&newcomm);

    local_heap_delta = (heap0 >= 0 && heap1 >= 0) ? heap1 - heap0 : -1;
    local_rss_delta = (rss0 >= 0 && rss1 >= 0) ? rss1 - rss0 : -1;
    local_hwm_delta = (hwm0 >= 0 && hwm1 >= 0) ? hwm1 - hwm0 : -1;

    MPI_Reduce(&local_heap_delta, heap_delta, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&local_rss_delta, rss_delta, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
    MPI_Reduce(&local_hwm_delta, hwm_delta, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);

    MPI_Barrier(comm);
    for (int i = 0; i < reps; i++) {
        double t0, t1;

        t0 = MPI_Wtime();
        MPI_Comm_split(comm, color, key, &newcomm);
        t1 = MPI_Wtime();
        local_total += t1 - t0;
        MPI_Comm_free(&newcomm);
    }

    local_total /= reps;
    MPI_Reduce(&local_total, time_sec, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

}

int main(int argc, char **argv)
{
    int wrank, wsize;
    int reps;
    int max_size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);

    reps = get_arg_int(argc, argv, "-reps", DEFAULT_REPS);
    max_size = get_arg_int(argc, argv, "-max-size", wsize);
    if (max_size > wsize)
        max_size = wsize;
    if (max_size < 1)
        max_size = 1;
    if (reps < 1)
        reps = 1;

    if (wrank == 0) {
        printf("# commsplit_sweep world_size=%d max_size=%d reps=%d\n", wsize, max_size, reps);
        printf("active_size,subcomm_size,reps,time_sec,heap_delta_bytes,rss_delta_kb,hwm_delta_kb\n");
    }

    for (int active_size = 1; active_size <= max_size; active_size++) {
        MPI_Comm active_comm;
        int active_color = (wrank < active_size) ? 0 : MPI_UNDEFINED;

        MPI_Comm_split(MPI_COMM_WORLD, active_color, wrank, &active_comm);
        if (active_comm != MPI_COMM_NULL) {
            int active_rank;

            MPI_Comm_rank(active_comm, &active_rank);
            for (int subcomm_size = 1; subcomm_size <= active_size; subcomm_size++) {
                double time_sec;
                long long heap_delta, rss_delta, hwm_delta;

                run_split_case(active_comm, subcomm_size, reps, active_rank, &time_sec, &heap_delta,
                               &rss_delta, &hwm_delta);

                if (active_rank == 0) {
                    printf("%d,%d,%d,%.9e,%lld,%lld,%lld\n", active_size, subcomm_size, reps,
                           time_sec, heap_delta, rss_delta, hwm_delta);
                }
            }
            MPI_Comm_free(&active_comm);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
