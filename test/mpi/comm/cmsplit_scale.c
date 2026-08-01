/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpi.h"
#include "mpitest.h"
#include <stdio.h>
#include <stdlib.h>

#define NUM_CASES 8
#define ERRLIMIT 20

static void get_split_case(int case_num, int rank, int size, int *color, int *key)
{
    switch (case_num) {
        case 0:
            *color = 0;
            *key = size - rank;
            break;
        case 1:
            *color = 0;
            *key = ((rank * 17 + 11) % 23) - 11;
            break;
        case 2:
            *color = rank % 7;
            *key = ((rank * 5 + 3) % 13) - 6;
            break;
        case 3:
            *color = (rank % 11 == 0) ? MPI_UNDEFINED : (rank / 3) % 5;
            *key = (rank % 3) - 1;
            break;
        case 4:
            *color = (rank < size / 3 || rank >= (2 * size) / 3) ? 2 : 1;
            *key = ((size - rank) % 9) - 4;
            break;
        case 5:
            *color = rank;
            *key = 0;
            break;
        case 6:
            *color = (rank % 2 == 0) ? 3 : rank % 5;
            *key = rank % 4;
            break;
        default:
            *color = (rank + size) % 9;
            *key = (rank % 2) ? -rank : rank % 17;
            break;
    }
}

static int split_rank_precedes(int case_num, int size, int rank1, int rank2)
{
    int color1, key1, color2, key2;

    get_split_case(case_num, rank1, size, &color1, &key1);
    get_split_case(case_num, rank2, size, &color2, &key2);

    if (color1 != color2)
        return 0;

    return key1 < key2 || (key1 == key2 && rank1 < rank2);
}

static int get_expected_rank(int case_num, int size, int rank)
{
    int color, key;
    int new_rank = 0;

    get_split_case(case_num, rank, size, &color, &key);
    if (color == MPI_UNDEFINED)
        return -1;

    for (int i = 0; i < size; i++) {
        if (split_rank_precedes(case_num, size, i, rank)) {
            new_rank++;
        }
    }

    return new_rank;
}

static int get_expected_size(int case_num, int size, int rank)
{
    int color;
    int key;
    int new_size = 0;

    get_split_case(case_num, rank, size, &color, &key);
    (void) key;
    if (color == MPI_UNDEFINED)
        return 0;

    for (int i = 0; i < size; i++) {
        int other_color, other_key;

        get_split_case(case_num, i, size, &other_color, &other_key);
        if (other_color == color) {
            new_size++;
        }
    }

    return new_size;
}

static void fill_expected_order(int case_num, int size, int rank, int expected_size,
                                int *expected_order)
{
    int color;
    int key;

    get_split_case(case_num, rank, size, &color, &key);
    (void) key;
    for (int i = 0; i < expected_size; i++) {
        expected_order[i] = -1;
    }

    for (int i = 0; i < size; i++) {
        int other_color, other_key;

        get_split_case(case_num, i, size, &other_color, &other_key);
        if (other_color == color) {
            int new_rank = get_expected_rank(case_num, size, i);
            expected_order[new_rank] = i;
        }
    }
}

static void report_error(int *errs, const char *msg, int case_num, int rank,
                         int expected, int actual)
{
    if (*errs < ERRLIMIT) {
        printf("case=%d world_rank=%d %s expected=%d actual=%d\n",
               case_num, rank, msg, expected, actual);
    }
    (*errs)++;
}

int main(int argc, char **argv)
{
    int errs = 0;
    int rank, size;

    MTest_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (int case_num = 0; case_num < NUM_CASES; case_num++) {
        MPI_Comm splitcomm;
        int color, key;

        get_split_case(case_num, rank, size, &color, &key);
        MPI_Comm_split(MPI_COMM_WORLD, color, key, &splitcomm);

        if (color == MPI_UNDEFINED) {
            if (splitcomm != MPI_COMM_NULL) {
                report_error(&errs, "expected MPI_COMM_NULL", case_num, rank, 1, 0);
                MPI_Comm_free(&splitcomm);
            }
            continue;
        }

        if (splitcomm == MPI_COMM_NULL) {
            report_error(&errs, "unexpected MPI_COMM_NULL", case_num, rank, 0, 1);
            continue;
        }

        int new_rank, new_size;
        int expected_rank = get_expected_rank(case_num, size, rank);
        int expected_size = get_expected_size(case_num, size, rank);

        MPI_Comm_rank(splitcomm, &new_rank);
        MPI_Comm_size(splitcomm, &new_size);

        if (new_rank != expected_rank) {
            report_error(&errs, "new rank mismatch", case_num, rank, expected_rank, new_rank);
        }
        if (new_size != expected_size) {
            report_error(&errs, "new size mismatch", case_num, rank, expected_size, new_size);
        }

        int *actual_order = malloc(new_size * sizeof(int));
        int *expected_order = malloc(expected_size * sizeof(int));
        if (!actual_order || !expected_order) {
            fprintf(stderr, "out of memory\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Allgather(&rank, 1, MPI_INT, actual_order, 1, MPI_INT, splitcomm);
        fill_expected_order(case_num, size, rank, expected_size, expected_order);

        if (new_size == expected_size) {
            for (int i = 0; i < new_size; i++) {
                if (actual_order[i] != expected_order[i]) {
                    report_error(&errs, "rank order mismatch", case_num, rank,
                                 expected_order[i], actual_order[i]);
                    break;
                }
            }
        }

        free(actual_order);
        free(expected_order);
        MPI_Comm_free(&splitcomm);
    }

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
