/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

typedef struct splittype {
    int color, key;
} splittype;

/* Same as splittype but with an additional field to stabilize the qsort.  We
 * could just use one combined type, but using separate types simplifies the
 * allgather step. */
typedef struct sorttype {
    int color, key;
    int orig_idx;
} sorttype;

#if defined(HAVE_QSORT)
static int sorttype_compare(const void *v1, const void *v2)
{
    const sorttype *s1 = v1;
    const sorttype *s2 = v2;

    if (s1->key > s2->key)
        return 1;
    if (s1->key < s2->key)
        return -1;

    /* (s1->key == s2->key), maintain original order */
    if (s1->orig_idx > s2->orig_idx)
        return 1;
    else if (s1->orig_idx < s2->orig_idx)
        return -1;

    /* --BEGIN ERROR HANDLING-- */
    return 0;   /* should never happen */
    /* --END ERROR HANDLING-- */
}
#endif

/* Sort the entries in keytable into increasing order by key.  A stable
   sort should be used in case the key values are not unique. */
static void MPIU_Sort_inttable(sorttype * keytable, int size)
{
    sorttype tmp;
    int i, j;

#if defined(HAVE_QSORT)
    /* temporary switch for profiling performance differences */
    if (MPIR_CVAR_COMM_SPLIT_USE_QSORT) {
        /* qsort isn't a stable sort, so we have to enforce stability by keeping
         * track of the original indices */
        for (i = 0; i < size; ++i)
            keytable[i].orig_idx = i;
        qsort(keytable, size, sizeof(sorttype), &sorttype_compare);
    } else
#endif
    {
        /* --BEGIN USEREXTENSION-- */
        /* fall through to insertion sort if qsort is unavailable/disabled */
        for (i = 1; i < size; ++i) {
            tmp = keytable[i];
            j = i - 1;
            while (1) {
                if (keytable[j].key > tmp.key) {
                    keytable[j + 1] = keytable[j];
                    j = j - 1;
                    if (j < 0)
                        break;
                } else {
                    break;
                }
            }
            keytable[j + 1] = tmp;
        }
        /* --END USEREXTENSION-- */
    }
}

typedef struct split_record {
    int color, key, orig_rank;
} split_record;

typedef struct split_segment {
    int total, first_color, first_count, last_color, last_count, all_same;
} split_segment;

static int split_record_compare(const split_record * a, const split_record * b)
{
    if (a->color > b->color)
        return 1;
    if (a->color < b->color)
        return -1;
    if (a->key > b->key)
        return 1;
    if (a->key < b->key)
        return -1;
    return 0;
}

static split_record split_record_median(split_record * records, int count)
{
    MPIR_Assert(count > 0);

    if (count == 1)
        return records[0];

    if (count == 2) {
        if (split_record_compare(&records[0], &records[1]) <= 0)
            return records[0];
        else
            return records[1];
    }

    if (split_record_compare(&records[0], &records[1]) > 0) {
        split_record tmp = records[0];
        records[0] = records[1];
        records[1] = tmp;
    }
    if (split_record_compare(&records[1], &records[2]) > 0) {
        split_record tmp = records[1];
        records[1] = records[2];
        records[2] = tmp;
    }
    if (split_record_compare(&records[0], &records[1]) > 0) {
        split_record tmp = records[0];
        records[0] = records[1];
        records[1] = tmp;
    }

    return records[1];
}

static split_segment split_segment_make(int color)
{
    split_segment segment;

    segment.total = 1;
    segment.first_color = color;
    segment.first_count = 1;
    segment.last_color = color;
    segment.last_count = 1;
    segment.all_same = TRUE;

    return segment;
}

static split_segment split_segment_empty(void)
{
    split_segment segment;

    segment.total = 0;
    segment.first_color = MPI_UNDEFINED;
    segment.first_count = 0;
    segment.last_color = MPI_UNDEFINED;
    segment.last_count = 0;
    segment.all_same = TRUE;

    return segment;
}

static split_segment split_segment_combine(split_segment left, split_segment right)
{
    split_segment out;

    if (left.total == 0)
        return right;
    if (right.total == 0)
        return left;

    out.total = left.total + right.total;
    out.first_color = left.first_color;
    out.first_count = left.first_count;
    out.last_color = right.last_color;
    out.last_count = right.last_count;
    out.all_same = left.all_same && right.all_same && left.first_color == right.first_color;

    if (left.all_same && left.last_color == right.first_color) {
        out.first_count = left.total + right.first_count;
    }
    if (right.all_same && left.last_color == right.first_color) {
        out.last_count = right.total + left.last_count;
    }

    return out;
}

static int comm_split_range_bcast_ints(MPIR_Comm * comm_ptr, int lo, int hi, int *buf, int count,
                                       int tag)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int relrank = rank - lo;
    int range_size = hi - lo + 1;

    for (int mask = 1; mask < range_size; mask <<= 1) {
        if (relrank < mask) {
            int dst_rel = relrank + mask;
            if (dst_rel < range_size) {
                mpi_errno = MPIC_Send(buf, count, MPIR_INT_INTERNAL, lo + dst_rel, tag,
                                      comm_ptr, MPIR_COLL_ATTR_SYNC);
                MPIR_ERR_CHECK(mpi_errno);
            }
        } else if (relrank < 2 * mask) {
            int src_rel = relrank - mask;
            mpi_errno = MPIC_Recv(buf, count, MPIR_INT_INTERNAL, lo + src_rel, tag,
                                  comm_ptr, MPI_STATUS_IGNORE);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_range_select_pivot(MPIR_Comm * comm_ptr, int lo, int hi,
                                         split_record local_record, split_record * pivot)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int relrank = rank - lo;
    int range_size = hi - lo + 1;
    int active_count = range_size;
    int stride = 1;
    split_record candidate = local_record;

    while (active_count > 1) {
        if (relrank % stride == 0) {
            int candidate_idx = relrank / stride;

            if (candidate_idx < active_count) {
                int group_pos = candidate_idx % 3;
                int leader_idx = candidate_idx - group_pos;
                int leader_rank = lo + leader_idx * stride;

                if (group_pos == 0) {
                    split_record records[3];
                    int record_count = 1;

                    records[0] = candidate;
                    for (int i = 1; i < 3; i++) {
                        int child_idx = leader_idx + i;
                        if (child_idx < active_count) {
                            mpi_errno = MPIC_Recv(&records[record_count], 3, MPIR_INT_INTERNAL,
                                                  lo + child_idx * stride, MPIR_REDUCE_TAG,
                                                  comm_ptr, MPI_STATUS_IGNORE);
                            MPIR_ERR_CHECK(mpi_errno);
                            record_count++;
                        }
                    }
                    candidate = split_record_median(records, record_count);
                } else {
                    mpi_errno = MPIC_Send(&candidate, 3, MPIR_INT_INTERNAL, leader_rank,
                                          MPIR_REDUCE_TAG, comm_ptr, MPIR_COLL_ATTR_SYNC);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
        }

        active_count = (active_count + 2) / 3;
        stride *= 3;
    }

    if (rank == lo)
        *pivot = candidate;

    mpi_errno = comm_split_range_bcast_ints(comm_ptr, lo, hi, (int *) pivot, 3, MPIR_BCAST_TAG);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_range_allreduce_sum3(MPIR_Comm * comm_ptr, int lo, int hi,
                                           const int local[3], int total[3])
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int relrank = rank - lo;
    int range_size = hi - lo + 1;
    int sent = FALSE;

    total[0] = local[0];
    total[1] = local[1];
    total[2] = local[2];

    for (int mask = 1; mask < range_size; mask <<= 1) {
        if (!sent) {
            if (relrank % (2 * mask) == 0) {
                int src_rel = relrank + mask;
                if (src_rel < range_size) {
                    int tmp[3];

                    mpi_errno = MPIC_Recv(tmp, 3, MPIR_INT_INTERNAL, lo + src_rel,
                                          MPIR_ALLREDUCE_TAG, comm_ptr, MPI_STATUS_IGNORE);
                    MPIR_ERR_CHECK(mpi_errno);
                    total[0] += tmp[0];
                    total[1] += tmp[1];
                    total[2] += tmp[2];
                }
            } else if (relrank % (2 * mask) == mask) {
                int dst_rel = relrank - mask;

                mpi_errno = MPIC_Send(total, 3, MPIR_INT_INTERNAL, lo + dst_rel,
                                      MPIR_ALLREDUCE_TAG, comm_ptr, MPIR_COLL_ATTR_SYNC);
                MPIR_ERR_CHECK(mpi_errno);
                sent = TRUE;
            }
        }
    }

    mpi_errno = comm_split_range_bcast_ints(comm_ptr, lo, hi, total, 3, MPIR_BCAST_TAG);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_range_exscan_sum3(MPIR_Comm * comm_ptr, int lo, int hi,
                                        const int local[3], int prefix[3])
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int relrank = rank - lo;
    int range_size = hi - lo + 1;
    int partial[3] = { local[0], local[1], local[2] };

    prefix[0] = 0;
    prefix[1] = 0;
    prefix[2] = 0;

    for (int mask = 1; mask < range_size; mask <<= 1) {
        int tmp[3] = { 0, 0, 0 };
        int dst = (relrank + mask < range_size) ? rank + mask : MPI_PROC_NULL;
        int src = (relrank >= mask) ? rank - mask : MPI_PROC_NULL;

        mpi_errno = MPIC_Sendrecv(partial, 3, MPIR_INT_INTERNAL, dst, MPIR_EXSCAN_TAG,
                                  tmp, 3, MPIR_INT_INTERNAL, src, MPIR_EXSCAN_TAG,
                                  comm_ptr, MPI_STATUS_IGNORE, MPIR_COLL_ATTR_SYNC);
        MPIR_ERR_CHECK(mpi_errno);

        if (src != MPI_PROC_NULL) {
            prefix[0] += tmp[0];
            prefix[1] += tmp[1];
            prefix[2] += tmp[2];
            partial[0] += tmp[0];
            partial[1] += tmp[1];
            partial[2] += tmp[2];
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_range_exscan_segment(MPIR_Comm * comm_ptr, int lo, int hi,
                                           int reverse, split_segment local,
                                           split_segment * prefix)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int relrank = reverse ? hi - rank : rank - lo;
    int range_size = hi - lo + 1;
    split_segment partial = local;

    *prefix = split_segment_empty();

    for (int mask = 1; mask < range_size; mask <<= 1) {
        split_segment tmp = split_segment_empty();
        int dst = MPI_PROC_NULL;
        int src = MPI_PROC_NULL;

        if (relrank + mask < range_size) {
            dst = reverse ? rank - mask : rank + mask;
        }
        if (relrank >= mask) {
            src = reverse ? rank + mask : rank - mask;
        }

        mpi_errno = MPIC_Sendrecv(&partial, 6, MPIR_INT_INTERNAL, dst, MPIR_EXSCAN_TAG,
                                  &tmp, 6, MPIR_INT_INTERNAL, src, MPIR_EXSCAN_TAG,
                                  comm_ptr, MPI_STATUS_IGNORE, MPIR_COLL_ATTR_SYNC);
        MPIR_ERR_CHECK(mpi_errno);

        if (src != MPI_PROC_NULL) {
            *prefix = split_segment_combine(tmp, *prefix);
            partial = split_segment_combine(tmp, partial);
        }
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_partition_scalable(MPIR_Comm * comm_ptr, int *lo, int *hi,
                                         split_record * record)
{
    int mpi_errno = MPI_SUCCESS;
    split_record pivot, new_record;
    int local_class[3] = { 0, 0, 0 };
    int prefix[3], total[3], dest;
    int cmp;

    mpi_errno = comm_split_range_select_pivot(comm_ptr, *lo, *hi, *record, &pivot);
    MPIR_ERR_CHECK(mpi_errno);

    cmp = split_record_compare(record, &pivot);
    if (cmp < 0) {
        local_class[0] = 1;
    } else if (cmp == 0) {
        local_class[1] = 1;
    } else {
        local_class[2] = 1;
    }

    mpi_errno = comm_split_range_exscan_sum3(comm_ptr, *lo, *hi, local_class, prefix);
    MPIR_ERR_CHECK(mpi_errno);
    mpi_errno = comm_split_range_allreduce_sum3(comm_ptr, *lo, *hi, local_class, total);
    MPIR_ERR_CHECK(mpi_errno);

    if (cmp < 0) {
        dest = *lo + prefix[0];
    } else if (cmp == 0) {
        dest = *lo + total[0] + prefix[1];
    } else {
        dest = *lo + total[0] + total[1] + prefix[2];
    }

    mpi_errno = MPIC_Sendrecv(record, 3, MPIR_INT_INTERNAL, dest, MPIR_ALLTOALL_TAG,
                              &new_record, 3, MPIR_INT_INTERNAL, MPI_ANY_SOURCE,
                              MPIR_ALLTOALL_TAG, comm_ptr, MPI_STATUS_IGNORE,
                              MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    *record = new_record;
    cmp = split_record_compare(record, &pivot);

    if (cmp < 0) {
        *hi = *lo + total[0] - 1;
    } else if (cmp == 0) {
        *lo = *lo + total[0];
        *hi = *lo + total[1] - 1;
    } else {
        *lo = *lo + total[0] + total[1];
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_rank_scalable(MPIR_Comm * comm_ptr, int color, int key,
                                    int *new_rank, int *new_size)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    int lo = 0, hi = size - 1;
    split_record record = { color, key, rank };
    split_segment local_segment, prefix, suffix;
    int result[2], my_result[2];

    while (lo < hi) {
        int old_lo = lo;
        int old_hi = hi;

        mpi_errno = comm_split_partition_scalable(comm_ptr, &lo, &hi, &record);
        MPIR_ERR_CHECK(mpi_errno);

        if (lo == old_lo && hi == old_hi)
            break;
    }

    local_segment = split_segment_make(record.color);
    mpi_errno = comm_split_range_exscan_segment(comm_ptr, 0, size - 1, FALSE, local_segment,
                                                &prefix);
    MPIR_ERR_CHECK(mpi_errno);
    mpi_errno = comm_split_range_exscan_segment(comm_ptr, 0, size - 1, TRUE, local_segment,
                                                &suffix);
    MPIR_ERR_CHECK(mpi_errno);

    if (record.color == MPI_UNDEFINED) {
        result[0] = MPI_UNDEFINED;
        result[1] = 0;
    } else {
        int before = (prefix.last_color == record.color) ? prefix.last_count : 0;
        int after = (suffix.last_color == record.color) ? suffix.last_count : 0;

        result[0] = before;
        result[1] = before + 1 + after;
    }

    mpi_errno = MPIC_Sendrecv(result, 2, MPIR_INT_INTERNAL, record.orig_rank,
                              MPIR_LOCALCOPY_TAG, my_result, 2, MPIR_INT_INTERNAL,
                              MPI_ANY_SOURCE, MPIR_LOCALCOPY_TAG, comm_ptr, MPI_STATUS_IGNORE,
                              MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    *new_rank = my_result[0];
    *new_size = my_result[1];

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_intra_scalable(MPIR_Comm * comm_ptr, int color, int key,
                                     MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    int rank = comm_ptr->rank;
    int size = comm_ptr->local_size;
    int new_size, my_new_rank, new_context_id;
    int in_newcomm;
    int *ranktable = NULL, *local_ranks = NULL;
    MPIR_CHKLMEM_DECL();

    mpi_errno = comm_split_rank_scalable(comm_ptr, color, key, &my_new_rank, &new_size);
    MPIR_ERR_CHECK(mpi_errno);

    in_newcomm = (color != MPI_UNDEFINED);

    MPIR_CHKLMEM_MALLOC(ranktable, 3 * size * sizeof(int));
    int myinfo[3] = { color, my_new_rank, rank };

    mpi_errno = MPIR_Allgather_fallback(myinfo, 3, MPIR_INT_INTERNAL,
                                        ranktable, 3, MPIR_INT_INTERNAL,
                                        comm_ptr, MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    if (in_newcomm) {
        MPIR_CHKLMEM_MALLOC(local_ranks, new_size * sizeof(int));
        for (int i = 0; i < size; i++) {
            int *entry = &ranktable[3 * i];
            if (entry[0] == color) {
                MPIR_Assert(entry[1] >= 0 && entry[1] < new_size);
                local_ranks[entry[1]] = entry[2];
            }
        }
    }

    mpi_errno = MPIR_Get_contextid_sparse(comm_ptr, &new_context_id, !in_newcomm);
    MPIR_ERR_CHECK(mpi_errno);
    MPIR_Assert(new_context_id != 0);

    *newcomm_ptr = NULL;

    if (in_newcomm) {
        mpi_errno = MPIR_Comm_create(newcomm_ptr);
        if (mpi_errno)
            goto fn_fail;

        (*newcomm_ptr)->recvcontext_id = new_context_id;
        (*newcomm_ptr)->local_size = new_size;
        (*newcomm_ptr)->comm_kind = MPIR_COMM_KIND__INTRACOMM;

        MPIR_Comm_set_session_ptr(*newcomm_ptr, comm_ptr->session_ptr);

        (*newcomm_ptr)->context_id = (*newcomm_ptr)->recvcontext_id;
        (*newcomm_ptr)->remote_size = new_size;
        (*newcomm_ptr)->rank = my_new_rank;

        mpi_errno = MPIR_Group_incl_impl(comm_ptr->local_group, new_size, local_ranks,
                                         &(*newcomm_ptr)->local_group);
        MPIR_ERR_CHECK(mpi_errno);

        MPID_THREAD_CS_ENTER(VCI, comm_ptr->mutex);
        (*newcomm_ptr)->errhandler = comm_ptr->errhandler;
        if (comm_ptr->errhandler) {
            MPIR_Errhandler_add_ref(comm_ptr->errhandler);
        }
        MPID_THREAD_CS_EXIT(VCI, comm_ptr->mutex);

        (*newcomm_ptr)->vcis_enabled = comm_ptr->vcis_enabled;
        mpi_errno = MPIR_Comm_commit(*newcomm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIR_Comm_split_impl(MPIR_Comm * comm_ptr, int color, int key, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *local_comm_ptr;
    splittype *table, *remotetable = 0;
    sorttype *keytable, *remotekeytable = 0;
    int rank, size, remote_size, i, new_size, new_remote_size,
        first_entry = 0, first_remote_entry = 0, *last_ptr;
    int in_newcomm;             /* TRUE iff *newcomm should be populated */
    int new_context_id, remote_context_id;
    MPIR_CHKLMEM_DECL();

    rank = comm_ptr->rank;
    size = comm_ptr->local_size;
    remote_size = comm_ptr->remote_size;

    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM &&
        MPIR_CVAR_COMM_SPLIT_SCALABLE_THRESHOLD >= 0 &&
        size >= MPIR_CVAR_COMM_SPLIT_SCALABLE_THRESHOLD) {
        return comm_split_intra_scalable(comm_ptr, color, key, newcomm_ptr);
    }

    /* Step 1: Find out what color and keys all of the processes have */
    MPIR_CHKLMEM_MALLOC(table, size * sizeof(splittype));
    table[rank].color = color;
    table[rank].key = key;

    /* Get the communicator to use in collectives on the local group of
     * processes */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        if (!comm_ptr->local_comm) {
            MPII_Setup_intercomm_localcomm(comm_ptr);
        }
        local_comm_ptr = comm_ptr->local_comm;
    } else {
        local_comm_ptr = comm_ptr;
    }
    /* Gather information on the local group of processes */
    mpi_errno = MPIR_Allgather_fallback(MPI_IN_PLACE, 2, MPIR_INT_INTERNAL,
                                        table, 2, MPIR_INT_INTERNAL,
                                        local_comm_ptr, MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    /* Step 2: How many processes have our same color? */
    new_size = 0;
    if (color != MPI_UNDEFINED) {
        /* Also replace the color value with the index of the *next* value
         * in this set.  The integer first_entry is the index of the
         * first element */
        last_ptr = &first_entry;
        for (i = 0; i < size; i++) {
            /* Replace color with the index in table of the next item
             * of the same color.  We use this to efficiently populate
             * the keyval table */
            if (table[i].color == color) {
                new_size++;
                *last_ptr = i;
                last_ptr = &table[i].color;
            }
        }
    }
    /* We don't need to set the last value to -1 because we loop through
     * the list for only the known size of the group */

    /* If we're an intercomm, we need to do the same thing for the remote
     * table, as we need to know the size of the remote group of the
     * same color before deciding to create the communicator */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        splittype mypair;
        /* For the remote group, the situation is more complicated.
         * We need to find the size of our "partner" group in the
         * remote comm.  The easiest way (in terms of code) is for
         * every process to essentially repeat the operation for the
         * local group - perform an (intercommunicator) all gather
         * of the color and rank information for the remote group.
         */
        MPIR_CHKLMEM_MALLOC(remotetable, remote_size * sizeof(splittype));
        /* This is an intercommunicator allgather */

        /* We must use a local splittype because we've already modified the
         * entries in table to indicate the location of the next rank of the
         * same color */
        mypair.color = color;
        mypair.key = key;
        mpi_errno = MPIR_Allgather_fallback(&mypair, 2, MPIR_INT_INTERNAL,
                                            remotetable, 2, MPIR_INT_INTERNAL,
                                            comm_ptr, MPIR_COLL_ATTR_SYNC);
        MPIR_ERR_CHECK(mpi_errno);

        /* Each process can now match its color with the entries in the table */
        new_remote_size = 0;
        last_ptr = &first_remote_entry;
        for (i = 0; i < remote_size; i++) {
            /* Replace color with the index in table of the next item
             * of the same color.  We use this to efficiently populate
             * the keyval table */
            if (remotetable[i].color == color) {
                new_remote_size++;
                *last_ptr = i;
                last_ptr = &remotetable[i].color;
            }
        }
        /* Note that it might find that there a now processes in the remote
         * group with the same color.  In that case, COMM_SPLIT will
         * return a null communicator */
    } else {
        /* Set the size of the remote group to the size of our group.
         * This simplifies the test below for intercomms with an empty remote
         * group (must create comm_null) */
        new_remote_size = new_size;
    }

    in_newcomm = (color != MPI_UNDEFINED && new_remote_size > 0);

    /* Step 3: Create the communicator */
    /* Collectively create a new context id.  The same context id will
     * be used by each (disjoint) collections of processes.  The
     * processes whose color is MPI_UNDEFINED will not influence the
     * resulting context id (by passing ignore_id==TRUE). */
    /* In the multi-threaded case, MPIR_Get_contextid_sparse assumes that the
     * calling routine already holds the single critical section */

    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTRACOMM) {
        mpi_errno = MPIR_Get_contextid_sparse(local_comm_ptr, &new_context_id, !in_newcomm);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_Assert(new_context_id != 0);
    } else {
        /* In the intercomm case, we need to exchange the context ids */
        /* We'll ask original root to do the exchange then bcast the result.
         * However, root may not "in_newcomm" (i.e. color is MPI_UNDEFINED).
         * For simplicity (and assuming non-critical), we always allocate new_context_id
         * on root and release it afterwards.
         */
        int ignore_id = (comm_ptr->rank == 0) ? 0 : !in_newcomm;
        mpi_errno = MPIR_Get_contextid_sparse(local_comm_ptr, &new_context_id, ignore_id);
        MPIR_ERR_CHECK(mpi_errno);
        MPIR_Assert(new_context_id != 0);

        if (comm_ptr->rank == 0) {
            mpi_errno = MPIC_Sendrecv(&new_context_id, 1, MPIR_CONTEXT_ID_T_DATATYPE, 0, 0,
                                      &remote_context_id, 1, MPIR_CONTEXT_ID_T_DATATYPE,
                                      0, 0, comm_ptr, MPI_STATUS_IGNORE, MPIR_COLL_ATTR_SYNC);
            MPIR_ERR_CHECK(mpi_errno);
            mpi_errno = MPIR_Bcast_fallback(&remote_context_id, 1, MPIR_CONTEXT_ID_T_DATATYPE, 0,
                                            local_comm_ptr, MPIR_COLL_ATTR_SYNC);
            MPIR_ERR_CHECK(mpi_errno);

            if (!in_newcomm) {
                MPIR_Free_contextid(new_context_id);
            }
        } else {
            /* Broadcast to the other members of the local group */
            mpi_errno = MPIR_Bcast_fallback(&remote_context_id, 1, MPIR_CONTEXT_ID_T_DATATYPE, 0,
                                            local_comm_ptr, MPIR_COLL_ATTR_SYNC);
            MPIR_ERR_CHECK(mpi_errno);
        }
    }

    *newcomm_ptr = NULL;

    /* Now, create the new communicator structure if necessary */
    if (in_newcomm) {

        mpi_errno = MPIR_Comm_create(newcomm_ptr);
        if (mpi_errno)
            goto fn_fail;

        (*newcomm_ptr)->recvcontext_id = new_context_id;
        (*newcomm_ptr)->local_size = new_size;
        (*newcomm_ptr)->comm_kind = comm_ptr->comm_kind;

        MPIR_Comm_set_session_ptr(*newcomm_ptr, comm_ptr->session_ptr);

        /* Other fields depend on whether this is an intercomm or intracomm */

        /* Step 4: Order the processes by their key values.  Sort the
         * list that is stored in table.  To simplify the sort, we
         * extract the table into a smaller array and sort that.
         * Also, store in the "color" entry the rank in the input communicator
         * of the entry. */
        MPIR_CHKLMEM_MALLOC(keytable, new_size * sizeof(sorttype));
        for (i = 0; i < new_size; i++) {
            keytable[i].key = table[first_entry].key;
            keytable[i].color = first_entry;
            first_entry = table[first_entry].color;
        }

        /* sort key table.  The "color" entry is the rank of the corresponding
         * process in the input communicator */
        MPIU_Sort_inttable(keytable, new_size);

        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
            MPIR_CHKLMEM_MALLOC(remotekeytable, new_remote_size * sizeof(sorttype));
            for (i = 0; i < new_remote_size; i++) {
                remotekeytable[i].key = remotetable[first_remote_entry].key;
                remotekeytable[i].color = first_remote_entry;
                first_remote_entry = remotetable[first_remote_entry].color;
            }

            /* sort key table.  The "color" entry is the rank of the
             * corresponding process in the input communicator */
            MPIU_Sort_inttable(remotekeytable, new_remote_size);

            int *local_ranks;
            local_ranks = MPL_malloc(new_size * sizeof(int), MPL_MEM_OTHER);
            MPIR_ERR_CHKANDJUMP(!local_ranks, mpi_errno, MPI_ERR_OTHER, "**nomem");

            for (i = 0; i < new_size; i++) {
                local_ranks[i] = keytable[i].color;
                if (keytable[i].color == comm_ptr->rank)
                    (*newcomm_ptr)->rank = i;
            }

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->local_group, new_size, local_ranks,
                                             &(*newcomm_ptr)->local_group);
            MPIR_ERR_CHECK(mpi_errno);
            MPL_free(local_ranks);

            /* For the remote group, the situation is more complicated.
             * We need to find the size of our "partner" group in the
             * remote comm.  The easiest way (in terms of code) is for
             * every process to essentially repeat the operation for the
             * local group - perform an (intercommunicator) all gather
             * of the color and rank information for the remote group.
             */
            /* We apply the same sorting algorithm to the entries that we've
             * found to get the correct order of the entries.
             *
             * Note that if new_remote_size is 0 (no matching processes with
             * the same color in the remote group), then MPI_COMM_SPLIT
             * is required to return MPI_COMM_NULL instead of an intercomm
             * with an empty remote group. */

            int *remote_ranks;
            remote_ranks = MPL_malloc(new_remote_size * sizeof(int), MPL_MEM_OTHER);
            MPIR_ERR_CHKANDJUMP(!remote_ranks, mpi_errno, MPI_ERR_OTHER, "**nomem");

            for (i = 0; i < new_remote_size; i++) {
                remote_ranks[i] = remotekeytable[i].color;
            }

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->remote_group,
                                             new_remote_size, remote_ranks,
                                             &(*newcomm_ptr)->remote_group);
            MPIR_ERR_CHECK(mpi_errno);
            MPL_free(remote_ranks);

            (*newcomm_ptr)->context_id = remote_context_id;
            (*newcomm_ptr)->remote_size = new_remote_size;
            (*newcomm_ptr)->local_comm = 0;
            (*newcomm_ptr)->is_low_group = comm_ptr->is_low_group;

        } else {
            /* INTRA Communicator */
            (*newcomm_ptr)->context_id = (*newcomm_ptr)->recvcontext_id;
            (*newcomm_ptr)->remote_size = new_size;

            int *local_ranks;
            local_ranks = MPL_malloc(new_size * sizeof(int), MPL_MEM_OTHER);
            MPIR_ERR_CHKANDJUMP(!local_ranks, mpi_errno, MPI_ERR_OTHER, "**nomem");

            for (i = 0; i < new_size; i++) {
                local_ranks[i] = keytable[i].color;
                if (keytable[i].color == comm_ptr->rank)
                    (*newcomm_ptr)->rank = i;
            }

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->local_group, new_size, local_ranks,
                                             &(*newcomm_ptr)->local_group);
            MPIR_ERR_CHECK(mpi_errno);
            MPL_free(local_ranks);
        }

        /* Inherit the error handler (if any) */
        MPID_THREAD_CS_ENTER(VCI, comm_ptr->mutex);
        (*newcomm_ptr)->errhandler = comm_ptr->errhandler;
        if (comm_ptr->errhandler) {
            MPIR_Errhandler_add_ref(comm_ptr->errhandler);
        }
        MPID_THREAD_CS_EXIT(VCI, comm_ptr->mutex);

        (*newcomm_ptr)->vcis_enabled = comm_ptr->vcis_enabled;
        mpi_errno = MPIR_Comm_commit(*newcomm_ptr);
        MPIR_ERR_CHECK(mpi_errno);
    }

  fn_exit:
    MPIR_CHKLMEM_FREEALL();
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
