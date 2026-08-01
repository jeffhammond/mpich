/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

enum {
    SPLIT_COLOR = 0,
    SPLIT_KEY = 1,
    SPLIT_RANK = 2,
    SPLIT_NFIELDS = 3
};

enum {
    SPLITRANK_COLOR = 0,
    SPLITRANK_NEW_RANK = 1,
    SPLITRANK_RANK = 2,
    SPLITRANK_NFIELDS = 3
};

enum {
    SPLITPAIR_COLOR = 0,
    SPLITPAIR_NEW_RANK = 1,
    SPLITPAIR_NFIELDS = 2
};

static int comm_split_count_rank(MPIR_Comm * local_comm_ptr, int size, int color, int key,
                                 int rank, int *new_rank, int *new_size)
{
    int mpi_errno = MPI_SUCCESS;
    int sendinfo[SPLIT_NFIELDS] = { color, key, rank };
    int recvinfo[SPLIT_NFIELDS];
    int local_rank = local_comm_ptr->rank;
    int left = (local_rank + size - 1) % size;
    int right = (local_rank + 1) % size;

    *new_size = 0;
    if (color != MPI_UNDEFINED) {
        *new_rank = 0;
        *new_size = 1;
    } else {
        *new_rank = MPI_UNDEFINED;
    }

    for (int i = 1; i < size; i++) {
        mpi_errno = MPIC_Sendrecv(sendinfo, SPLIT_NFIELDS, MPIR_INT_INTERNAL,
                                  right, MPIR_ALLGATHER_TAG,
                                  recvinfo, SPLIT_NFIELDS, MPIR_INT_INTERNAL,
                                  left, MPIR_ALLGATHER_TAG,
                                  local_comm_ptr, MPI_STATUS_IGNORE, MPIR_COLL_ATTR_SYNC);
        MPIR_ERR_CHECK(mpi_errno);

        if (color != MPI_UNDEFINED) {
            if (recvinfo[SPLIT_COLOR] == color) {
                (*new_size)++;
                if (recvinfo[SPLIT_KEY] < key ||
                    (recvinfo[SPLIT_KEY] == key && recvinfo[SPLIT_RANK] < rank)) {
                    (*new_rank)++;
                }
            }
        }

        sendinfo[SPLIT_COLOR] = recvinfo[SPLIT_COLOR];
        sendinfo[SPLIT_KEY] = recvinfo[SPLIT_KEY];
        sendinfo[SPLIT_RANK] = recvinfo[SPLIT_RANK];
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_build_local_ranks(MPIR_Comm * local_comm_ptr, int size, int color,
                                        int my_new_rank, int rank, int new_size, int *local_ranks)
{
    int mpi_errno = MPI_SUCCESS;
    int sendinfo[SPLITRANK_NFIELDS] = { color, my_new_rank, rank };
    int recvinfo[SPLITRANK_NFIELDS];
    int local_rank = local_comm_ptr->rank;
    int left = (local_rank + size - 1) % size;
    int right = (local_rank + 1) % size;

    if (local_ranks && color != MPI_UNDEFINED) {
        MPIR_Assert(my_new_rank >= 0 && my_new_rank < new_size);
        local_ranks[my_new_rank] = rank;
    }

    for (int i = 1; i < size; i++) {
        mpi_errno = MPIC_Sendrecv(sendinfo, SPLITRANK_NFIELDS, MPIR_INT_INTERNAL,
                                  right, MPIR_ALLGATHER_TAG,
                                  recvinfo, SPLITRANK_NFIELDS, MPIR_INT_INTERNAL,
                                  left, MPIR_ALLGATHER_TAG,
                                  local_comm_ptr, MPI_STATUS_IGNORE, MPIR_COLL_ATTR_SYNC);
        MPIR_ERR_CHECK(mpi_errno);

        if (local_ranks && recvinfo[SPLITRANK_COLOR] == color) {
            int new_rank = recvinfo[SPLITRANK_NEW_RANK];
            MPIR_Assert(new_rank >= 0 && new_rank < new_size);
            local_ranks[new_rank] = recvinfo[SPLITRANK_RANK];
        }

        sendinfo[SPLITRANK_COLOR] = recvinfo[SPLITRANK_COLOR];
        sendinfo[SPLITRANK_NEW_RANK] = recvinfo[SPLITRANK_NEW_RANK];
        sendinfo[SPLITRANK_RANK] = recvinfo[SPLITRANK_RANK];
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

static int comm_split_count_remote_size(const int *table, int size, int color)
{
    int new_size = 0;

    for (int i = 0; i < size; i++) {
        const int *entry = &table[i * SPLITPAIR_NFIELDS];
        if (entry[SPLITPAIR_COLOR] == color) {
            new_size++;
        }
    }

    return new_size;
}

static void comm_split_build_remote_ranks(const int *table, int size, int color,
                                          int new_size, int *remote_ranks)
{
    for (int i = 0; i < size; i++) {
        const int *entry = &table[i * SPLITPAIR_NFIELDS];
        if (entry[SPLITPAIR_COLOR] == color) {
            int new_rank = entry[SPLITPAIR_NEW_RANK];
            MPIR_Assert(new_rank >= 0 && new_rank < new_size);
            remote_ranks[new_rank] = i;
        }
    }
}

int MPIR_Comm_split_impl(MPIR_Comm * comm_ptr, int color, int key, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *local_comm_ptr;
    int *remotetable = 0;
    int *local_ranks = NULL, *remote_ranks = NULL;
    int rank, size, remote_size, new_size, new_remote_size, my_new_rank;
    int in_newcomm;             /* TRUE iff *newcomm should be populated */
    int new_context_id, remote_context_id;
    MPIR_CHKLMEM_DECL();

    rank = comm_ptr->rank;
    size = comm_ptr->local_size;
    remote_size = comm_ptr->remote_size;

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

    /* Step 1: Count how many processes have our same color, and use a ring
     * pass over the input values to determine this rank's location in the new
     * communicator without first replicating all color/key pairs. */
    mpi_errno = comm_split_count_rank(local_comm_ptr, size, color, key, rank,
                                      &my_new_rank, &new_size);
    MPIR_ERR_CHECK(mpi_errno);

    /* If we're an intercomm, we need to get the remote rank table before
     * deciding whether to create the communicator. */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        /* For the remote group, the situation is more complicated.
         * We need to find the size of our "partner" group in the
         * remote comm.  The easiest way (in terms of code) is for
         * every process to receive the color and precomputed rank information
         * for the remote group.
         */
        int mypair[SPLITPAIR_NFIELDS] = { color, my_new_rank };
        MPIR_CHKLMEM_MALLOC(remotetable, remote_size * SPLITPAIR_NFIELDS * sizeof(int));
        /* This is an intercommunicator allgather */

        mpi_errno = MPIR_Allgather_fallback(mypair, SPLITPAIR_NFIELDS, MPIR_INT_INTERNAL,
                                            remotetable, SPLITPAIR_NFIELDS, MPIR_INT_INTERNAL,
                                            comm_ptr, MPIR_COLL_ATTR_SYNC);
        MPIR_ERR_CHECK(mpi_errno);

        /* Each process can now match its color with the entries in the table */
        new_remote_size = comm_split_count_remote_size(remotetable, remote_size, color);
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

    if (in_newcomm) {
        MPIR_CHKLMEM_MALLOC(local_ranks, new_size * sizeof(int));
    }
    mpi_errno = comm_split_build_local_ranks(local_comm_ptr, size, color, my_new_rank, rank,
                                            new_size, local_ranks);
    MPIR_ERR_CHECK(mpi_errno);

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

        /* Step 4: Order the processes by key values.  The local rank list was
         * built by circulating each process's computed rank in input rank
         * order. */

        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
            (*newcomm_ptr)->rank = my_new_rank;

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->local_group, new_size, local_ranks,
                                             &(*newcomm_ptr)->local_group);
            MPIR_ERR_CHECK(mpi_errno);

            MPIR_CHKLMEM_MALLOC(remote_ranks, new_remote_size * sizeof(int));

            comm_split_build_remote_ranks(remotetable, remote_size, color,
                                          new_remote_size, remote_ranks);

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->remote_group,
                                             new_remote_size, remote_ranks,
                                             &(*newcomm_ptr)->remote_group);
            MPIR_ERR_CHECK(mpi_errno);

            (*newcomm_ptr)->context_id = remote_context_id;
            (*newcomm_ptr)->remote_size = new_remote_size;
            (*newcomm_ptr)->local_comm = 0;
            (*newcomm_ptr)->is_low_group = comm_ptr->is_low_group;

        } else {
            /* INTRA Communicator */
            (*newcomm_ptr)->context_id = (*newcomm_ptr)->recvcontext_id;
            (*newcomm_ptr)->remote_size = new_size;

            (*newcomm_ptr)->rank = my_new_rank;

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->local_group, new_size, local_ranks,
                                             &(*newcomm_ptr)->local_group);
            MPIR_ERR_CHECK(mpi_errno);
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
