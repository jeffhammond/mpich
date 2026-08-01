/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpiimpl.h"

enum {
    SPLITPAIR_COLOR = 0,
    SPLITPAIR_KEY = 1,
    SPLITPAIR_NEW_RANK = 1,
    SPLITPAIR_NFIELDS = 2
};

static void comm_split_count_rank(const int *table, int size, int color, int key,
                                  int rank, int *new_rank, int *new_size)
{
    *new_size = 0;
    if (color != MPI_UNDEFINED) {
        *new_rank = 0;
    } else {
        *new_rank = MPI_UNDEFINED;
        return;
    }

    for (int i = 0; i < size; i++) {
        const int *entry = &table[i * SPLITPAIR_NFIELDS];

        if (entry[SPLITPAIR_COLOR] == color) {
            (*new_size)++;
            if (entry[SPLITPAIR_KEY] < key || (entry[SPLITPAIR_KEY] == key && i < rank)) {
                (*new_rank)++;
            }
        }
    }
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

static void comm_split_build_ranks(const int *table, int size, int color,
                                   int new_size, int *ranks)
{
    for (int i = 0; i < size; i++) {
        const int *entry = &table[i * SPLITPAIR_NFIELDS];
        if (entry[SPLITPAIR_COLOR] == color) {
            int new_rank = entry[SPLITPAIR_NEW_RANK];
            MPIR_Assert(new_rank >= 0 && new_rank < new_size);
            ranks[new_rank] = i;
        }
    }
}

int MPIR_Comm_split_impl(MPIR_Comm * comm_ptr, int color, int key, MPIR_Comm ** newcomm_ptr)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Comm *local_comm_ptr;
    int *localtable = 0, *remotetable = 0;
    int *local_ranks = NULL, *remote_ranks = NULL;
    int mypair[SPLITPAIR_NFIELDS];
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

    /* Step 1: Find out what color and keys all of the processes have */
    MPIR_CHKLMEM_MALLOC(localtable, size * SPLITPAIR_NFIELDS * sizeof(int));

    mypair[SPLITPAIR_COLOR] = color;
    mypair[SPLITPAIR_KEY] = key;
    mpi_errno = MPIR_Allgather_fallback(mypair, SPLITPAIR_NFIELDS, MPIR_INT_INTERNAL,
                                        localtable, SPLITPAIR_NFIELDS, MPIR_INT_INTERNAL,
                                        local_comm_ptr, MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    /* Step 2: Count how many processes have our same color, and determine this
     * rank's location in the new communicator by counting lower keys. */
    comm_split_count_rank(localtable, size, color, key, rank, &my_new_rank, &new_size);

    mypair[SPLITPAIR_COLOR] = color;
    mypair[SPLITPAIR_NEW_RANK] = my_new_rank;
    /* Gather information on the local group of processes */
    mpi_errno = MPIR_Allgather_fallback(mypair, SPLITPAIR_NFIELDS, MPIR_INT_INTERNAL,
                                        localtable, SPLITPAIR_NFIELDS, MPIR_INT_INTERNAL,
                                        local_comm_ptr, MPIR_COLL_ATTR_SYNC);
    MPIR_ERR_CHECK(mpi_errno);

    /* If we're an intercomm, we need to do the same thing for the remote
     * table, as we need to know the size of the remote group of the
     * same color before deciding to create the communicator */
    if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
        /* For the remote group, the situation is more complicated.
         * We need to find the size of our "partner" group in the
         * remote comm.  The easiest way (in terms of code) is for
         * every process to receive the color and precomputed rank information
         * for the remote group.
         */
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
        comm_split_build_ranks(localtable, size, color, new_size, local_ranks);
    }

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
         * built from each process's computed rank in input rank order. */

        if (comm_ptr->comm_kind == MPIR_COMM_KIND__INTERCOMM) {
            (*newcomm_ptr)->rank = my_new_rank;

            mpi_errno = MPIR_Group_incl_impl(comm_ptr->local_group, new_size, local_ranks,
                                             &(*newcomm_ptr)->local_group);
            MPIR_ERR_CHECK(mpi_errno);

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

            MPIR_CHKLMEM_MALLOC(remote_ranks, new_remote_size * sizeof(int));
            comm_split_build_ranks(remotetable, remote_size, color, new_remote_size, remote_ranks);

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
