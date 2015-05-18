# -*- coding: utf-8 -*-
"""
   Copyright 2015 Creare

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import numpy as np
cimport numpy as np


#==============================================================================
# Drain a single array's connections: If connected to start, change value
#==============================================================================

def drain_connections(np.ndarray[int, ndim=1, cast=True] arr,
                      np.ndarray[int, ndim=1, cast=True] ids,
                       np.ndarray[int, ndim=1] indptr,
                      np.ndarray[int, ndim=1] indices, int set_to=0):
    cdef int n_ids = ids.size
    cdef int n_A = indices.size

    cdef np.ndarray[int, ndim=1] ids_old = np.zeros(n_ids, dtype=int)

    _drain_connections(&(arr[0]), &(ids[0]), &(ids_old[0]),
                       &(indptr[0]), &(indices[0]), n_ids, n_A, set_to)
    return arr


cdef void _drain_connections(int *arr,
                             int *ids, int *ids_old, int *indptr, int *indices,
                             int n_ids, int n_A, int tf):
    cdef int i = 0
    cdef int j = 0
    cdef int keep_going = 1
    cdef int *tmp
    while keep_going: #If I use ids.sum() > 0 then I might get stuck in circular references.
#        print "c",
        # switch pointers
        tmp = ids_old
        ids_old = ids
        ids = tmp

        _zero_arr(ids, n_ids)
        for i in xrange(n_ids):
            if ids_old[i] == 0:
                continue #If this id is not active, just skip the next part
            for j in xrange(indptr[i], indptr[i + 1]):
                row_id = indices[j]
                ids[row_id] += arr[row_id] != tf
                arr[row_id] = tf

        keep_going = _check_id_changed(ids, ids_old, n_ids)


#==============================================================================
# Update area using the correct weighting factors and not double-dipping
#==============================================================================
def drain_area(np.ndarray[double, ndim=1] area,
               np.ndarray[int, ndim=1, cast=True] done,
               np.ndarray[int, ndim=1, cast=True] ids,
               np.ndarray[int, ndim=1] col_indptr,
               np.ndarray[int, ndim=1] col_indices,
               np.ndarray[double, ndim=1] col_data,
               np.ndarray[int, ndim=1] row_indptr,
               np.ndarray[int, ndim=1] row_indices,
               int n_rows, int n_cols,
               np.ndarray[double, ndim=1, cast=True] edge_todo=None,
               np.ndarray[double, ndim=1, cast=True] edge_todo_no_mask=None,
               skip_edge=0):
    cdef int n_ids = ids.size
    cdef int n_A = col_indices.size

    cdef int do_edge_todo
    cdef np.ndarray[double, ndim=1] edge_todo2 = np.zeros(1, dtype=float)
    if edge_todo is None:
        edge_todo = edge_todo2
        do_edge_todo = 0
    else:
        do_edge_todo = 1

    cdef int do_edge_todo_no_mask
    cdef np.ndarray[double, ndim=1] edge_todo_no_mask2 = np.zeros(1, dtype=float)
    if edge_todo_no_mask is None:
        edge_todo_no_mask = edge_todo_no_mask2
        do_edge_todo_no_mask = 0
    else:
        do_edge_todo_no_mask = 1

    cdef np.ndarray[int, ndim=1] ids_old = np.zeros(n_ids, dtype=int)

    _drain_area(&(area[0]), &(done[0]), &(ids[0]), &(ids_old[0]),
                       &(col_indptr[0]), &(col_indices[0]), &(col_data[0]),
                       &(row_indptr[0]), &(row_indices[0]),
                       n_rows, n_cols, n_ids, n_A,
                       &(edge_todo[0]), do_edge_todo,
                       &(edge_todo_no_mask[0]), do_edge_todo_no_mask,
                       skip_edge)
    return area, done, edge_todo, edge_todo_no_mask


cdef void _drain_area(double *area, int* done,
                             int *ids, int *ids_old,
                             int *col_indptr, int *col_indices, double *col_data,
                             int *row_indptr, int *row_indices,
                             int n_rows, int n_cols, int n_ids, int n_A,
                             double* edge_todo, int do_edge_todo,
                             double* edge_todo_no_mask, int do_edge_todo_no_mask,
                             int skip_edge):
    cdef int i = 0
    cdef int j = 0
    cdef int keep_going = 1
    cdef int *tmp
    cdef double factor = 0
    cdef int wait_for_neighbors = 0
    while keep_going: # If I use ids.sum() > 0 then I might get stuck in circular references.
#        print "oc",
        # Set the points that are about to be drained as done
        # This has to be done before the next loop because the next candidates
        # check to make sure that all the points that drain into them are done
        for i in xrange(n_ids):
            if ids[i] > 0:
                done[i] = 1
#                print i, done[i]

        # switch pointers
        tmp = ids_old
        ids_old = ids
        ids = tmp

        # Initialize array to zero for the points that will be drained next round
        _zero_arr(ids, n_ids)

        for i in xrange(n_ids):
            if ids_old[i] == 0:
                continue # If this id is not active, just skip the next part
            for j in xrange(col_indptr[i], col_indptr[i + 1]):
                row_id = col_indices[j]
                factor = col_data[j]
                # check if this is on the edge, if so, do not modify it and go
                # to the next candidate
                if ((skip_edge or (done[row_id])) and \
                        _check_id_on_edge(row_id, n_rows, n_cols)):
                    continue

                area[row_id] += area[i] * factor
                if do_edge_todo:
                    edge_todo[row_id] += edge_todo[i] * factor
                if do_edge_todo_no_mask:
                    edge_todo_no_mask[row_id] += edge_todo_no_mask[i] * factor

                # If the point that has just been drained into has all its
                # source points marked as done, then it can be drained next
                # round
                wait_for_neighbors = 0
                for k in xrange(row_indptr[row_id], row_indptr[row_id+1]):
                    col_id = row_indices[k]
#                    print 'row', row_id, 'col', col_id, 'done', done[col_id]
                    if done[col_id] < 1:
                        wait_for_neighbors = 1
                        break
#                print wait_for_neighbors,
                if wait_for_neighbors == 0:
                    ids[row_id] = 1

                if do_edge_todo:
                    done[i] = 1

        keep_going = _check_id_changed(ids, ids_old, n_ids)

#==============================================================================
# Helper functions
#==============================================================================

cdef int _check_id_changed(int *ids, int *ids_old, int n_ids):
    cdef int i
    cdef int ret = 0
    for i in xrange(n_ids):
        if ids[i] != ids_old[i]:
            ret = 1
            return ret
    return ret

cdef _zero_arr(int *arr, int n):
    cdef int i
    for i in xrange(n):
        arr[i] = 0

cdef int _check_id_on_edge(int row_id, int n_rows, int n_cols):
    # Everything should have been done using c-order.
    # So, if the id is on the top edge, it will have a value less than n_cols
    if row_id < n_cols:
        return 1
    # if it is on the bottom row, it will have an id greater or equal than
    # the maximum id minus the number of columns
    elif row_id >= (n_cols * n_rows - n_cols):
        return 1
    # if we divide the id by the number of columns and the remainder is 0,
    # then the id is in the first column, or the left edge
    elif (row_id % n_cols) == 0:
        return 1
    # if we divide the id by the number of columns and the remainder is one
    # less than the number of columns, then the id is in the last column, or
    # the right column
    elif row_id % n_cols == (n_cols - 1):
        return 1
    else: # Otherwise, it's not on an edge
        return 0

