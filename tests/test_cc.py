from clump_tracker_rust import compute_adjacency_cartesian, compute_cc
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

def _compute_adjacency_cartesian_ref(indexes,x,y,z,max_distance):
    """
    computes and returns adjacency matrix
    this is a reference implementation for
    testing purposes only 
    """
    adj = np.zeros((len(indexes),len(indexes)),dtype=bool)

    for i in range(len(indexes)):
        for j in range(len(indexes)):
            d2 = (x[indexes[i][0]]-x[indexes[j][0]])**2+(y[indexes[i][1]]-y[indexes[j][1]])**2+(z[indexes[i][2]]-z[indexes[j][2]])**2                    
            adj[i,j] = (d2 <= max_distance*max_distance)
    return adj

def _compute_cc_ref(indexes,x,y,z,max_distance):
    """
    indexes : list of coordinates (array indexes)
    
    this function checks the distance between all pairs in indexes, if it is less than `max_distance `
    they are of the same clump

    this is a reference implementation for
    testing purposes only 
    """

    clumps = []

    deja_vus = np.zeros((len(indexes)),dtype=bool)
    a_visiter = np.zeros((len(indexes)),dtype=bool)
    composante_connexes = []

    adj = _compute_adjacency_cartesian_ref(indexes,x,y,z,max_distance)

    # ajoute les voisins de p0 dans a_visiter
    for i,_ in enumerate(indexes):
        a_visiter = adj[i]
        a_visiter[i] = False
        deja_vus[i] = True
        a_visiter = np.logical_and(a_visiter,np.logical_not(deja_vus))
        composante_connexes.append([i])
        while np.sum(a_visiter)>0:# il reste des gens a visiter
            for j,c in enumerate(indexes):
                #print(f"{i = }, {j = }, {a_visiter = }, {deja_vus = })")
                #time.sleep(1)
                if not deja_vus[j]:
                    a_visiter += adj[j]
                    a_visiter[j] = False
                    deja_vus[j] = True
                    a_visiter = np.logical_and(a_visiter,np.logical_not(deja_vus))
                    composante_connexes[-1].append(j)
        if np.sum(deja_vus) == len(indexes):
            break
            
    return composante_connexes
        


@pytest.mark.parametrize("indexes", 
[
    [[0,0,0],[0,1,0],[1,0,1]],
    [[0,0,0],[49,19,4]],
    [[i,0,0] for i in range(50)]+[[i,0,4] for i in range(50)]


])

def test_adjacency(indexes):
    print(indexes)
    x = np.linspace(0,10,50)
    y = np.linspace(0,5,20) 
    z = np.linspace(0,1,5)

    assert_array_equal(compute_adjacency_cartesian(indexes,x,y,z,1.),_compute_adjacency_cartesian_ref(indexes,x,y,z,1.))

@pytest.mark.parametrize("indexes", 
[
    [[0,0,0],[0,1,0],[1,0,1]],
    [[0,0,0],[49,19,4]],
    [[i,0,0] for i in range(50)]+[[i,0,4] for i in range(50)]


])
def test_cc(indexes):
    print(indexes)
    x = np.linspace(0,10,50)
    y = np.linspace(0,5,20) 
    z = np.linspace(0,1,5)

    assert_array_equal(compute_cc(indexes,x,y,z,1.,"cartesian"),_compute_cc_ref(indexes,x,y,z,1.))
@pytest.mark.parametrize("indexes", 
[
    [[0,0,0],[0,1,0],[1,0,1]],
    [[0,0,0],[49,19,4]],
    [[i,0,0] for i in range(50)]+[[i,0,4] for i in range(50)]


])
@pytest.mark.xfail
def test_cc_not_implemented(indexes):
    print(indexes)
    x = np.linspace(0,10,50)
    y = np.linspace(0,5,20) 
    z = np.linspace(0,1,5)

    assert_array_equal(compute_cc(indexes,x,y,z,1.,"polar"),_compute_cc_ref(indexes,x,y,z,1.))
