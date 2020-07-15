# -*- coding: utf-8 -*-

import numpy as np
from skeleton3D import get_line_length
import collections


def skeleton_main_branch(skel):
    
    main_skeletons = []
    
    n_branch = len(skel)
    
    nodes_coord, branch_length = skeleton_info(skel)
    longest_branch = np.argmax(np.array(branch_length))
    
    graph = form_graph(nodes_coord)
    graph_e = {i: [i*2, i*2+1] for i in range(n_branch)}
    graph_v = {i: i/2 for i in range(2*n_branch)} 
    graph = rearrange_graph(graph, graph_v, graph_e)
    
    dec_junction = [i for i in graph if len(graph[i])>2]   
    dec_leaf = set()
    while len(graph)!=0:
        
        path = set()
        obj = {}
        
        for i in range(n_branch):
            if i not in graph_e:
                branch_length[i] = -np.inf
        
        path0 = np.argmax(np.array(branch_length))
        path.add(path0)
        
        root1, root2 = graph_e[path0]
        
        path1, dec_leaf1 = path_to_leaf(graph, root1, root2, graph_v, graph_e, skel, nodes_coord)
        path2, dec_leaf2 = path_to_leaf(graph, root2, root1, graph_v, graph_e, skel, nodes_coord)
        
        path.update(set(path1)); path.update(set(path2))
        dec_leaf.update(set(dec_leaf1)); dec_leaf.update(set(dec_leaf2))
        
        dec_nodes = detect_decomposing_nodes(path, dec_junction, list(dec_leaf), graph_e)

        graph, rgraph, rgraph_e, rgraph_v = update_graphs(graph, graph_v, graph_e, path)
        
        if cyclic_graph(rgraph, rgraph_v, rgraph_e) == False:

            parametrized_skel = skeleton_parametrization(skel, rgraph, rgraph_v, rgraph_e, nodes_coord)
            
            coord_dec_nodes = pair_dec_nodes(rgraph, dec_nodes, nodes_coord)        
    
            obj['skeleton'] = parametrized_skel
            obj['dec_nodes'] = coord_dec_nodes
                        
            if len(path) > 1:
                main_skeletons.append(obj)
                
            elif (len(path) == 1) & (list(path)[0] == longest_branch):
                main_skeletons.append(obj)
                    
    return main_skeletons



def cyclic_graph(g, rgraph_v, rgraph_e):
    
    Flag = False
    for i in g:
        for neighbour in g[i]:
            if len(set(rgraph_e[rgraph_v[neighbour]]) & set(g[i])) == 2:
                Flag = True
    return Flag



def pair_dec_nodes(g, dec_nodes, nodes_coord):

    p, visited = [], set()
    leaf = [i for i in g if len(g[i])==1]
    for i in dec_nodes:
        
        if i not in visited:
            visited.add(i)
        
            if i in leaf:
                p.append([i])
                
            elif len(set(g[i]) & set(dec_nodes)) == 0:
                p.append([i])
        
            else:
                for neighbour in g[i]:
                    if neighbour in dec_nodes:
                        if neighbour not in visited:
                            p.append([i, neighbour])
                            visited.add(neighbour)
                         
    coord_dec_nodes = [[nodes_coord[j] for j in i] for i in p]
    return coord_dec_nodes


def skeleton_parametrization(skel, n_graph, graph_v, graph_e, nodes_coord):

    parametrized_skel = np.empty((0,3))
    visited = set()
     
    leaf = [i for i in n_graph if len(n_graph[i])==1]   
    
    st_node = leaf.pop()
    visited.add(st_node)
    
    branch = graph_v[st_node]    
    o_branch = order_branch(skel[branch], nodes_coord[st_node], 'ascend')
    parametrized_skel = np.append(parametrized_skel, o_branch, axis=0)
    
    while st_node not in leaf:
        
        for neighbour in n_graph[st_node]:
            
            if neighbour not in visited:
                
                visited.add(neighbour)
                st_node = neighbour
                
                if neighbour not in graph_e[branch]:
                    branch = graph_v[st_node]
                    o_branch = order_branch(skel[branch], nodes_coord[st_node], 'ascend')
                    parametrized_skel = np.append(parametrized_skel, o_branch, axis=0)                           
    return parametrized_skel

def detect_decomposing_nodes(path, dec_junction, dec_leaf, graph_e):
     
    dec_nodes = set()
    for i in path:        
        intersect_j = set(graph_e[i]) & set(dec_junction); dec_nodes.update(intersect_j)
        intersect_l = set(graph_e[i]) & set(dec_leaf); dec_nodes.update(intersect_l)
    return list(dec_nodes)


def update_graphs(graph, graph_v, graph_e, path):
    
    all_path = set(graph_e.keys()); path_comp = all_path - path
    
    path_v = [j for i in path for j in graph_e[i]]
    path_comp_v = [j for i in path_comp for j in graph_e[i]]
    
    
    rgraph = {i:graph.pop(i) for i in path_v}
    rgraph_v = {i:graph_v.pop(i) for i in path_v}
    rgraph_e = {i:graph_e.pop(i) for i in path}
            
    graph = {i:list(set(graph[i]) - set(path_v)) for i in graph}
    rgraph = {i:list(set(rgraph[i]) - set(path_comp_v)) for i in rgraph}
    
    return graph, rgraph, rgraph_e, rgraph_v


def path_to_leaf(graph, st_node, counter_st_node, graph_v, graph_e, skel, nodes_coord):

    path = []
    dec_leaf = []
    leaf = [i for i in graph if len(graph[i])==1]
    visited = set([counter_st_node])
    
    while st_node not in leaf:
        st_node_old = st_node
        st_node, p, visited = detect_next_node(st_node, graph, visited, graph_v, graph_e, skel, nodes_coord)
        if st_node == st_node_old:
            leaf.append(st_node)
            ngbh = list(set(graph[st_node]) - set(graph_e[graph_v[st_node]]))
            dec_leaf.append(st_node)
            for i in ngbh:
                dec_leaf.append(i)
        else:
            path.append(p)
    return path, dec_leaf


def detect_next_node(st_node, graph, visited, graph_v, graph_e, skel, nodes_coord):
    
    orientation_diff = []
    comparing_branches = []
    
    for neighbour in graph[st_node]:
  
        if neighbour not in visited: 
            
            visited.add(neighbour) 
            
            branch_ref, branch_comp = graph_v[st_node], graph_v[neighbour]
                        
            skel_ref = skel[branch_ref]
            ord_skelRef = order_branch(skel_ref, nodes_coord[st_node], 'ascend')
            vs_ref = tangent_vector_sum(ord_skelRef)
            
            skel_comp = skel[branch_comp]
            ord_skelComp = order_branch(skel_comp, nodes_coord[neighbour], 'ascend')                                        
            vs_comp = tangent_vector_sum(ord_skelComp)
            
            angle = np.arccos(np.clip(np.dot(vs_ref, vs_comp), -1.0, 1.0))
            angle = angle*(180/np.pi)
            if angle > 90:
                orientation_diff.append(angle)
                comparing_branches.append(branch_comp)
    
    if len(comparing_branches) != 0:
        
        next_branch = comparing_branches[np.argmax(np.array(orientation_diff))]
        next_st_node = list(set(graph_e[next_branch]) - visited)
        if len(next_st_node) != 0:
            next_st_node = next_st_node[0]
        else:
            next_st_node = st_node

    else:
        next_branch = []
        next_st_node = st_node
    return next_st_node, next_branch, visited


def detect_junction_coordinates(end_points_coord, junctions_as_endpoint):
    
    junction_coordinates = []
    for junction in junctions_as_endpoint:        
         if len(junction)>2:
             junction_coordinate = np.array([end_points_coord[i] for i in list(junction)])
             junction_coordinates.append(junction_coordinate) 
    return junction_coordinates 
    
    
def branch_endpoints_inx(branch_inx):
    
    ep_inx = set([branch_inx*2, branch_inx*2 + 1])
    return ep_inx               
        

def skeleton_info(skel):
    
    end_point=[]
    branch_length=[]
    for branch in skel:
        end_point.append(branch[0])
        end_point.append(branch[-1])
        branch_length.append(get_line_length(branch))
    return end_point, branch_length


def end_points_cross_distance(end_points):
    
    l=len(end_points)
    dist=np.empty((l,0))
    for ep in end_points:
        euclidean_dist=np.expand_dims(np.sum((end_points-ep)**2,axis=1)**0.5,axis=1)
        dist=np.append(dist,euclidean_dist,axis=1)
    return dist


def form_graph(end_points):
    
    ep_cross_dist = end_points_cross_distance(end_points)
    sz = ep_cross_dist.shape 
    ep_cross_dist[np.diag_indices(sz[0])] = np.inf
    for i in range(1, sz[0], 2):
        ep_cross_dist[i,i-1] = np.inf
        ep_cross_dist[i-1,i] = np.inf
    
    graph = detect_fully_connected_graph(ep_cross_dist)      
    return graph


def detect_fully_connected_graph(ep_cross_dist):
    
    num_eps = ep_cross_dist.shape[0]
    junction_eps = np.empty((0,2), dtype=np.int)
    visited = set()
    
    while len(visited) != num_eps:
                          
        conn_dist = np.min(ep_cross_dist)
        
        if conn_dist == np.inf:
            visited, graph = fully_connected_tree(junction_eps, num_eps)
        else:
            min_ind = np.where(ep_cross_dist==conn_dist)
            ep_cross_dist[min_ind] = np.inf
            junction_ep = np.array(min_ind).T
            junction_eps = np.append(junction_eps, junction_ep, axis=0)
            visited, graph = fully_connected_tree(junction_eps, num_eps)
    return graph



def fully_connected_tree(junction_eps, num_eps):
      
    graph = {i: [i+1] for i in range(0, num_eps, 2)} 
    graph_odd = {i: [i-1] for i in range(1, num_eps, 2)} 
    graph.update(graph_odd)
    for i in junction_eps:
        graph[i[0]].append(i[1])
        
    visited = breadth_first_search(graph, root = 0)
    return visited, graph



def breadth_first_search(graph, root): 
    
    visited, queue = set(), collections.deque([root])
    while queue: 
        vertex = queue.popleft()
        for neighbour in graph[vertex]: 
            if neighbour not in visited: 
                visited.add(neighbour) 
                queue.append(neighbour) 
    return visited



def rearrange_graph(graph, graph_v, graph_e):
    
    for i in graph:
        ngb =  set(graph[i])
        if len(ngb)>1:
            l_ngb = -1
            while len(ngb) > l_ngb:
                l_ngb = len(ngb)
                nngb = ngb - set(graph_e[graph_v[i]])
                for n in nngb:
                    ngb |= set(graph[n]) - set(graph_e[graph_v[n]]) - set([i])
                    
            graph[i] = ngb
    return graph

            
        
def order_branch(branch, junction, order):
    
    e1 = np.sum(branch[0]-junction)**2
    e2 = np.sum(branch[-1]-junction)**2
    
    if order=='descend':
        if e1<e2:
            branch = np.flip(branch, axis=0)
            
    elif order=='ascend':
        if e1>e2:
            branch = np.flip(branch, axis=0)
    return branch    


def unique(mylist):
    
    unique_list = []
    for x in mylist:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list


def tangent_vector_sum(branch):
    
    d=np.gradient(branch,axis=0)
    ds=np.sum((d**2),axis=1)**0.5
    ds=np.repeat(np.expand_dims(ds,axis=1),3,axis=1)
    ds[ds==0] = 1e-5
    nTangVec=d/ds
    vec_sum = np.sum(nTangVec,axis=0)
    vec_sum = vec_sum / np.linalg.norm(vec_sum)
    return vec_sum
    
    
    
    
    
    
    