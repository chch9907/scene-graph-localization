import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import pickle
from networkx.algorithms.components import connected_components

from scipy.spatial.distance import cdist
from PIL import ImageDraw, Image
from scipy.spatial import Voronoi, voronoi_plot_2d
import scipy.ndimage as ndimage
import os
from utils.utils import get_center_dist, nearest_neighbor

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __call__(self):
        return np.array([self.x, self.y])


class DynamicSceneGraph(object):
    def __init__(self, obs, cfg, num_text, map_landmarks=None, is_global_map=False):
        self.is_global_map = is_global_map
        self.prune = cfg.prune
        self.edge_thred = cfg.edge_thred
        self.neighbor_thred = cfg.neighbor_thred
        self.search_type = cfg.search_type
        self.cfg = cfg
        self.map_landmarks = map_landmarks
        self.scores_list = []
        
        if self.is_global_map:
            self.image = obs
            self.depth = None
        else:
            self.image = obs['rgb'] 
            self.depth = obs['depth']   
        self.num_text = num_text
        self.node_landmarks = []
        self.node_probs = []
        self.node_bboxs = []
        self.node_centers = []
        self.node_viewpoints = []
        self.node_neighbors = [[] for _ in range(num_text)]
        self.edges = np.zeros((num_text, num_text))
        self.graph = nx.Graph()
        self.tsp = nx.approximation.traveling_salesman_problem
        self.prune = cfg.prune
        self.edge_thred = cfg.edge_thred
        self.neighbor_thred = cfg.neighbor_thred
        self.search_type = cfg.search_type
        # self.map_OCR_dict = map_OCR_dict
        self.scores_list = []
        self.is_global_map = is_global_map
        self.graph_path = cfg.map_path.replace('png', 'pkl')
        if self.is_global_map and os.path.exists(self.graph_path):
            # assert os.path.exists(graph_path)
            self.load_graph(self.graph_path)
        else:
            self.generate_graph(OCR_dict, map_landmarks)
            # self.save_graph(self.graph_path)
        # print("node_neighbors:", self.node_neighbors)
    
    def offline_process(self,):
        print('offline process topological graph.')
        # self.generate_graph(self.OCR_dict, map_landmarks)
        self.save_graph(self.graph_path)
    
    def __len__(self) -> int:
        return len(self.node_landmarks)
    
    def generate_graph(self, ocr_list: List[Dict], map_graph: List[Dict]):
        print('generate graph......')


    def update(self, target_room_idx, target_room_score, target_room_pose, target_room_visited):
        for i, (idx, score, pose) in enumerate(zip(target_room_idx, target_room_score, target_room_pose)):
            if idx not in self.node_idx:
                self.node_idxs.append(idx)
                self.node_scores.append(score)
                self.node_centers.append(pose)
                self.node_visited.append(0)
                self.graph.add_node(idx)
                
            else:
                # i = self.node_idx.index(idx)
                self.node_scores[i] = score
                if idx in target_room_visited:
                    self.node_visited[i] = 1
        
        ## edges
        for i in range(len(self)):
            for j in range(len(self)):
                dist = get_center_dist(self.node_centers[i], self.node_centers[j])
                self.edges[i, j] = dist
                if dist <= self.edge_thred and i != j:
                    self.graph.add_edge(i, j, weight=self.node_scores[j])
                    self.graph.add_edge(j, i, weight=self.node_scores[i])
                    # self.node_neighbors[i].append(j)
        
        # if self.prune:
        #     self.graph = nx.transitive_reduction(self.graph)  #! not implemented for undirected or cyclic graph
    
    
    
    def search_route(self, start_idx, goal_idx) -> List: 
        # dijkstra_path, astar_path
        if self.search_type == 'dijkstra':
            shortest_route = nx.dijkstra_path(self.graph, start_idx, goal_idx)  # , weight='weight'
        elif self.search_type == 'astar':
            shortest_route = nx.astar_path(self.graph, start_idx, goal_idx)
        else:
            raise ValueError("self.search_type should be within ['dijkstra', 'astar']")
        
        
        # all_shortest_gen = nx.all_shortest_paths(self.graph, start_idx, goal_idx)  # dijkstra
        # all_shortest_routes = [route for route in all_shortest_gen]
        # all_simple_gen = nx.all_simple_paths(self.graph, start_idx, goal_idx, cutoff=None)
        # all_simple_routes = [route for route in all_simple_gen]
        return shortest_route


    def get_weighted_tsp_route(self, ):
        ## find a path with weights on nodes: 
        # 1) https://stackoverflow.com/questions/55403130/create-a-networkx-weighted-graph-and-find-the-path-between-2-nodes-with-the-smal
        # 2) https://stackoverflow.com/questions/72668452/finding-minimal-path-with-networkx-where-the-weights-are-the-nodes
        
        tsp_path = self.tsp(self.graph, cycle=False)
        return tsp_path
    

    def graph_serialization(self,):
        return nx.readwrite.json_graph.node_link_data(self.graph)
    
    
    def plot_graph_nx(self, ):
        # nx.draw_networkx_edges(self.graph, pos=nx.spring_layout(self.graph))
        nx.draw_networkx(self.graph, pos=nx.spring_layout(self.graph))
        plt.show()

    def save_graph(self, save_path):
        data = {
            "node_landmarks": self.node_landmarks,
            "node_probs": self.node_probs,
            "node_centers": self.node_centers,
            "node_bboxs": self.node_bboxs,
            "node_neighbors": self.node_neighbors,
            "edges": self.edges,
            "json_graph": self.graph_serialization()
        }
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print('save graph:', save_path)
    
    
    
    def load_graph(self, load_path):
        with open(load_path, "rb") as f:
            data = pickle.load(f)
        self.node_landmarks = data['node_landmarks']
        self.node_probs = data['node_probs']
        self.node_centers = data['node_centers']
        self.node_neighbors = data['node_neighbors']
        self.node_bboxs = data['node_bboxs']
        self.edges = data['edges']
        self.num_text = len(self.node_landmarks)
        self.graph = nx.readwrite.json_graph.node_link_graph(data['json_graph'])
        print('load graph:', load_path)
        
    def clear(self):
        self.graph.clear()
            
        