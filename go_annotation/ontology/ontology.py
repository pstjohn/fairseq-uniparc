import itertools
import gzip
import re
import os 

import numpy as np
import pandas as pd
import networkx as nx

dir_path = os.path.dirname(os.path.realpath(__file__))

def parse_group(group):
    out =  {}
    out['type'] = group[0]
    out['is_a'] = []
    out['relationship'] = []
    
    for line in group[1:]:
        key, val = line.split(': ', 1)
        
        # Strip out GO names
        if '!' in val:
            val = re.sub('\ !\ .*$', '', val)
        
        if key == 'relationship':
            val = val.split(' ')
            
        # Convert to lists of GO names
        if key not in out:
            out[key] = val
        else:
            try:
                out[key] += [val]
            except TypeError:
                out[key] = [out[key], val]

    return out

add_rels = False


class Ontology(object):
    def __init__(self, obo_file=None, with_relationships=False, restrict_terms=True):
        """ Class to parse an .obo.gz file containing a gene ontology description,
        and build a networkx graph. Allows for propogating scores and annotations
        to descendent nodes.
        
        obo_file: a gzipped obo file that corresponds to the ontology used in the
        training data. Here, QuickGO annotations used GO releases/2020-06-24
        
        with_relationships (bool): whether to include GO relationships as explicit
        links in the dependency graph
        
        """
        
        if obo_file is None:
            obo_file = os.path.join(dir_path, 'go-basic.obo.gz')
            
        self.G = self.create_graph(obo_file, with_relationships)
        
        if restrict_terms is False:
            to_include = set(self.G.nodes)
            
        else:
            term_file = os.path.join(dir_path, 'terms.csv.gz')
            to_include = set(pd.read_csv(term_file, header=None)[0])
            

        self.term_index = {}
        for i, (node, data) in enumerate(filter(
            lambda x: x[0] in to_include, self.G.nodes.items())):
        
            data['index'] = i
            self.term_index[i] = node
        
        self.total_nodes = i + 1
    
    def create_graph(self, obo_file, with_relationships):

        G = nx.DiGraph()
        
        with gzip.open(obo_file, mode='rt') as f:


            groups = ([l.strip() for l in g] for k, g in
                      itertools.groupby(f, lambda line: line == '\n'))

            for group in groups:
                data = parse_group(group)

                if ('is_obsolete' in data) or (data['type'] != '[Term]'):
                    continue

                G.add_node(data['id'], name=data.get('name'), namespace=data.get('namespace'))

                for target in data['is_a']:
                    G.add_edge(target, data['id'], type='is_a')

                if with_relationships:
                    for type_, target in data['relationship']:
                        G.add_edge(target, data['id'], type=type_)
        
        nx.set_node_attributes(G, None, 'index')

        return G
    
    
    def terms_to_indices(self, terms):
        """ Return a sorted list of indices for the given terms, omitting
        those less common than the threshold """
        return sorted([self.G.nodes[term]['index'] for term in terms if 
                       self.G.nodes[term]['index'] is not None])

    
    def get_ancestors(self, terms):
        """ Includes the query terms themselves """
        if type(terms) is str:
            terms = (terms,)
            
        return set.union(set(terms), *(nx.ancestors(self.G, term) for term in terms))
    

    def get_descendants(self, terms):
        """ Includes the query term """
        if type(terms) is str:
            terms = (terms,)
            
        return set.union(set(terms), *(nx.descendants(self.G, term) for term in terms))    
    
    
    def termlist_to_array(self, terms, dtype=bool):
        """ Propogate labels to ancestor nodes """
        arr = np.zeros(self.total_nodes, dtype=dtype)
        arr[np.asarray(self.terms_to_indices(terms))] = 1
        return arr
    

    def array_to_termlist(self, array):
        """ Return term ids where array evaluates to True. Uses np.where """
        return [self.term_index[i] for i in np.where(array)[1]]
    
    
    def iter_ancestor_array(self):
        """ Constructs the necessary arrays for the tensorflow segment operation.
        Returns a generator of (node_id, ancestor_id) pairs. Use via
        `segments, ids = zip(*self.iter_ancestor_array())` """

        for node, node_index in self.G.nodes(data='index'):
            if node_index is not None:
                for ancestor_index in self.terms_to_indices(
                    self.get_ancestors(node)):
                    
                    yield node_index, ancestor_index

    
    def get_head_node_indices(self):
        return self.terms_to_indices([node for node, degree 
                                      in self.G.in_degree if degree == 0])
    
    
# notes, use nx.shortest_path_length(G, root) to find depth? score accuracy by tree depth?

# BP = ont.G.subgraph(ont.get_descendants('GO:0008150'))
# MF = ont.G.subgraph(ont.get_descendants('GO:0003674'))
# CC = ont.G.subgraph(ont.get_descendants('GO:0005575'))