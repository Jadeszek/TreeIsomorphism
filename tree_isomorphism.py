"""
 13. (*) Find an algorithm that test whether two given trees are isomorphic.
 Implement the algorithm using Python.
 What is the complexity of the algorithm?
 Give a short overview about the general graph isomorphism problem.
"""
import unittest
from queue import Queue

from IPython.display import display
from graphviz import Source


class RootedTree:
    """A class for simple rooted trees."""

    def __init__(self, root, structure):
        self.root = root
        self.structure = dict(structure)
        self.father_map = dict()
        self.father_map[self.root] = None
        for node, sons in structure.items():
            for son in sons:
                self.father_map[son] = node
                if self.structure.get(son) is None:
                    self.structure[son] = []

    def get_sons(self, v):
        return self.structure.get(v, [])

    def get_father(self, v):
        return self.father_map.get(v, None)

    def is_leaf(self, v):
        return len(self.get_sons(v)) == 0

    def is_last_son(self, v):
        if v == self.root:
            return True
        sons = self.get_sons(self.get_father(v))
        index = sons.index(v)
        if index == len(sons) - 1:
            return True
        return False

    def first(self, v):
        if not self.is_leaf(v):
            return self.get_sons(v)[0]
        raise IndexError("first of leaf cannot be read")

    def next(self, v):
        if self.is_last_son(v):
            raise IndexError("next of last son cannot be read")
        sons = self.get_sons(self.get_father(v))
        return sons[sons.index(v) + 1]

    def get_root(self):
        return self.root

    def get_nodes(self):
        return self.structure.keys()

    def get_order(self):
        return len(self.structure)

    def get_size(self):
        return self.get_order() - 1

    def get_dot_string(self, graph_name="Tree", labeled=False):

        def get_labels(t):
            labels = ""
            for node in t.get_nodes():
                labels += "\t" + node + "\t[xlabel=\"" + str(t.label(node)) + "\"];\n"
            return labels

        def get_edges(node, t):
            edges = ""
            for son in t.get_sons(node):
                edges += "\t" + node + " -- " + son + ";\n"
            for son in t.get_sons(node):
                edges += get_edges(son, t)
            return edges

        return "graph " + graph_name + " { \n forcelabels=true; \n" + get_labels(self) + get_edges(self.get_root(),
                                                                                                   self) + " } "

    def render(self, labeled=False):
        dot = self.get_dot_string(labeled=labeled)
        # print(dot)
        src = Source(dot)
        display(src)

    def dfs(self, fun):
        visited = {node: False for node in self.get_nodes()}

        def _dfs(node, node_function):
            visited[node] = True
            node_function(node)
            for son in self.get_sons(node):
                if not visited[son]:
                    _dfs(son, node_function)

        _dfs(self.get_root(), fun)

    def bfs(self, fun):
        visited = {node: False for node in self.get_nodes()}
        root = self.get_root()
        q = Queue()
        fun(root)
        q.put(root)
        while not q.empty():
            node = q.get()
            for v in self.get_sons(node):
                if not visited[v]:
                    visited[v] = True
                    fun(v)
                    q.put(v)

    def label(self, node=None):
        if node is None:
            node = self.get_root()
        sons = list(self.get_sons(node))
        number = len(sons)
        labels = []
        while sons:
            # backup and remove first vertex from sons
            vertex = sons.pop(0)
            labels.append(self.label(vertex))

        label = [[number]] + sorted(labels)

        return [item for sublist in label for item in sublist]

    def print_labels(self):
        def fun(node):
            print(node, "\thas label ", self.label(node))

        self.bfs(fun)


def ordered_rooted_tree_iso(t1, t2):
    if t1.get_size() != t2.get_size():
        return False

    def get_node_number_mapping(t):
        mapping = {node: None for node in t.get_nodes()}

        def int_gen():
            i = 0
            while True:
                yield i
                i += 1

        gen = int_gen()

        def labeler(node):
            mapping[node] = next(gen)

        t.dfs(labeler)
        return mapping

    node_1_to_number = get_node_number_mapping(t1)
    # inverse note to number2 mapping
    number_to_node_2 = {v: k for k, v in get_node_number_mapping(t2).items()}
    sigma = {node1: number_to_node_2[number] for node1, number in node_1_to_number.items()}

    for node in t1.get_nodes():
        if not t1.is_leaf(node):
            if t2.is_leaf(sigma[node]) or (sigma[t1.first(node)] != t2.first(sigma[node])):
                return False
        if not t1.is_last_son(node):
            if t2.is_last_son(sigma[node]) or (sigma[t1.next(node)] != t2.next(sigma[node])):
                return False
    return True


def rooted_tree_iso(t1, t2):
    return t1.label() == t2.label()

class RotatedTreeTest(unittest.TestCase):
    def setUp(self):
        self.testTree = RootedTree('ROOT', {
            'ROOT': ['L', 'L'],
            'L': ['LL', 'LR'],
            'R': ['RL', 'RM', 'RR'],
        })

    def tearDown(self):
        del self.testTree

    def test_get_root(self):
        self.assertEqual(self.testTree.get_root(), 'ROOT')

    def test_get_order(self):
        self.assertEqual(self.testTree.get_order(), 8)

    def test_get_size(self):
        self.assertEqual(self.testTree.get_size(), 7)


class IsomorphismAlgorithmTest(unittest.TestCase):
    def testOrderedRootedTreeIsomorphism(self):
        test = RootedTree('R', {
            'R': ['A', 'B'],
            'A': ['AX', 'AY'],
            'B': ['BX', 'BY', 'BZ'],
            'BZ': ['1', '2', '3', '4', '5', '6']
        })

        iso = RootedTree('r', {
            'r': ['L', 'R'],
            'L': ['LL', 'LR'],
            'R': ['RL', 'RM', 'RR'],
            'RR': ['a', 'b', 'c', 'd', 'e', 's']
        })

        noniso_size = RootedTree('r', {
            'r': ['L', 'C', 'R'],
            'L': ['LL', 'LR'],
            'R': ['RL', 'RM', 'RR'],
            'RR': ['a', 'b', 'c', 'd', 'e', 's']
        })

        noniso_order = RootedTree('r', {
            'r': ['L', 'R'],
            'L': ['LL', 'LR'],
            'R': ['RL', 'RM', 'RR'],
            'RM': ['a', 'b', 'c', 'd', 'e', 's']
        })

        self.assertTrue(ordered_rooted_tree_iso(test, iso))
        self.assertFalse(ordered_rooted_tree_iso(test, noniso_size))
        self.assertFalse(ordered_rooted_tree_iso(test, noniso_order))

    def testRootedTreeIsomorphism(self):
        test = RootedTree('R', {
            'R': ['A', 'B'],
            'A': ['AX', 'AY'],
            'B': ['BX', 'BY', 'BZ'],
            'BZ': ['1', '2', '3', '4', '5', '6']
        })

        iso = RootedTree('r', {
            'r': ['L', 'R'],
            'L': ['LL', 'LR'],
            'R': ['RL', 'RM', 'RR'],
            'RL': ['a', 'b', 'c', 'd', 'e', 's']
        })

        noniso = RootedTree('r', {
            'r': ['L', 'C', 'R'],
            'L': ['LL', 'LR'],
            'R': ['RL', 'RM', 'RR'],
            'RR': ['a', 'b', 'c', 'd', 'e', 's']
        })

        self.assertEqual(test.label(), iso.label())
        self.assertFalse(rooted_tree_iso(test, noniso))



if __name__ == "__main__":
    unittest.main(exit=False)

    # y = [1,[2],3]
    # x = [[item] if isinstance(item, int) else item for item in y]
    # print(x)
    # print( isinstance([1], int) )
    #
    # exit()

    # test = RootedTree('r', {
    #     'r': ['L', 'R'],
    #     'L': ['l1', 'l2', 'l3'],
    #     'R': ['r1', 'r2', 'r3'],
    #     'r2': ['a', 'b'],
    #     'r3': ['c', 'd']
    # })

    test = RootedTree('R', {
        'R': ['A', 'c', 'B'],
        'A': ['AX', 'AY'],
        'B': ['BX', 'BY', 'BZ'],
        'BZ': ['1', '2', '3', '4', '5', '6']
    })

    iso = RootedTree('r', {
        'r': ['L', 'C', 'R'],
        'L': ['LL', 'LR'],
        'R': ['RL', 'RM', 'RR'],
        'RL': ['a', 'b', 'c', 'd', 'e', 's']
    })

    test.render()
    iso.render()
    print('test', test.label())
    print('iso', iso.label())

    test.render(labeled=True)
    iso.render(labeled=True)
