from pathlib import Path
from typing import List
import time
import utils
from utils import l
import config
import args
import csv
from decimal import Decimal as D
import os
from itertools import combinations
from collections import defaultdict, namedtuple
class HashTreeNode:  
    def __init__(self):
        self.children = {}
        self.isLeaf = True
        self.bucket = {}
class HashTree:
    def __init__(self, max_leaf_size, max_children):
        self.root = HashTreeNode()
        self.max_leaf_size = max_leaf_size
        self.max_children = max_children
        self.frequent_itemsets = {}
    def hash_f(self, value):
        return int(value) % int(self.max_children)
     
    def insert_recur(self, itemset, Node, index, count):
        if index == len(itemset):
            if itemset in Node.bucket: 
                Node.bucket[itemset] += count
            else:
                Node.bucket[itemset] = count 
            return
        
        if Node.isLeaf:
            if itemset in Node.bucket:
                Node.bucket[itemset] += count
            else:
                Node.bucket[itemset] = count
                
            if len(Node.bucket) == self.max_leaf_size:
                for prev_itemset, prev_count in Node.bucket.items():
                    key = self.hash_f(prev_itemset[index])
                    if key not in Node.children:
                        Node.children[key] = HashTreeNode()
                    
                    self.insert_recur(prev_itemset, Node.children[key], index + 1, prev_count)
                
                del Node.bucket
                Node.isLeaf = False
        else:
            key = self.hash_f(itemset[index])
            if key not in Node.children:
                Node.children[key] = HashTreeNode()
            self.insert_recur(itemset, Node.children[key], index + 1, count)
     
    def insert(self, itemset):
        self.insert_recur(itemset, self.root, 0, 0)
      
    def increment_freq(self, itemset):
        track_node = self.root 
        index = 0
        while True:
            if track_node.isLeaf:
                if itemset in list(track_node.bucket):
                    track_node.bucket[itemset] += 1
                break
            key = self.hash_f(itemset[index])
            if key in track_node.children:
                track_node = track_node.children[key]
            else:
                break
            index += 1
            
    def update_dict_freq_itemsets(self, HashTreeNode, minsup):     
        if HashTreeNode.isLeaf:
            for itemset, count in HashTreeNode.bucket.items():
                if count >= minsup:
                    self.frequent_itemsets[itemset] = count
        for child in HashTreeNode.children.values():
            self.update_dict_freq_itemsets(child, minsup)
        
def build_tree(list_candidates, max_leaf_size, max_children):
    tree = HashTree(max_leaf_size, max_children)
    for i, itemset in enumerate(list_candidates):
        tree.insert(itemset)
    return tree
 
def first_generate(itemList, minsup):
    counts = {}
    freq=[]
    freqSet={}
    for transaction in itemList:
        for item in transaction:
            try:
                counts[item] += 1
            except:
                counts[item] = 1
    counts_filtered=dict()
    for key, value in counts.items():
        if value >= minsup:
            freq.append(([key],value))
            counts_filtered[key] = value
            freqSet[tuple([key])]=value
    return counts_filtered,freq,freqSet

def generate_subsets(itemset, length):
    subsets = []
    if len(itemset) >= length:
        itemset = set(itemset)
        subsets = list(combinations(itemset,length))
        for i in range(len(subsets)):   
            subsets[i] = tuple(sorted(subsets[i]))
    return subsets

def generate_candidates(candidates, length,tatal,minsup):
    combination = [(x, y) for x in candidates for y in candidates]
    new_candidates = []
    drops = [] 
    for candidate in combination:
        new = []
        for element in candidate:
            if isinstance(element, tuple):
                for item in element:
                    new.append(item)
            else:
                new.append(element)
        new_candidates.append(tuple(new))
    new_candidates_final=[]    
    for i, candidate in enumerate(new_candidates):
        new_candidates[i] = tuple(set(candidate))
        if len(new_candidates[i]) == length:
            new_candidates_final.append(tuple(sorted(new_candidates[i])))
    new_candidates_final = set(new_candidates_final)
    return list(new_candidates_final)


def perform_pruning(prev_candidates, new_candidates, length_prune):
    candidates_after_prune = []
    for i, candidate in enumerate(new_candidates):
        all_subsets = list(combinations(set(candidate), length_prune))
        found = True
        for itemset in all_subsets:
            itemset = tuple(sorted(itemset))
            if itemset not in list(prev_candidates):
                found = False 
                break
        if found == True:
            candidates_after_prune.append(candidate)
    return candidates_after_prune

def generateAllFreqSet(dataList,a):
    if 0 < a.min_sup <= 1:
        minimum_support = a.min_sup * len(dataList)
    prev_candidates,freqList,freqSet = first_generate(dataList, minimum_support)
    length = 2
    while True:
        new_candidates = generate_candidates(prev_candidates, length,len(dataList),minimum_support)
        MAX_CHILDREN = []
        for candi in new_candidates:
            MAX_CHILDREN.append(list(candi)[length-1])
        if length > 2:
            new_candidates = perform_pruning(prev_candidates, new_candidates, length-1)
        if len(new_candidates) > 5:
            MAX_LEAF_SIZE = int(len(new_candidates))
            MAX_CHILDREN = max(MAX_CHILDREN)
            tree = build_tree(new_candidates, MAX_LEAF_SIZE, MAX_CHILDREN)
            for i, transaction in enumerate(dataList):
                subsets = generate_subsets(transaction, length)
                if len(subsets) > 0:
                    for subset in subsets:
                        tree.increment_freq(subset)
            tree.update_dict_freq_itemsets(tree.root, minimum_support)
            new_counts_tree = tree.frequent_itemsets
            if not bool(new_counts_tree):
                break
            else:
                for Set in new_counts_tree:
                    freqList.append((list(Set),new_counts_tree[Set]))
                    freqSet[tuple(Set)]=new_counts_tree[Set]
                length += 1
                prev_candidates = new_counts_tree.keys()
        else:
            new_counts = {}
            for transaction in dataList:
                for candidate in new_candidates:
                    candi = set(candidate)
                    inter = candi.intersection(transaction)
                    if len(inter) == length:
                        try:    
                            new_counts[candidate] += 1
                        except:
                            new_counts[candidate] = 1
            if new_counts == {}:
                break

            else:
                length += 1
                new_counts_filtered = {key: value for key, value in new_counts.items() if value >= minimum_support}
                for Set in new_counts_filtered:
                    freqList.append((list(Set),new_counts_filtered[Set]))
                    freqSet[tuple(Set)]=new_counts_filtered[Set]
                prev_candidates = new_counts_filtered.keys()
    return freqList,freqSet
class FPNode(object):
    def __init__(self, tree, item, count=1):
        self._tree = tree
        self._item = item
        self._count = count
        self._parent = None
        self._children = {}
        self._neighbor = None
    def add(self, child):
    
        if child.item not in self._children:
            self._children[child.item] = child
            child.parent = self
    def search(self, item):
        try:
            return self._children[item]
        except KeyError:
            return None

    @property
    def tree(self):
        return self._tree

    @property
    def item(self):
        return self._item

    @property
    def count(self):
        return self._count

    def increment(self):
        self._count += 1

    @property
    def root(self):
        return self._item is None and self._count is None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @property
    def neighbor(self):
        return self._neighbor

    @neighbor.setter
    def neighbor(self, value):
        self._neighbor = value

    @property
    def children(self):
        return tuple(self._children.values())

class FPTree(object):
    Route = namedtuple("Route", "head tail")
    def __init__(self):
        self._root = FPNode(self, None, None)
        self._routes = {}

    @property
    def root(self):
        return self._root

    def add(self, transaction):
        point = self._root
        for item in transaction:
            next_point = point.search(item)
            if next_point:
                next_point.increment()
            else:
                next_point = FPNode(self, item)
                point.add(next_point)
                self._update_route(next_point)
            point = next_point

    def _update_route(self, point):
        assert self is point.tree
        try:
            route = self._routes[point.item]
            route[1].neighbor = point
            self._routes[point.item] = self.Route(route[0], point)
        except KeyError:
            self._routes[point.item] = self.Route(point, point)

    def items(self):
        for item in self._routes.keys():
            yield (item, self.nodes(item))

    def nodes(self, item):
        try:
            node = self._routes[item][0]
        except KeyError:
            return
        while node:
            yield node
            node = node.neighbor

    def prefix_paths(self, item):
        def collect_path(node):
            path = []
            while node and not node.root:
                path.append(node)
                node = node.parent
            path.reverse()
            return path
        return (collect_path(node) for node in self.nodes(item))

def find_frequent_itemsets(itemList, minimum_support, include_support=False):
    def clean_itemList(itemList):
        itemList = filter(lambda v: v in items, itemList)
        itemList = sorted(itemList, key=lambda v: items[v], reverse=True)
        return itemList
    
    def find_with_suffix(tree, suffix):
        for item, nodes in tree.items():
            support = sum(n.count for n in nodes)
            if support >= minimum_support and item not in suffix:
                found_set = [item] + suffix
                yield (found_set, support) if include_support else found_set
                cond_tree = conditional_tree_from_paths(tree.prefix_paths(item))
                for s in find_with_suffix(cond_tree, found_set):
                    yield s
    items = defaultdict(lambda: 0)
    if 0 < minimum_support <= 1:
        minimum_support = minimum_support * len(itemList)
    for Items in itemList:
        for item in Items:
            items[item] += 1
    items = dict((item, support) for item, support in items.items() if support >= minimum_support)
    master = FPTree()
    for ItemList in list(map(clean_itemList, itemList)):
        master.add(ItemList)
    for itemset in find_with_suffix(master, []):
        yield itemset

def conditional_tree_from_paths(paths):
    tree = FPTree()
    condition_item = None
    items = set()
    for path in paths:
        if condition_item is None:
            condition_item = path[-1].item

        point = tree.root
        for node in path:
            next_point = point.search(node.item)
            if not next_point:
                items.add(node.item)
                count = node.count if node.item == condition_item else 0
                next_point = FPNode(tree, node.item, count)
                point.add(next_point)
                tree._update_route(next_point)
            point = next_point

    assert condition_item is not None

    for path in tree.prefix_paths(condition_item):
        count = path[-1].count
        for node in reversed(path[:-1]):
            node._count += count
    return tree

def subs(l):
    assert type(l) is list
    if len(l) == 1:
        return [l]
    res=[[i] for i in l]
    for i in range(2,len(l)+1):
        x=combinations(l,i)
        res+=[list(k) for k in x]
    return res


def assoc_rule(freq,size, min_conf=0.6):
    assert type(freq) is dict
    result = []
    for item, sup in freq.items():
        for subitem in subs(list(item)):
            sb = [x for x in item if x not in subitem]
            if sb == [] or subitem == []:
                continue
            conf = sup /freq[tuple(subitem)]
            if conf >= min_conf:
                result.append([subitem,sb,sup/size,conf,(conf*size)/freq[tuple(sb)]])
    return result
            
def readCSVData(f):
    numberCount=0
    item=[]
    for row in f:
        item.append([row[0],row[2]])
        numberCount+=1
    return item,numberCount
    
def transform(data):
    itemDict={}
    itemList=[]
    for i,j in data:
        if i not in itemDict:
            itemDict[i]=[j]
        else:
            itemDict[i].append(j)
    for i in itemDict:
        itemList.append(itemDict[i])
    return itemList

def apriori(dataList,a):
    resultRules=[]
    res_for_rul = {}
    result,res_for_rul=generateAllFreqSet(dataList,a)
    rules = assoc_rule(res_for_rul, len(dataList),a.min_conf)
    return rules

def FPgrowth(dataList,a):
    result = []
    resultRules=[]
    res_for_rul = {}
    for itemset, support in find_frequent_itemsets(dataList, a.min_sup, True):
        result.append((itemset, support))
        res_for_rul[tuple(itemset)] = support
    rules = assoc_rule(res_for_rul, len(dataList),a.min_conf)
    return rules
def main():
    a = args.parse_args()
    l.info(f"Arguments: {a}")  
    f=utils.read_file(config.IN_DIR / a.dataset)
    item,numberCount=readCSVData(f)
    filename = Path(a.dataset).stem
    itemList=transform(item)
    Start=time.time()
    apriori_out = apriori(itemList, a)
    End=time.time()
    print("apriori spends:{:.4f}s".format(End-Start))
    utils.write_file(
        data=apriori_out,
         filename=config.OUT_DIR / f"{filename}-apriori-{a.min_sup}-{a.min_conf}.csv"
    )
    Start=time.time()
    fp_growth_out = FPgrowth(itemList,a)
    End=time.time()
    print("fp_growth spends:{:.4f}s".format(End-Start))
    utils.write_file(
        data=fp_growth_out,
        filename=config.OUT_DIR / f"{filename}-fp_growth-{a.min_sup}-{a.min_conf}.csv"
    )

if __name__ == "__main__":
    main()