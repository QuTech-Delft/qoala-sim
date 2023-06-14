import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from qoala.runtime.task import TaskGraph


class TaskGraphWriter:
    def __init__(self, task_graph: TaskGraph) -> None:
        self._tg = task_graph

        G = nx.DiGraph()

        for task_id, tinfo in task_graph.get_tasks().items():
            typ = tinfo.task.__class__.__name__
            G.add_node(task_id, typ=typ)

        for task_id, tinfo in task_graph.get_tasks().items():
            for pred in tinfo.predecessors:
                G.add_edge(pred, task_id)
            # for succ in tinfo.successors:
            #     G.add_edge(task_id, succ)

        self._nx_graph = G

    def draw(self, path: str) -> None:
        matplotlib.use("Agg")
        f = plt.figure()
        pos = nx.planar_layout(self._nx_graph)
        labels = nx.get_node_attributes(self._nx_graph, "typ")
        nx.draw(
            self._nx_graph,
            pos,
            ax=f.add_subplot(111),
            with_labels=True,
            labels=labels,
            font_size=8,
        )
        f.savefig(path)
