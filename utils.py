from SumProductNets import *
import numpy as np
import json


def spn_to_json(spn):
    nodes_id = dict()
    rvs_id = dict()

    for idx, n in enumerate(spn.nodes):
        if type(n) is not LeafNode:
            nodes_id[n] = idx

    for idx, rv in enumerate(spn.rvs):
        rvs_id[rv] = idx

    nodes = list()

    for n in spn.nodes:
        if type(n) is SumNode:
            nodes.append({
                'id': nodes_id[n],
                'type': 'sum',
                'ch': [nodes_id[j] for j in n.ch],
                'w': list(n.w)
            })
        elif type(n) is ProductNode:
            nodes.append({
                'id': nodes_id[n],
                'type': 'prod',
                'ch': [nodes_id[j] for j in n.ch]
            })
        elif type(n) is RVNode:
            nodes.append({
                'id': nodes_id[n],
                'type': 'rv',
                'rv': rvs_id[n.rv],
                'w': list(n.w)
            })

    rvs = list()

    for rv in spn.rvs:
        rvs.append({
            'id': rvs_id[rv],
            'domain': list(rv.domain)
        })

    return json.dumps({'nodes': nodes, 'rvs': rvs})


def json_to_spn(s):
    data = json.loads(s)

    rvs = list()

    for rv in data['rvs']:
        rvs.append(RV(rv['domain']))

    nodes = dict()

    for n in reversed(data['nodes']):
        if n['type'] == 'sum':
            nodes[n['id']] = SumNode(
                ch=[nodes[idx] for idx in n['ch']],
                w=np.array(n['w'])
            )
        elif n['type'] == 'prod':
            nodes[n['id']] = ProductNode(
                ch=[nodes[idx] for idx in n['ch']],
            )
        elif n['type'] == 'rv':
            nodes[n['id']] = RVNode(
                rv=rvs[n['rv']],
                w=np.array(n['w'])
            )

    return SPN(nodes[0], rvs), rvs


def save_spn(f, spn):
    with open(f, 'w+') as file:
        file.write(spn_to_json(spn))


def load_spn(f):
    with open(f, 'r') as file:
        s = file.read()
        return json_to_spn(s)
