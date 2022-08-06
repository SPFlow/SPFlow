from typing import Optional
from spflow.base.structure.nodes.node import LeafNode
from spflow.meta.scope.scope import Scope

class DummyNode(LeafNode):
    """Dummy node class without children (to simulate leaf nodes)."""
    def __init__(self, scope: Optional[Scope]=None):

        if scope is None:
            scope = Scope([0])

        super(DummyNode, self).__init__(scope=scope)