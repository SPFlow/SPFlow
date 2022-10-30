from typing import Optional
from spflow.base.structure.module import Module, NestedModule
from spflow.meta.data.scope import Scope
from typing import List


class DummyModule(Module):
    def __init__(self, n: int, scope: Optional[Scope] = None):

        if scope is None:
            scope = Scope([0])

        self.scope = scope
        self.n = n

        super(DummyModule, self).__init__(children=[])

    @property
    def n_out(self) -> int:
        return self.n

    @property
    def scopes_out(self) -> List[Scope]:
        return [self.scope for _ in range(self.n)]


class DummyNestedModule(NestedModule):
    def __init__(self, children):

        super(DummyNestedModule, self).__init__(children=children)

        self.n_in = sum([child.n_out for child in children])
        self.create_placeholder(list(range(self.n_in)))

    @property
    def n_out(self) -> int:
        return 0

    @property
    def scopes_out(self) -> List[Scope]:
        return []
