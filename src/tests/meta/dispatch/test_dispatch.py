from spflow.meta.dispatch.dispatch import dispatch
from spflow.meta.structure.module import MetaModule
from spflow.meta.contexts.dispatch_context import (
    DispatchContext,
    init_default_dispatch_context,
    default_dispatch_context,
)
from typing import Optional
import unittest


# dispatching requires an instance of abstract class 'MetaModule' as first argument (e.g., for caching etc.)
class Module(MetaModule):
    pass


class TestNode(unittest.TestCase):
    def test_dispatch(self):

        # dispatch two different signatures of the same function
        @dispatch
        def func(
            module: Module,
            i: int,
            dispatch_ctx: Optional[DispatchContext] = None,
        ) -> int:
            return 0

        @dispatch
        def func(
            module: Module,
            i: float,
            dispatch_ctx: Optional[DispatchContext] = None,
        ) -> int:
            return 1

        # make sure correct functions are called
        self.assertTrue(func(Module(), 1) == 0)
        self.assertTrue(func(Module(), 1.0) == 1)

    def test_dispatch_memoization(self):

        # dispatch function with memoization
        @dispatch(memoize=True)
        def func(
            module: Module,
            i: int,
            dispatch_ctx: Optional[DispatchContext] = None,
        ) -> int:
            return 0

        # create dummy module
        module = Module()
        # create empty dispatch context
        dispatch_ctx = default_dispatch_context()

        res = func(module, 1, dispatch_ctx=dispatch_ctx)

        # make sure return value is correctly cached
        self.assertTrue("func" in dispatch_ctx.cache)
        self.assertTrue(module in dispatch_ctx.cache["func"])
        self.assertTrue(dispatch_ctx.cache["func"][module] == res)
        self.assertTrue(res == 0)

        # manipulate cached value
        dispatch_ctx.cache["func"][module] = 1

        res = func(module, 1, dispatch_ctx=dispatch_ctx)

        # make sure that manipulated value is returned from cache instead of calling function again
        self.assertTrue("func" in dispatch_ctx.cache)
        self.assertTrue(module in dispatch_ctx.cache["func"])
        self.assertTrue(dispatch_ctx.cache["func"][module] == res)
        self.assertTrue(res == 1)

    def test_dispatch_substitutable(self):

        # dispatch function and allow alternative functions to be passed in dispatch context
        @dispatch(substitutable=True)
        def func(
            module: Module,
            i: int,
            dispatch_ctx: Optional[DispatchContext] = None,
        ) -> int:
            return 0

        # alternate function with same signature, but different return value
        def alternate_func(
            module: Module,
            i: int,
            dispatch_ctx: Optional[DispatchContext] = None,
        ) -> int:
            return 1

        self.assertTrue(func(Module(), 1) == 0)

        # provide alternate function for module in a dispatch context
        dispatch_ctx = default_dispatch_context()
        dispatch_ctx.funcs[Module] = alternate_func

        self.assertTrue(func(Module(), 1, dispatch_ctx=dispatch_ctx) == 1)

    def test_dispatch_log_likelihood_without_memoization(self):

        with self.assertWarns(Warning):
            # dispatch function called 'log_likelihood' WITHOUT memoization
            @dispatch(memoize=False)
            def log_likelihood(
                module: Module, dispatch_ctx: Optional[DispatchContext] = None
            ) -> int:
                return 0

    def test_dispatch_em_without_memoization(self):

        with self.assertWarns(Warning):
            # dispatch function called 'em' WITHOUT memoization
            @dispatch(memoize=False)
            def em(
                module: Module, dispatch_ctx: Optional[DispatchContext] = None
            ) -> int:
                return 0

    def test_dispatch_maximum_likelihood_estimation_without_memoization(self):

        with self.assertWarns(Warning):
            # dispatch function called 'maximum_likelihood_estimation' WITHOUT memoization
            @dispatch(memoize=False)
            def maximum_likelihood_estimation(
                module: Module, dispatch_ctx: Optional[DispatchContext] = None
            ) -> int:
                return 0


if __name__ == "__main__":
    unittest.main()
