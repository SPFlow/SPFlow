from spflow.meta.data.scope import Scope
from spflow.meta.data.feature_types import FeatureTypes
from spflow.meta.data.feature_context import FeatureContext
import unittest


class TestFeatureContext(unittest.TestCase):
    def test_feature_context_initialization(self):

        feature_ctx = FeatureContext(Scope([0, 1, 2, 3]))

        # make sure scope is correctly stored
        self.assertEqual(Scope([0, 1, 2, 3]), feature_ctx.scope)
        # make sure default domains are correctly set
        self.assertTrue(set(feature_ctx.domain_map.keys()) == set([0, 1, 2, 3]))
        self.assertTrue(
            all(
                [
                    domain == FeatureTypes.Unknown
                    for domain in feature_ctx.domain_map.values()
                ]
            )
        )

        # ----- initialize with domain dict -----
        feature_ctx = FeatureContext(
            Scope([0, 1, 2, 3]),
            {0: FeatureTypes.Continuous, 2: FeatureTypes.Discrete},
        )

        # make sure default domains are correctly set
        self.assertTrue(feature_ctx.domain_map[0] == FeatureTypes.Continuous)
        self.assertTrue(feature_ctx.domain_map[1] == FeatureTypes.Unknown)
        self.assertTrue(feature_ctx.domain_map[2] == FeatureTypes.Discrete)
        self.assertTrue(feature_ctx.domain_map[3] == FeatureTypes.Unknown)

        # domains for feature ids that are out of scope
        self.assertRaises(
            ValueError,
            FeatureContext,
            Scope([0, 1]),
            {0: FeatureTypes.Continuous, 2: FeatureTypes.Discrete},
        )

    def test_set_domains(self):

        feature_ctx = FeatureContext(Scope([0, 1, 2, 3]))

        # ----- add domains as iterable -----
        feature_ctx.set_domains(
            {0: FeatureTypes.Continuous, 2: FeatureTypes.Discrete}
        )

        # make sure default domains are correctly set
        self.assertTrue(feature_ctx.domain_map[0] == FeatureTypes.Continuous)
        self.assertTrue(feature_ctx.domain_map[1] == FeatureTypes.Unknown)
        self.assertTrue(feature_ctx.domain_map[2] == FeatureTypes.Discrete)
        self.assertTrue(feature_ctx.domain_map[3] == FeatureTypes.Unknown)

        # domains for feature ids that are out of scope
        self.assertRaises(
            ValueError, feature_ctx.set_domains, {4: FeatureTypes.Discrete}
        )

        # set domain for feature id with existing domain
        self.assertRaises(
            ValueError,
            feature_ctx.set_domains,
            {0: FeatureTypes.Discrete},
            overwrite=False,
        )

        feature_ctx.set_domains({0: FeatureTypes.Discrete}, overwrite=True)
        self.assertTrue(feature_ctx.domain_map[0] == FeatureTypes.Discrete)

    def test_get_domains(self):

        feature_ctx = FeatureContext(
            Scope([0, 1, 2, 3]),
            {0: FeatureTypes.Continuous, 2: FeatureTypes.Discrete},
        )

        self.assertEqual(
            feature_ctx.get_domains([2, 0]),
            [FeatureTypes.Discrete, FeatureTypes.Continuous],
        )  # also tests if order is correct
        self.assertRaises(KeyError, feature_ctx.get_domains, [0, 4])


if __name__ == "__main__":
    unittest.main()
