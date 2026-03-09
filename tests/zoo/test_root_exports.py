from __future__ import annotations

import importlib

from spflow.zoo.apc import AutoencodingPC
from spflow.zoo.conv import ConvPc
from spflow.zoo.einet import Einet
from spflow.zoo.naive_bayes import NaiveBayes
from spflow.zoo.rat import RatSPN
from spflow.zoo.sos import ExpSOCS, ExpSOSModel, SOCS, SOSModel


def test_root_zoo_package_re_exports_high_level_models() -> None:
    module = importlib.import_module("spflow.zoo")

    assert module.NaiveBayes is NaiveBayes
    assert module.Einet is Einet
    assert module.RatSPN is RatSPN
    assert module.ConvPc is ConvPc
    assert module.AutoencodingPC is AutoencodingPC
    assert module.SOCS is SOCS
    assert module.ExpSOCS is ExpSOCS
    assert module.SOSModel is SOSModel
    assert module.ExpSOSModel is ExpSOSModel


def test_root_zoo_package_all_lists_only_re_exported_models() -> None:
    module = importlib.import_module("spflow.zoo")

    assert module.__all__ == [
        "NaiveBayes",
        "Einet",
        "RatSPN",
        "ConvPc",
        "AutoencodingPC",
        "SOCS",
        "ExpSOCS",
        "SOSModel",
        "ExpSOSModel",
    ]
