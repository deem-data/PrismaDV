"""The num_batches hyperparameter: n batches from the split's train+eval labels only, shared by train and eval."""
import argparse
from types import SimpleNamespace

import dspy
import pytest

from workflow_sifta._0_splits import DATASETS, SPLIT_SEEDS, load_split
from workflow_sifta._2_generation import program_of_run
from workflow_sifta.metrics.optimization_metrics import example_labels
from workflow_sifta.optimization_fns.optimization_labels import choose_batches
from workflow_sifta.run import job_name, optimize_command

LABELS = [f"{k}-{n}" for k in range(1, 8) for n in range(1, 6)]   # 35 labels, as imdb


@pytest.mark.parametrize("n", [1, 6, 12, 35])
def test_n_distinct_batches_of_the_given_labels(n):
    batches = choose_batches(LABELS, n)
    assert len(batches) == len(set(batches)) == n
    assert set(batches) <= set(LABELS)


def test_all_labels_is_the_main_arm():
    assert sorted(choose_batches(LABELS, len(LABELS))) == sorted(LABELS)


def test_smaller_n_is_a_subset_of_larger_n():
    assert choose_batches(LABELS, 6) == choose_batches(LABELS, 12)[:6] == choose_batches(LABELS, 35)[:6]


def test_draw_does_not_depend_on_label_order_or_global_random():
    import random
    random.seed(0)
    first = choose_batches(LABELS, 6)
    random.seed(1)
    assert choose_batches(list(reversed(LABELS)), 6) == first


@pytest.mark.parametrize("n", [0, 36, 40])
def test_invalid_n_raises(n):
    with pytest.raises(ValueError):
        choose_batches(LABELS, n)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("seed", SPLIT_SEEDS)
def test_batches_never_come_from_test_labels(dataset, seed):
    split = load_split(dataset, seed)
    for n in (6, 12):
        batches = choose_batches(split["train_eval_labels"], n)
        assert set(batches) <= set(split["train_eval_labels"])
        assert not set(batches) & set(split["test_labels"])


def test_an_example_is_scored_on_its_own_labels():
    workflow = SimpleNamespace(train_processed_data_label_list=["1-1", "1-2"])
    example = dspy.Example(target_column="c", new_data_safety={"3-1": True, "2-4": False})
    assert example_labels(example, workflow) == ["3-1", "2-4"]
    assert example_labels(dspy.Example(target_column="c"), workflow) == ["1-1", "1-2"]


def test_program_names():
    main = {"batch_sampler_strategy": "worst_first", "feedback_level": "full"}
    assert program_of_run("sifta", main)[0] == "sifta"                  # a run from before the setting
    assert program_of_run("sifta", {**main, "num_batches": None})[0] == "sifta"
    assert program_of_run("sifta", {**main, "num_batches": 6})[0] == "sifta_batches6"
    assert program_of_run("sifta", {**main, "num_batches": 12})[0] == "sifta_batches12"
    assert program_of_run("sifta", {**main, "feedback_constraints": 3, "replicate_seed": 1})[0] == "sifta_fc3"
    with pytest.raises(ValueError):
        program_of_run("sifta", {**main, "num_batches": 8})


def test_run_passes_the_option_only_when_set():
    base = dict(stage="optimize", arm="sifta", dataset="imdb", split=1, passes=15, llm="gemini-3.8-flash",
                strategy="worst_first", feedback_level="full", feedback_constraints=5, metric="f1",
                silent_as_safe=0.0, replicate_seed=0)
    plain = argparse.Namespace(**base, num_batches=None)
    assert "--num_batches" not in optimize_command(plain)
    assert job_name(plain) == "sifta-opt-sifta-imdb-s1"
    a = argparse.Namespace(**base, num_batches=12)
    cmd = optimize_command(a)
    assert cmd[cmd.index("--num_batches") + 1] == "12"
    assert job_name(a) == "sifta-opt-sifta_batches12-imdb-s1"
