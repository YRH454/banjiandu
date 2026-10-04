"""Public, explicitly synthetic aggregate fixtures; never real experiment data."""
from __future__ import annotations

import copy


def synthetic_holdout():
    return dict(synthetic_unit_test_only=True, dataset_manifest_sha256="1"*64,
                validation_sha256="2"*64, test_sha256="3"*64, test_index_sha256="4"*64,
                test_anchors=16, registration_sha256="5"*64)


def synthetic_teacher_accounting(l_anchors=16):
    return dict(teacher_upstream_seconds=2., teacher_successful_updates=(l_anchors+15)//16,
                teacher_validation_calls=1, teacher_l_forward_pairs=6*l_anchors, teacher_backward_pairs=2*l_anchors,
                teacher_validation_forward_pairs=800, teacher_descriptor_pairs=4*l_anchors+800)


def synthetic_teacher_resources():
    return dict(trainable_parameters=1, descriptor_parameters=1,
                peak_cuda_allocated_bytes=None, peak_cuda_reserved_bytes=None)


def synthetic_metrics(step, accuracy=.75, auc=.80, model="ema", split="validation", anchors=400, threshold=.5):
    # The confusion values only exercise aggregate validation. They do not
    # purport to be predictions from an ALBEF model or a real dataset.
    correct = round(.75*anchors)
    binary = dict(n=2*anchors, threshold=threshold, accuracy=correct/anchors,
                  macro_f1=correct/anchors, auroc=auc,
                  confusion_matrix=dict(tp=correct, tn=correct, fp=anchors-correct, fn=anchors-correct))
    return {**binary, "threshold_0_5": {**copy.deepcopy(binary), "threshold": .5},
            "paired_accuracy": accuracy, "evaluation_model": model, "evaluation_split": split,
            "step": step, f"{split}_anchors": anchors, f"{split}_pairs": 2*anchors}
