"""V4 private contracts and disjoint, shared phase-local augmentation clocks."""
from fair_benchmark.data import BoundPairData
from .contracts import validate_test_selection
from .spec import load_protocol, validate_config


class PlateauPairData(BoundPairData):
    BINDING_FORMAT = "fair_private_inputs_v4"
    HOLDOUT_FORMAT = "fair_holdout_registration_v4"
    TEACHER_FORMAT = "fair_pairusa_targets_v4"
    load_protocol = staticmethod(load_protocol)
    validate_config = staticmethod(validate_config)
    validate_test_selection = staticmethod(validate_test_selection)

    def _view_step(self, step):
        return step if self.cfg["stage"] == "bce" else step+self.p["phases"]["bce"]["max_successful_steps"]

    def batch(self, pairs, step, view, physical):
        return super().batch(pairs, self._view_step(step), view, physical)

    def teacher_vectors(self, selected_l, step):
        return super().teacher_vectors(selected_l, self._view_step(step))
