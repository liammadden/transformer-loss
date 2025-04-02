from dataclasses import dataclass, field


@dataclass
class Run:
    m: int
    d: int
    model_obj: any = None
    model_num_params: int = 0
    training_loss_values: any = field(default_factory=list)
    test_loss_values: any = field(default_factory=list)
