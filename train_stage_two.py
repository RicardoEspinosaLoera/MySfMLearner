from __future__ import absolute_import, division, print_function

#from trainer_stage_two import Trainer
from trainer_stage_two_new import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()
