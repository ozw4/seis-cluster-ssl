# v2_dis0_con0

K=[6,8,10], feature distillation=0, cross-head consistency=0.
The existing v2_dis0 recipe is inherited except consistency 0.1 becomes 0,
with new model/output identities. Three source families each train 25 epochs,
then all five layouts and three label budgets are evaluated.

See the [shared experiment definition](../../../hmm_v2/v2_dis0_con0_v1/README.md)
and its SHA-bound matrix for the fixed controls and execution commands.
