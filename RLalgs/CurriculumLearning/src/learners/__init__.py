from learners.q_learner import QLearner
from learners.curriculum_learner import CurriculumLearner

REGISTRY = {}

from learners.coma_learner import COMALearner as COMA
from learners.qtran_learner import QLearner as QTranLearner
from learners.offpg_learner import OffPGCritic
from learners.ppo_learner import PPOLearner
from learners.max_q_learner import MAXQLearner
from learners.dmaq_qatten_learner import DMAQ_qattenLearner
from learners.fmac_learner import FMACLearner
from learners.lica_learner import LICALearner
from learners.nq_learner import NQLearner
from learners.policy_gradient_v2 import PGLearner_v2 as PolicyGradientLearner

REGISTRY["q_learner"] = QLearner
REGISTRY["coma"] = COMA
REGISTRY["qtran"] = QTranLearner
REGISTRY["offpg"] = OffPGCritic
REGISTRY["ppo"] = PPOLearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["curriculum_learner"] = CurriculumLearner
REGISTRY["dmaq_qatten"] = DMAQ_qattenLearner
REGISTRY["fmac"] = FMACLearner
REGISTRY["lica"] = LICALearner
REGISTRY["nq"] = NQLearner
REGISTRY["policy_gradient"] = PolicyGradientLearner
