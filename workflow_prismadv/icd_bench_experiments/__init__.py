from pathlib import Path
from prismadv.utils import get_project_root

from benchmarks.ICDBench.math_ops_log import MathOpsLog
from benchmarks.ICDBench.math_ops_sqrt import MathOpsSqrt
from benchmarks.ICDBench.math_ops_stddev import MathOpsStddev
from benchmarks.ICDBench.math_ops_subtractlog import MathOpsSubtractLog
from benchmarks.ICDBench.math_ops_subtractcolumnlog import MathOpsSubtractColumnLog
from benchmarks.ICDBench.math_ops_log_difference import MathOpsLogDifference
from benchmarks.ICDBench.hidden_range_hashmap import HiddenRangeHashMap
from benchmarks.ICDBench.hidden_range_list import HiddenRangeList
from benchmarks.ICDBench.hidden_range_listwith_default import HiddenRangeListWithDefault
from benchmarks.ICDBench.hidden_range_hashmapwith_default import HiddenRangeHashMapWithDefault
from benchmarks.ICDBench.access_dependency_forloop import AccessDependencyForLoop
from benchmarks.ICDBench.access_dependency_filterloop import AccessDependencyFilterLoop
from benchmarks.ICDBench.access_dependency_filterlooppolars import AccessDependencyFilterLoopPolars
from benchmarks.ICDBench.access_dependency_twoloops import AccessDependencyTwoLoops
from benchmarks.ICDBench.sklearn_logregnan import SklearnLogregNan
from benchmarks.ICDBench.sklearn_logregnancondition import SklearnLogregNanWithCondition
from benchmarks.ICDBench.sklearn_onehotunknown import SklearnOneHotUnknown
from benchmarks.ICDBench.sklearn_labelpresent import SklearnLabelPresent
from benchmarks.ICDBench.sklearn_functiontransformerlog import SklearnFunctionTransformerLog
from benchmarks.ICDBench.explicit_assert_null_ratio import ExplicitAssertNullRatio
from benchmarks.ICDBench.explicit_assert_positivevalues import ExplicitAssertPositiveValues
from benchmarks.ICDBench.explicit_assert_numdistinct import ExplicitAssertNumDistinct
from benchmarks.ICDBench.explicit_assert_numdistinctforloop import ExplicitAssertNumDistinctForLoop
from benchmarks.ICDBench.explicit_assert_columncondition import ExplicitAssertColumnCondition
from benchmarks.ICDBench.explicit_assert_range import ExplicitAssertRange
from benchmarks.ICDBench.knowledge_in_sql_range import KnowledgeInSQLRange
from benchmarks.ICDBench.knowledge_in_sql_column_condition import KnowledgeInSQLColumnCondition
from benchmarks.ICDBench.knowledge_in_sql_subtractcolumnlog import KnowledgeInSQLSubtractColumnLog
from benchmarks.ICDBench.knowledge_in_sql_invalid import KnowledgeInSQLInvalid
from benchmarks.ICDBench.knowledge_in_sql_unique import KnowledgeInSQLUnique
from benchmarks.ICDBench.domain_cricket_complexrule import DomainCricketComplexRule
from benchmarks.ICDBench.domain_cricket_hiddendependency import DomainCricketHiddenDependency
from benchmarks.ICDBench.domain_cricket_hiddendependency2 import DomainCricketHiddenDependency2
from benchmarks.ICDBench.domain_cricket_hiddendependency3 import DomainCricketHiddenDependency3
from benchmarks.ICDBench.domain_cricket_hiddendependencywithassert import DomainCricketHiddenDependencyWithAssert
from benchmarks.ICDBench.domain_cricket_rarevalues import DomainCricketRareValues
from benchmarks.ICDBench.domain_cricket_rulesincomments import DomainCricketRulesInComments
from benchmarks.ICDBench.domain_cricket_rulesincomments2 import DomainCricketRulesInComments2
from benchmarks.ICDBench.domain_cricket_rulesincomments3 import DomainCricketRulesInComments3
from benchmarks.ICDBench.domain_cricket_validrange import DomainCricketValidRange

from benchmarks.ICDBench.github_creditcardfraud_class import GithubCreditcardFraudClass
from benchmarks.ICDBench.github_fashiontrends_category import GithubFashionTrendsCategory
from benchmarks.ICDBench.github_fashiontrends_category2 import GithubFashionTrendsCategory2
from benchmarks.ICDBench.github_iplmatchwinner_city import GithubIPLMatchWinnerCity
from benchmarks.ICDBench.github_iplmatchwinner_rules1 import GithubIPLMatchWinnerRules1
from benchmarks.ICDBench.github_iplmatchwinner_rules2 import GithubIPLMatchWinnerRules2
from benchmarks.ICDBench.github_iplmatchwinner_team1 import GithubIPLMatchWinnerTeam1
from benchmarks.ICDBench.github_iplmatchwinner_team2 import GithubIPLMatchWinnerTeam2
from benchmarks.ICDBench.github_musicrecommender_key import GithubMusicRecommenderKey
from benchmarks.ICDBench.github_musicrecommender_key2 import GithubMusicRecommenderKey2
from benchmarks.ICDBench.github_payments_handvalidation_batchbooking import GithubPaymentsHandValidationBatchBooking
from benchmarks.ICDBench.github_payments_handvalidation_batchbooking2 import GithubPaymentsHandValidationBatchBooking2
from benchmarks.ICDBench.github_payments_handvalidation_creditorname import GithubPaymentsHandValidationCreditorName
from benchmarks.ICDBench.github_payments_handvalidation_creditorname2 import GithubPaymentsHandValidationCreditorName2
from benchmarks.ICDBench.github_payments_handvalidation_ctrlsum import GithubPaymentsHandValidationCtrlSum
from benchmarks.ICDBench.github_worldofwarcraft_behavior import GithubWorldOfWarcraftBehavior
from benchmarks.ICDBench.github_worldofwarcraft_behavior2 import GithubWorldOfWarcraftBehavior2
from benchmarks.ICDBench.github_worldofwarcraft_lvls import GithubWorldOfWarcraftLvls
from benchmarks.ICDBench.github_worldofwarcraft_timehours import GithubWorldOfWarcraftTimehours
from benchmarks.ICDBench.github_worldofwarcraft_zone import GithubWorldOfWarcraftZone
from benchmarks.ICDBench.github_wowauctions_timeleft import GithubWorldofWarcraftAuctionsTimeLeft
from benchmarks.ICDBench.github_wowauctions_unitprice import GithubWorldofWarcraftAuctionsUnitPrice
from benchmarks.ICDBench.sklearn_labelcomplete import SklearnLabelComplete

ALL_EVALUATION_CASES = [
    MathOpsLog(), MathOpsSqrt(), MathOpsStddev(), MathOpsSubtractLog(), MathOpsSubtractColumnLog(),
    MathOpsLogDifference(), HiddenRangeHashMap(), HiddenRangeList(), HiddenRangeListWithDefault(),
    HiddenRangeHashMapWithDefault(), AccessDependencyForLoop(), AccessDependencyFilterLoop(),
    AccessDependencyFilterLoopPolars(), AccessDependencyTwoLoops(), SklearnLogregNan(),
    SklearnLogregNanWithCondition(), SklearnOneHotUnknown(), SklearnLabelPresent(),
    SklearnFunctionTransformerLog(), ExplicitAssertNullRatio(), ExplicitAssertPositiveValues(),
    ExplicitAssertNumDistinct(), ExplicitAssertNumDistinctForLoop(), ExplicitAssertColumnCondition(),
    ExplicitAssertRange(), KnowledgeInSQLRange(), KnowledgeInSQLColumnCondition(),
    KnowledgeInSQLSubtractColumnLog(), KnowledgeInSQLInvalid(), KnowledgeInSQLUnique(),
    DomainCricketComplexRule(), DomainCricketHiddenDependency(), DomainCricketHiddenDependency2(),
    DomainCricketHiddenDependency3(), DomainCricketHiddenDependencyWithAssert(), DomainCricketRareValues(),
    DomainCricketRulesInComments(), DomainCricketRulesInComments2(), DomainCricketRulesInComments3(),
    DomainCricketValidRange(), GithubCreditcardFraudClass(), GithubFashionTrendsCategory(),
    GithubFashionTrendsCategory2(), GithubIPLMatchWinnerCity(), GithubIPLMatchWinnerRules1(),
    GithubIPLMatchWinnerRules2(), GithubIPLMatchWinnerTeam1(), GithubIPLMatchWinnerTeam2(),
    GithubMusicRecommenderKey(), GithubMusicRecommenderKey2(), GithubPaymentsHandValidationBatchBooking(),
    GithubPaymentsHandValidationBatchBooking2(), GithubPaymentsHandValidationCreditorName(),
    GithubPaymentsHandValidationCreditorName2(), GithubPaymentsHandValidationCtrlSum(),
    GithubWorldOfWarcraftBehavior(), GithubWorldOfWarcraftBehavior2(), GithubWorldOfWarcraftLvls(),
    GithubWorldOfWarcraftTimehours(), GithubWorldOfWarcraftZone(), GithubWorldofWarcraftAuctionsTimeLeft(),
    GithubWorldofWarcraftAuctionsUnitPrice(), SklearnLabelComplete()]


RED = "\033[31m"
RESET = "\033[0m"

def instantiate_evaluation_case(case_dir):
    import importlib
    case_package, case_class = case_dir.split('.')
    module = importlib.import_module(f"benchmarks.ICDBench.{case_package}")
    cls = getattr(module, case_class)
    evaluation_case = cls()
    return evaluation_case

def find_diverse_constraints_output_path(evaluation_case, approach_name):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench" / target_dir / "constraints")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"{approach_name}.yaml")
    return constraints_output_path

def find_constraints_output_path(evaluation_case, model_name):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench" / target_dir / "constraints")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"prismadv_constraints--{model_name}.yaml")
    return constraints_output_path


def find_single_prompt_constraints_output_path(evaluation_case, model_name):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench" / target_dir / "constraints")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"single_prompt_constraints--{model_name}.yaml")
    return constraints_output_path

def find_fewshot_prompt_constraints_output_path(evaluation_case, model_name):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench" / target_dir / "constraints")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"fewshot_prompt_constraints--{model_name}.yaml")
    return constraints_output_path

def find_minisweagent_task_output_path(evaluation_case):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench" / target_dir / "minisweagent")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"task.txt")
    return constraints_output_path

def find_minisweagent_constraints_output_path(evaluation_case):
    target_dir = evaluation_case.__class__.__module__.split('.')[-1] + "." + evaluation_case.__class__.__name__
    constraints_output_dir = (get_project_root() / "data_processed" / "icd_bench" / target_dir / "minisweagent")
    Path(constraints_output_dir).mkdir(parents=True, exist_ok=True)
    constraints_output_path = (constraints_output_dir / f"constraints.json")
    return constraints_output_path


ASSUMPTION_MATCH_PROMPT = """
You are an expert judge for a data engineering research project. The goal of the project is to test how well a prototype system can infer hidden assumptions from code. You will be given the ground truth assumption in natural language (GROUND_TRUTH_ASSUMPTION) and a list of candidate assumptions (CANDIDATE_ASSUMPTIONS) that were generated by the system. You have to judge whether any of the candidate assumptions semantically matches the ground truth assumption. If this is the case, return the index of the candidate assumption. 

BE VERY STRICT AND MAKE SURE THAT THE LOGIC OF THE GROUND TRUTH ASSUMPTION IS CAPTURED IN THE CANDIDATE ASSUMPTION!!! IN CASE OF DOUBT, DECIDE AGAINST REPORTING MATCHING ASSUMPTIONS.

If none of the candidate assumptions matches the ground truth assumptions, return -1.



### HERE ARE SOME EXAMPLES:

ground truth assumption:
The status column should have at most 4 distinct values.

candidate assumptions:
- index: 0; candidate_assumption: The 'status' column is expected to contain only a small set of distinct string values, specifically 'COMPLETED', 'CANCELLED', and 'IN_PROGRESS', as indicated by the data distribution. The code checks for the presence of null or missing values with 'if row['status']', implying that nulls could exist, and only adds non-null statuses to the set.
- index: 1; candidate_assumption: Given that the code filters out null or missing 'status' values before adding to the set, it assumes that nulls may be present in the input data. Therefore, the 'status' column could contain nulls, but the downstream logic operates only on non-null values.
- index: 2; candidate_assumption: The assertion 'assert len(statuses) <= 4' suggests that the total number of distinct 'status' values in the data should not exceed 4, accounting for the three known categories plus possibly an additional category or nulls. This indicates an expectation that the 'status' column should not contain unexpected or numerous unique values.

Expected answer:
{{ chosen_assumption: 2}}

ground truth assumption:
All non-null values in the status column should be have one of the following values: -1, 0, 1, 2, 3.

candidate assumptions:
- index: 0; candidate_assumption: The 'status' column should not contain null or missing values, as the code explicitly checks for non-null 'status' values before processing. Since the 'status' column is complete (no missing values), this assumption is consistent with the data.
- index: 1; candidate_assumption: The 'status' column contains only the values 'COMPLETED' and 'CANCELLED' in the dataset, each at 50%. The code uses a mapping for specific statuses ('COMPLETED' and 'CANCELLED') and also references other statuses ('IN_PROGRESS', 'UNDER_REVIEW') which are not present in the data. This suggests that the expected valid statuses include at least 'COMPLETED' and 'CANCELLED', and potentially others. To avoid errors during processing, the input data should only contain these known status values.
- index: 2; candidate_assumption: The 'status' column is of string data type, and the code compares string literals, implying that the data should be of string type with consistent casing. The dataset's 'status' column has only string values, supporting this assumption.

Expected answer:
{{ chosen_assumption: -1}}

### NOW TO YOUR TASK:

### Here is the ground truth assumption:
{ground_truth}

### Here are the candidate assumptions:
{candidate_assumption_list}

### Return a JSON object with the property chosen_assumption and your answer.
"""