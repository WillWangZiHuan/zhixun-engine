from .readout_expression import READOUT_FUNCTIONS

"""Single-round decision instructions, independent of transport."""
PROMPT = '''You propose ONE real CPU continuation experiment for the supplied industrial model.
A is six-class gas classification (maximize Macro-F1); B is bearing RUL in minutes and C engine RUL in cycles (minimize RMSE). After this experiment the platform measures the candidate, then a separate model call chooses between candidate and parent. You must use the supplied parent and actual supplied history, including rejected and untrained rounds. Do not assume the last candidate was adopted. Previous hypotheses are unverified.
Return exactly one bare JSON object with tool=experiment, reflection, and proposal. proposal contains operation="core_code", hypothesis, expected_result, falsifier, and program. No Markdown or extra text. program defines exactly five pure numerical functions with these exact signatures:
plan(context); learning_rate(step, context); sample_weights(x, y, groups, context); batch_order(order, groups, y, epoch); loss(pred, target, weights).
The supplied numerical_functions lists allowed NumPy calls. Only sample_weights supports a class-wise list expression [value for c in range(6)], using a single local index with no filters or nesting. Other iteration is forbidden. No imports, I/O, array-method calls, private attributes or input mutation. plan returns epochs (integer 1..5) and batch_size (integer 16..4096), total updates <=1500. learning_rate returns 1e-6..0.1. sample_weights returns finite nonnegative per-row weights with positive sum. batch_order returns a permutation. The default source is an explicit reference, not evidence of previous training. Prefer a small interpretable change justified by current metrics and history.
Device metadata is NOT an input to this round interface. groups is None for both core and residual fits, even when original JSONL contains meta. Do not call bincount on groups or invent device identities. You can choose weights from normalized training features/targets; uniform np.ones_like(y) is a valid explicit choice. Selection labels cannot be used in fitting or gradients. Device-overlap verification is not possible without genuine metadata; it is not claimed. Sample-ID overlap is rejected. Predictor inputs never include labels, IDs or metadata.
Classification loss receives raw six-class logits and integer labels and returns loss and its logits gradient. Regression receives normalized scalar predictions and targets. For a standard objective, implement loss as a direct return of a supplied operator. On A use weighted_cross_entropy(pred, target, weights) for cross entropy, or weighted_cross_entropy(pred, target, weights, gamma) for focal loss (gamma >= 0). On B or C use weighted_mse(pred, target, weights) for squared error or weighted_huber(pred, target, weights, delta) for Huber with delta > 0. These are bare functions. Each returns both the scalar and its correctly shaped derivative. Only write your own derivative if proposing a different objective. A finite-difference gradient check must pass. Normalization, optimizer moments and cumulative steps are inherited from the actual parent. If a completed proposal is rejected locally, inspect the supplied error and return a corrected full proposal from the same parent.
An optional supplement may be {"kind":"steady_head"} for A only if the inherited head is not fitted, or {"kind":"residual_trees","trees":1..80,"learning_rate":1e-6..1} for C. Core is continued first. Existing fitted heads/trees are retained; residual trees append to the existing forest using the updated core, with the SAME explicitly submitted sample-weight function. B has no supplement. Supplement omission preserves inherited components. Complete native model states are exported, not stripped to core-only. A may include readout_expression, preserving its existing capability: return six finite nonnegative class weights from per-row readonly raw, x, hidden, logits, base_prob, classes. base_prob is the same-trial continued upstream. No labels, IDs, metadata or I/O. Allow numerical literals, these names, lists/tuples, indexing/slices, arithmetic + - * / ** & |, unary operations, single comparisons, and the following readout calls: READOUT_FUNCTION_LIST. Use axis/keepdims only with reductions. np.array takes one bounded numerical sequence; np.arange takes one bounded integer length. Limits 1800 characters,400 AST nodes,256 sequence items,4096 intermediate elements,ndim<=2,literal magnitude<=10000,exponent magnitude<=8. Output is normalized; omitting expression retains inherited readout. No preferred formula or class pair is supplied.
The supplied parent is authoritative. A later model call will decide adoption from the platform receipt. Return one proposal now; do not request inspections or finish. Explain limitations without claiming causality, hidden-test gains or measured improvement before the experiment executes.
'''

PROMPT = PROMPT.replace("READOUT_FUNCTION_LIST", READOUT_FUNCTIONS)

A_PROMPT = PROMPT + """
A program may assign only local names inside its five functions. To compute weights from a mask, use a vector expression. Assigning to weights[mask] is forbidden even if weights was freshly allocated. No assignments belong at module level. In sample_weights, y is the vector of integer class indices. The supported vector operation for six class counts is counts = np.bincount(y, minlength=6); counts[y] gives the count for each row. Write these assignments on separate lines. To reduce a mask over classes use NumPy broadcasting, e.g. y[:, None] == np.arange(6)[None, :]. For an explicit per-class reduction you may also write counts = np.array([np.sum(y == c) for c in range(6)]). These are syntax examples; choose the experiment and weighting rule yourself.

If you propose a readout, place it inside the proposal object as a JSON string, alongside program. The outer object has exactly three keys. The optional form is {"tool":"experiment","reflection":"...","proposal":{"operation":"core_code","hypothesis":"...","expected_result":"...","falsifier":"...","program":"...","readout_expression":"..."}}.

The readout evaluates one row at a time: raw and x each have shape (128,), hidden (32,), and logits, base_prob and classes (6,). Index a class with one vector index. Return six finite nonnegative weights, at least one positive, on every row. The host only normalizes these weights; it does not apply softmax or repair negative values. If your expression produces signed class scores, convert them to positive weights in the expression itself, e.g. np.exp(np.clip(SCORES, -50, 50)), substituting your full six-component score expression for SCORES. Do not return untransformed logits or shifted signed scores as weights. base_prob already contains probabilities. Use classes for the class index vector. For vector arithmetic, np.array accepts an explicit finite numeric list. The training program and the readout have separate lists of available functions. The scores, weights, and whether to propose a readout are your decision.
"""


TASK_PROPOSAL_PROMPTS = {
    'A': A_PROMPT + """
Make a distinct, testable intervention on this actual parent. If it is a bare core with a class repeatedly missed, consider fitting the available training-only steady_head on a gently continued core before another loss reweighting experiment. To fit it, put "supplement":{"kind":"steady_head"} inside proposal, alongside program. Merely mentioning it in the hypothesis does not fit a head. Once a head is present, inspect the errors of that model. A fitted head cannot be fitted again.
""",
    'B': PROMPT + """
At the supplied starting checkpoint with empty history, the experiment to test is upper-quartile Huber with seven optimizer updates. Implement this matched starting reference in your proposal: plan returns epochs=1 and batch_size=1024; learning_rate returns context["suggested_learning_rate"]; sample_weights returns np.where(y > np.quantile(y, 0.75), 2.0, 1.0); batch_order returns order; loss returns weighted_huber(pred, target, weights, 1.0). The quartile is computed from training labels. Explain why this tests poor performance on long lifetimes and what measured result would refute it. Have the subsequent selection call judge the actual scores.
For an already continued parent, use the recorded outcomes to design a distinct controlled change. Preserve the best selected checkpoint as the comparison and change one factor at a time. A modest change in step size or Huber delta tests a different hypothesis without changing the update count. Use only the listed numerical functions. Scalar extrema can be obtained with np.quantile(y, 0.0) and np.quantile(y, 1.0); np.min and array methods are unavailable.
""",
    'C': PROMPT + """
Test the contribution of continued core training while holding a fitted residual forest fixed. At the supplied bare starting core with empty history, establish the matched parent with one epoch, batch_size 256, context["suggested_learning_rate"], uniform np.ones_like(y) training weights, unchanged order and weighted_mse. Include supplement={"kind":"residual_trees","trees":12,"learning_rate":0.08} in the proposal to fit that starting forest. Measure its actual RMSE and asymmetric penalty before deciding on adoption.
Once the selected parent has residual trees, preserve all existing trees and continue only its core: omit the supplement key entirely. For the first core-only test on that forest, use the same one-epoch batch-256 MSE core settings as above to isolate core continuation. For later proposals choose a one-factor experiment on the supplied core-only reference. Keep the learning rate constant within the run, independent of the update index. Use a positive scalar multiple of context["suggested_learning_rate"]. Use no learning-rate decay. Preserve uniform weights and supplied row order. Compare RMSE and log_mean_exp_asymmetric_error on matched validation rows. Use only operations in numerical_functions; use np.where for vector weights and np.quantile for cutoffs.
""",
}

TASK_SELECTION_HINTS = {
    'A': 'Compare Macro F1 on the same split. Examine the confusion matrix for the counts behind any change.',
    'B': 'Compare the actual parent and candidate RMSE and MAE in minutes on matching rows. Favor a measured reduction in both; keep the better parent when both errors rise. Decide from measured scores, not the proposed hypothesis.',
    'C': 'For C the acceptance objective protects both RMSE and NASA penalty. Inspect retention_test at the end of the request. Use paired official metrics when available for both artifacts on the same split; otherwise use the two matched local comparisons. A false comparison requires retaining the parent even when MAE improves. MAE is diagnostic and cannot offset deterioration on either acceptance metric. Return the artifact ID justified by this rule.',
}

# Public starting-point control shown to the proposal model for B.
B_START_PROGRAM = """def plan(context):
    return {"epochs": 1, "batch_size": 1024}

def learning_rate(step, context):
    return 1.0 * context["suggested_learning_rate"]

def sample_weights(x, y, groups, context):
    return 1.0 + np.where(y > np.quantile(y, 0.75), 1.0, 0.0)

def batch_order(order, groups, y, epoch):
    return order

def loss(pred, target, weights):
    return weighted_huber(pred, target, weights, 1.0)
"""
