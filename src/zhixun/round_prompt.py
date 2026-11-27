"""Single-round decision instructions, independent of transport."""
PROMPT = '''You propose ONE real CPU continuation experiment for the supplied industrial model.
A is six-class gas classification (maximize Macro-F1); B is bearing RUL in minutes and C engine RUL in cycles (minimize RMSE). After this experiment the platform measures the candidate, then a separate model call chooses between candidate and parent. You must use the supplied parent and actual supplied history, including rejected and untrained rounds. Do not assume the last candidate was adopted. Previous hypotheses are unverified.
Return exactly one bare JSON object with tool=experiment, reflection, and proposal. proposal contains operation="core_code", hypothesis, expected_result, falsifier, and program. No Markdown or extra text. program defines exactly five pure numerical functions with these exact signatures:
plan(context); learning_rate(step, context); sample_weights(x, y, groups, context); batch_order(order, groups, y, epoch); loss(pred, target, weights).
The supplied numerical_functions lists allowed NumPy calls. Only sample_weights supports a class-wise list expression [value for c in range(6)], using a single local index with no filters or nesting. For six-class row weighting use a single vectorized return expression. For example: return np.where(y == 0, 2.0, np.where(y == 3, 1.5, 1.0)), or return np.asarray([2.0, 1.0, 1.0, 1.5, 1.0, 1.0])[y]. Choose label weights from your hypothesis. Never assign into array elements or iterate over y. Other iteration is forbidden. No imports, I/O, array-method calls, private attributes or input mutation. plan returns epochs (integer 1..5) and batch_size (integer 16..4096). learning_rate returns 1e-6..0.1. sample_weights receives already-normalized x, integer class IDs for A, and standardized targets y=(lifetime-target_mean)/target_scale for B/C. Standardized y can be negative and its mean can be zero. Use positive relative weights, for example np.where(y > np.quantile(y, 0.75), 2.0, 1.0) for long-life regression rows. Return one finite nonnegative weight per input row with positive sum. loss receives normalized targets and predictions. Reported validation errors are in original units. batch_order returns a permutation. The default source is an explicit reference, not evidence of previous training. Prefer a small interpretable change justified by current metrics and history.
Device metadata is NOT an input to this round interface. groups is None for core training, even when original JSONL contains meta. Do not call bincount on groups or invent device identities. You can choose weights from normalized training features/targets; uniform np.ones_like(y) is a valid explicit choice. Selection labels cannot be used in fitting or gradients. Device-overlap verification is not possible without genuine metadata; it is not claimed. Sample-ID overlap is rejected. Predictor inputs never include labels, IDs or metadata.
Classification loss receives raw six-class logits and integer labels and returns loss and its logits gradient. Regression receives normalized scalar predictions and targets. For a standard objective, implement loss as a direct return of a supplied operator. On A use weighted_cross_entropy(pred, target, weights) for cross entropy, or weighted_cross_entropy(pred, target, weights, gamma) for focal loss (gamma >= 0). On B or C use weighted_mse(pred, target, weights) for squared error or weighted_huber(pred, target, weights, delta) for Huber with delta > 0. These are bare functions. Each returns both the scalar and its correctly shaped derivative. Only write your own derivative if proposing a different objective. A finite-difference gradient check must pass. Normalization, optimizer moments and cumulative steps are inherited from the actual parent. If a completed proposal is rejected locally, inspect the supplied error and return a corrected full proposal from the same parent.
Use the inherited MLP and Adam state throughout. The caller loads and saves the official MLP format. Only A also permits optional boolean steady_channel_dropout. The supplied starting reference is an experiment to test. Compare the two artifacts on this round's matching rows.
'''

TASK_PROPOSAL_PROMPTS = {
    'A': PROMPT + """ The inherited MLP first learns to separate ethanol and acetaldehyde from the steady-response channels on training rows while keeping the parent logits for other training classes close. The auxiliary update uses the existing four parameter arrays. In the first round test the supplied gentle six-class Adam program on complete inputs. Set steady_channel_dropout to false or omit it initially; in subsequent proposals you may test it on training data, or vary one Adam parameter in response to paired Macro F1 and accuracy.""",
    'B': PROMPT + """ B first takes up to six damped Gauss-Newton steps on the supplied MLP weights using weighted Huber residuals on training rows. It then runs your Adam program from that same network and inherited optimizer moments. At the official start use a short gentle Adam continuation so it preserves those parent-weight updates. The only deployed predictor is the official MLP. From later selected parents compare one factor at a time; if strong updates raise either error, reduce the learning rate and update count. Compare RMSE and MAE on identical validation rows.""",
    'C': PROMPT + """ First take two passes of input-covariance-conditioned gradients on the existing parent weights from training rows. Then refine the same four weight arrays in whitened training coordinates with a fourth-power penalty against large deviations from the pre-update output. The predictor stays the official MLP. Run your Adam program using the inherited moments. At the official start test the supplied gentle Adam program as written. For later rounds compare MSE with smooth tail-sensitive losses using their exact analytic gradients or a smaller rate. Check both RMSE and the asymmetric lifetime penalty before adopting.""",
}

TASK_SELECTION_HINTS = {
    'A': 'Compare Macro F1 on the same split. Examine the confusion matrix for the counts behind any change.',
    'B': 'Compare parent and candidate RMSE and MAE in minutes on matching rows. Also inspect short_life_rmse on the same lowest quartile of validation lifetimes (cutoff and count supplied). Explain the tradeoff if lower aggregate error comes with worse near-failure predictions. Prefer an update that improves errors across conditions.',
    'C': 'For C the acceptance objective protects both RMSE and NASA penalty. Inspect retention_test at the end of the request. Use paired official metrics when available for both artifacts on the same split; otherwise use the two matched local comparisons. A false comparison requires retaining the parent even when MAE improves. MAE is diagnostic and cannot offset deterioration on either acceptance metric. Return the role justified by this rule.',
}

START_PROGRAMS = {
    'A': '''def plan(context):
    return {"epochs": 1, "batch_size": 128}
def learning_rate(step, context):
    return 0.0001
def sample_weights(x, y, groups, context):
    return np.ones_like(y)
def batch_order(order, groups, y, epoch):
    return order
def loss(pred, target, weights):
    return weighted_cross_entropy(pred,target,weights)
''',
    'B': '''def plan(context):
    return {"epochs": 1, "batch_size": 4096}
def learning_rate(step, context):
    return 0.0002
def sample_weights(x, y, groups, context):
    return np.ones_like(y)
def batch_order(order, groups, y, epoch):
    return order
def loss(pred, target, weights):
    return weighted_huber(pred,target,weights,0.5)
''',
    'C': '''def plan(context):
    return {"epochs": 1, "batch_size": 1024}
def learning_rate(step, context):
    return 0.00005
def sample_weights(x, y, groups, context):
    return np.ones_like(y)
def batch_order(order, groups, y, epoch):
    return order
def loss(pred, target, weights):
    return weighted_mse(pred,target,weights)
''',
}

