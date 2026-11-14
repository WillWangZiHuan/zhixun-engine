"""A small chronological evidence table, not an embedding / LLM memory."""
from dataclasses import dataclass, field
import math


@dataclass
class Experience:
    entries: list = field(default_factory=list)

    def choose(self, context, windows):
        # Only recent, matching severity evidence influences the next action.
        recent = [e for e in self.entries if e['context'] == context][-8:]
        values = {window: [] for window in windows}
        for entry in recent:
            if entry['window'] in values:
                values[entry['window']].append(entry['reward'])
        unseen = [w for w in windows if not values[w]]
        if unseen:
            return unseen[0], {'reason': 'untried_window', 'evidence_ids': [e['id'] for e in recent]}
        chosen = max(windows, key=lambda w: sum(values[w]) / len(values[w]))
        return chosen, {'reason': 'recent_mean_reward', 'evidence_ids': [e['id'] for e in recent]}

    def record(self, *, context, window, status, reward, at, evidence):
        entry = dict(id=len(self.entries) + 1, context=context, window=window,
                     status=status, reward=float(reward), observed_through=at, evidence=evidence)
        self.entries.append(entry)
        return entry['id']

    def rollback(self, entry_id, loss_increase, at):
        entry = self.entries[entry_id - 1]
        entry['status'] = 'rolled_back'
        entry['reward'] = -abs(float(loss_increase))
        entry['rollback_observed_through'] = at

    def complete_followup(self, entry_id, *, start, end, rows, candidate_loss, incumbent_loss):
        entry = self.entries[entry_id - 1]
        if (entry['status'] != 'promoted' or 'followup' in entry
                or start != entry['observed_through'] or end - start != rows or rows <= 0
                or not all(math.isfinite(v) and v >= 0 for v in (candidate_loss, incumbent_loss))):
            raise ValueError('Invalid or duplicate post-promotion evidence')
        entry['followup'] = dict(start=start, end=end, rows=rows,
            normalized_candidate_loss=float(candidate_loss),
            normalized_incumbent_loss=float(incumbent_loss),
            normalized_gain=float(incumbent_loss-candidate_loss))
        return entry['followup'].copy()

    def driver_entries(self, *, at, reserved_rows, include_followup):
        result = []
        for entry in self.entries[-8:]:
            if entry['observed_through'] > at:
                raise ValueError('Experience is from a future observation')
            evidence = entry['evidence']
            value = {k: entry[k] for k in ('id', 'context', 'window', 'status', 'reward', 'observed_through')}
            value.update(evidence={k: evidence[k] for k in ('incumbent_loss', 'candidate_loss') if k in evidence},
                estimator=evidence.get('estimator'),
                training_cost={k: v for k, v in evidence.get('training', {}).items()
                    if k in ('wall_seconds', 'fit_seconds', 'peak_rss_bytes', 'train_rows', 'tune_rows')},
                error=evidence.get('error'))
            followup = entry.get('followup')
            # Never expose aggregate outcomes that touch the next tune/gate tail.
            if include_followup and followup and followup['end'] <= at-reserved_rows:
                value['followup'] = followup.copy()
            result.append(value)
        return result
