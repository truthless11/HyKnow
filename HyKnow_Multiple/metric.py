import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize as tknz
import math, re, argparse
import json, logging
import utils
from config import global_config as cfg
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from fuzzywuzzy import fuzz

en_sws = set(stopwords.words())


def similar(a, b):
    return a == b or a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]


def setsub(a, b):
    junks_a = []
    useless_constraint = ['temperature', 'week', 'est ', 'quick', 'reminder', 'near']
    for i in a:
        flg = False
        for j in b:
            if similar(i, j):
                flg = True
        if not flg:
            junks_a.append(i)
    for junk in junks_a:
        flg = False
        for item in useless_constraint:
            if item in junk:
                flg = True
        if not flg:
            return False
    return True


def setsim(a, b):
    a, b = set(a), set(b)
    return setsub(a, b) and setsub(b, a)


class BLEUScorer(object):
    # BLEU score calculator via GentScorer interface
    # it calculates the BLEU-4 by taking the entire corpus in
    # Calculate based multiple candidates against multiple references
    # code from https://github.com/shawnwun/NNDIAL
    def __init__(self):
        pass

    def score(self, parallel_corpus):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-10
        bp = 1 if c > r else math.exp(1 - float(r) / (float(c) + p0))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class METEORScorer(object):
    def __init__(self):
        pass

    def score(self, corpus):
        sample = 1e-10
        score = 0.0
        for hyp, ref in corpus:
            if hyp is not None and ref is not None:
                score += meteor_score(ref, hyp)
                sample += 1
        meteor = score/sample
        return meteor


class ROUGEScorer(object):
    def __init__(self):
        pass

    def score(self, corpus):
        sample = 1e-10
        score = 0.0
        rouge = Rouge()
        for hyp, ref in corpus:
            if hyp is not None and ref is not None:
                score += rouge.get_scores(hyp, ref)[0]['rouge-l']['f']  # consider ROUGE-L
                sample += 1
        rouge = score/sample
        return rouge


class GenericEvaluator:
    def __init__(self, reader):
        self.reader = reader
        self.metric_dict = {}

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            if not turn['sk']:  # only consider the original turns
                dials[dial_id].append(turn)
        return dials

    def run_metrics(self, data, ctr=False):
        raise ValueError('Please specify the evaluator first')


    def generation_metric(self, data, type='bleu'):
        # def clean(s):
        #     s = s.replace('<go_r> ', '')
        #     s = '<GO> ' + s
        #     return s

        gen, truth = [], []
        for row in data:
            gen.append(self.clean(row['resp_gen']))
            # gen.append(self.clean(row['resp']))
            truth.append(self.clean(row['resp']))
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        bleu = BLEUScorer().score(zip(wrap_generated, wrap_truth))
        meteor = METEORScorer().score(zip(gen, wrap_truth))
        rouge = ROUGEScorer().score(zip(wrap_generated, wrap_truth))
        return bleu, meteor, rouge


    def _normalize_constraint(self, constraint, ignore_dontcare=False, intersection=True):
        """
        Normalize belief span, e.g. delete repeated words
        :param constraint - {'food': 'asian oritental', 'pricerange': 'cheap'}
        :param intersection: if true, only keeps the words that appear in th ontology
                                        we set intersection=True as in previous works
        :returns: normalized constraint dict
                      e.g. - {'food': 'asian oritental', 'pricerange': 'cheap', 'area': ''}
        """
        normalized = {}
        for s in self.informable_slots:
            normalized[s] = ''
        for s, v in constraint.items():
            if ignore_dontcare and v == 'dontcare':
                continue
            if intersection and v != 'dontcare' and v not in self.entities_flat:
                continue
            normalized[s] = v
        return normalized


    def _normalize_act(self, aspn, intersection=False):
        aspn_list = aspn.split('|')
        normalized = {}
        for i, v in enumerate(aspn_list):
            seq = v.strip()
            word_set = set()
            for w in seq.split():
                if intersection:
                    if self.reader.act_order[i] == 'av':
                        if '[value' in w:
                            word_set.add(w)
                    else:
                        if w in self.requestable_slots:
                            word_set.add(w)
                else:
                    word_set.add(w)
            normalized[self.reader.act_order[i]] = word_set
        return normalized


    def tracker_metric(self, data, normalize=True):
        # turn level metric
        tp, fp, fn, db_correct = 0, 0, 0, 0
        goal_accr, slot_accr, total = 0, {}, 1e-8
        for s in self.informable_slots:
            slot_accr[s] = 0

        for row in data:
            if row['sk'] is True:  # only focus on original turns
                continue
            if normalize:
                gen = self._normalize_constraint(row['bspn_gen'])
                truth = self._normalize_constraint(row['bspn'])
            else:
                gen = self._normalize_constraint(row['bspn_gen'], intersection=False)
                truth = self._normalize_constraint(row['bspn'],  intersection=False)
            valid = 'thank' not in row['user'] and 'bye' not in row['user']
            if valid:
                for slot, value in gen.items():
                    if value in truth[slot]:
                        tp += 1
                    else:
                        fp += 1
                for slot, value in truth.items():
                    if value not in gen[slot]:
                        fn += 1

            if truth and valid:
                total += 1
                for s in self.informable_slots:
                    if gen[s] == truth[s]:
                        slot_accr[s] += 1
                if gen == truth:
                    goal_accr += 1
                if row['db_gen'] == row['db_match']:
                    db_correct += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        goal_accr /= total
        db_correct /= total
        for s in slot_accr:
            slot_accr[s] /= total
        return precision, recall, f1, goal_accr, slot_accr, db_correct


    def request_metric(self, data):
        # dialog level metric
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            truth_req, gen_req = set(), set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                resp_gen_token = self.clean(turn['resp_gen']).split()
                resp_token = self.clean(turn['resp']).split()
                for w in resp_gen_token:
                    if '[value_' in w and w.endswith(']') and w != '[value_name]':
                        gen_req.add(w[1:-1].split('_')[1])
                for w in resp_token:
                    if '[value_' in w and w.endswith(']') and w != '[value_name]':
                        truth_req.add(w[1:-1].split('_')[1])
            # print(dial_id)
            # print('gen_req:', gen_req)
            # print('truth_req:', truth_req)
                    # print('')
            for req in gen_req:
                if req in truth_req:
                    tp += 1
                else:
                    fp += 1
            for req in truth_req:
                if req not in gen_req:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        # print('precision:', precision, 'recall:', recall)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall


    def act_metric(self, data):
        # turn level metric
        tp, fp, fn = {'all_s': 0, 'all_v': 0}, {'all_s': 0, 'all_v': 0}, {'all_s': 0, 'all_v': 0}
        for s in self.requestable_slots:
            tp[s], fp[s], fn[s] = 0, 0, 0
            tp['[value_%s]'%s], fp['[value_%s]'%s], fn['[value_%s]'%s] = 0, 0, 0

        for row in data:
            gen = self._normalize_act(row['aspn_gen'])
            truth = self._normalize_act(row['aspn'])
            valid = 'thank' not in row['user'] and 'bye' not in row['user']
            if valid:
                # how well the act decoder captures user's requests
                for value in gen['av']:
                    if value in truth['av']:
                        tp['all_v'] += 1
                        if tp.get(value):
                            tp[value] += 1
                    else:
                        fp['all_v'] += 1
                        if fp.get(value):
                            fp[value] += 1
                for value in truth['av']:
                    if value not in gen['av']:
                        fn['all_v'] += 1
                        if fn.get(value):
                            fn[value] += 1

                # how accurately the act decoder predicts system's question
                if 'as' not in gen:
                    continue
                for slot in gen['as']:
                    if slot in truth['as']:
                        tp['all_s'] += 1
                        if tp.get(slot):
                            tp[slot] += 1
                    else:
                        fp['all_s'] += 1
                        if fp.get(slot):
                            fp[slot] += 1
                for slot in truth['as']:
                    if slot not in gen['as']:
                        fn['all_s'] += 1
                        if fn.get(slot):
                            fn[slot] += 1

        result = {}
        for k, v in tp.items():
            precision, recall = tp[k] / (tp[k] + fp[k] + 1e-8), tp[k] / (tp[k] + fn[k] + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            result[k] = [f1, precision, recall]
        return result


class MultiwozEvaluator(GenericEvaluator):
    def __init__(self, reader):
        super().__init__(reader)
        self.otlg = self.reader.otlg
        self.dataset = self.reader.dataset
        self.vocab = self.reader.vocab
        self.goals = json.loads(open(cfg.dial_goals, 'r').read().lower())
        self.informable_slots = self.reader.otlg.informable_slots
        self.requestable_slots = self.reader.otlg.requestable_slots
        self.eval_requestables = ['phone', 'address', 'postcode', 'reference', 'id']

    def clean(self, resp):
        return resp

    def run_metrics(self, results, ctr=False):
        metrics = {}
        bleu, meteor, rouge = self.generation_metric(results)
        p, r, f1, goal_acc, slot_acc, db_acc = self.tracker_metric(results, normalize=False)
        dt_p, dt_r, dt_f1, mrr5, r1, r5 = self.knowledge_selection_eval(results)
        success, inform, counts, dial_num = self.context_to_response_eval(results, ctr=ctr)

        logging.info('[RES] BLEU: %.3f  METEOR: %.3f  ROUGE-L: %.3f' % (bleu, meteor, rouge))
        logging.info('[TASK] Inform: %.3f  Success: %.3f' % (inform, success))
        logging.info('[KS] Detection F1: %.3f  MRR@5: %.3f  R@1: %.3f  R@5: %.3f' % (dt_f1, mrr5, r1, r5))
        # logging.info('[DST] joint goal: %.3f  slot p: %.3f  r: %.3f  f1: %.3f'%(goal_acc, p, r, f1))
        # slot_accu_str = ''
        # for slot, accu in slot_acc.items():
        #     slot_accu_str += '%s: %.3f '%(slot.split('-')[1], accu)
        logging.info('[DST] Joint Goal Acc: %.3f  DB Acc: %.3f' % (goal_acc, db_acc))

        metrics['bleu'] = bleu
        metrics['meteor'] = meteor
        metrics['rouge_l'] = rouge
        metrics['inform'] = inform
        metrics['success'] = success
        metrics['detection-p/r/f1'] = (dt_p, dt_r, dt_f1)
        metrics['mrr5'] = mrr5
        metrics['r1'] = r1
        metrics['r5'] = r5
        metrics['joint_goal'] = goal_acc
        metrics['slot_accu'] = slot_acc
        metrics['slot-p/r/f1'] = (p, r, f1)
        metrics['db_acc'] = db_acc

        if cfg.model_act:
            act_metric = self.act_metric(results)
            logging.info('[ACT] value f1: %.3f  slot f1: %.3f'%(act_metric['all_v'][0], act_metric['all_s'][0]))
            metrics['value_pred_f1'] = act_metric['all_v'][0]
            metrics['slot_pred_f1'] = act_metric['all_s'][0]
            metrics['act_verbose'] = act_metric
        else:
            metrics['value_pred_f1'], metrics['slot_pred_f1'], metrics['act_verbose']  = '', '', ''

        return metrics

    def reciprocal_rank(self, k_ref, k_pre, pre_num=5):
        match = []
        for idx in range(pre_num):
            match.append(fuzz.ratio(k_ref, k_pre[idx]) >= 95)
        if True in match:
            rank = match.index(True)
            return 1.0/(rank+1)
        else:
            return 0.0

    def recall_at_k(self, k_ref, k_pre, topk=5):
        match = []
        for idx in range(topk):
            match.append(fuzz.ratio(k_ref, k_pre[idx]) >= 95)
        if True in match:
            return 1.0
        else:
            return 0.0

    def compute_prf(self, score_sum, tp, fp, fn):
        if tp + fp > 0.0:
            score_p = score_sum/(tp + fp)
        else:
            score_p = 0.0

        if tp + fn > 0.0:
            score_r = score_sum/(tp + fn)
        else:
            score_r = 0.0

        if score_p + score_r > 0.0:
            score_f = 2*score_p*score_r/(score_p+score_r)
        else:
            score_f = 0.0

        return score_p, score_r, score_f

    def knowledge_selection_eval(self, data):
        detect_tp, detect_fp, detect_tn, detect_fn = 0.0, 0.0, 0.0, 0.0
        sum_mrr5, sum_r1, sum_r5 = 0.0, 0.0, 0.0

        for row in data:
            sk_true = row['sk']
            domains = row['dom'].split()
            bspn_gen = row['bspn_gen']
            sk_pre = False
            for dom in domains:
                if dom + "-sfek" in bspn_gen.keys():
                    sk_pre = True
            k_ref = row['knowledge']
            k_pre = row['knowledge_gen']

            if sk_true is True:
                if sk_pre is True:
                    detect_tp += 1
                    sum_mrr5 += self.reciprocal_rank(k_ref, k_pre, 5)
                    sum_r1 += self.recall_at_k(k_ref, k_pre, 1)
                    sum_r5 += self.recall_at_k(k_ref, k_pre, 5)
                else:
                    detect_fn += 1
            else:
                if sk_pre is True:
                    detect_fp += 1
                else:
                    detect_tn += 1

        detect_p, detect_r, detect_f = self.compute_prf(detect_tp, detect_tp, detect_fp, detect_fn)
        mrr_p, mrr_r, mrr_f = self.compute_prf(sum_mrr5, detect_tp, detect_fp, detect_fn)
        r1_p, r1_r, r1_f = self.compute_prf(sum_r1, detect_tp, detect_fp, detect_fn)
        r5_p, r5_r, r5_f = self.compute_prf(sum_r5, detect_tp, detect_fp, detect_fn)

        return detect_p, detect_r, detect_f, mrr_f, r1_f, r5_f

    def context_to_response_eval(self, data, eval_dial_list=None, same_eval_as_cambridge=True, ctr=False):
        dials = self.pack_dial(data)  # pack dialogs and remove aug turns
        counts = {}
        for req in self.eval_requestables:
            counts[req+'_total'] = 0
            counts[req+'_offer'] = 0

        dial_num, successes, matches = 0, 0, 0

        for dial_id in dials:
            if eval_dial_list and dial_id +'.json' not in eval_dial_list:
                continue
            dial = dials[dial_id]
            reqs = {}
            goal = {}
            for domain in self.otlg.all_domains:
                if self.goals[dial_id].get(domain):
                    true_goal = self.goals[dial_id]
                    goal = self._parseGoal(goal, true_goal, domain)
            # print(goal)
            for domain in goal.keys():
                reqs[domain] = goal[domain]['requestable']
            # print('\n',dial_id)
            success, match, stats, counts = self._evaluateGeneratedDialogue(dial, goal, reqs, counts, ctr=ctr,
                                                                    same_eval_as_cambridge=same_eval_as_cambridge)

            successes += success
            matches += match
            dial_num += 1

        # self.logger.info(report)
        succ_rate = successes/( float(dial_num) + 1e-10)
        match_rate = matches/(float(dial_num) + 1e-10)
        return succ_rate, match_rate, counts, dial_num

    def get_constraint_dict(self, constraint, intersection=False):
        """
        """
        normalized = {}
        for d_s, v in constraint.items():
            d, s = d_s.split('-')
            if d not in normalized:
                normalized[d] = {}
            normalized[d][s] =v
        return normalized



    def _evaluateGeneratedDialogue(self, dialog, goal, real_requestables, counts, ctr=False,
                                                          soft_acc=False, same_eval_as_cambridge=True):
        """Evaluates the dialogue created by the model.
            First we load the user goal of the dialogue, then for each turn
            generated by the system we look for key-words.
            For the Inform rate we look whether the entity was proposed.
            For the Success rate we look for requestables slots"""
        # for computing corpus success
         #'id'
        requestables = self.eval_requestables

        # CHECK IF MATCH HAPPENED
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []
        bspans = {}

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, turn in enumerate(dialog):
            sent_t = turn['resp_gen']
            # sent_t = turn['resp']
            for domain in goal.keys():
                # for computing success
                if same_eval_as_cambridge:
                    # [restaurant_name], [hotel_name] instead of [value_name]
                    if cfg.use_true_domain_for_ctr_eval or ctr:
                        dom_pred = turn['dom'].split()
                    else:
                        dom_pred = turn['dom_gen'].split()
                    # else:
                    #     raise NotImplementedError('Just use true domain label')
                    # print(domain, dom_pred)
                    if domain not in dom_pred:  # fail
                        continue
                if '[value_name]' in sent_t or '[value_id]' in sent_t:
                    if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        if cfg.use_true_bspn_for_ctr_eval or ctr:
                            bspn = turn['bspn']
                        else:
                            bspn = turn['bspn_gen']
                        # bspn = turn['bspn']

                        constraint_dict = self.get_constraint_dict(bspn)
                        if constraint_dict.get(domain):
                            venues = self.reader.db.queryJsons(domain, constraint_dict[domain], return_name=True)
                        else:
                            venues = []

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            # venue_offered[domain] = random.sample(venues, 1)
                            venue_offered[domain] = venues
                            bspans[domain] = constraint_dict[domain]
                        else:
                            # flag = False
                            # for ven in venues:
                            #     if venue_offered[domain][0] == ven:
                            #         flag = True
                            #         break
                            # if not flag and venues:
                            flag = False
                            for ven in venues:
                                if  ven not in venue_offered[domain]:
                                    flag = True
                                    break
                            # if flag and venues:
                            if flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                # venue_offered[domain] = random.sample(venues, 1)
                                venue_offered[domain] = venues
                                bspans[domain] = constraint_dict[domain]
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[value_name]'

                # ATTENTION: assumption here - we didn't provide phone or address twice! etc
                for requestable in requestables:
                    if requestable == 'reference':
                        if '[value_reference]' in sent_t:
                            if 'booked' in turn['db_vec'] or 'ok' in turn['db_vec']:  # if pointer was allowing for that?
                                provided_requestables[domain].append('reference')
                            # provided_requestables[domain].append('reference')
                    else:
                        if '[value_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            if 'name' in goal[domain]['informable']:
                venue_offered[domain] = '[value_name]'

            # special domains - entity does not need to be provided
            if domain in ['taxi', 'police', 'hospital']:
                venue_offered[domain] = '[value_name]'

            if domain == 'train':
                if not venue_offered[domain] and 'id' not in goal[domain]['requestable']:
                    venue_offered[domain] = '[value_name]'

        """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
        """
        # HARD EVAL
        stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                 'taxi': [0, 0, 0],
                 'hospital': [0, 0, 0], 'police': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['restaurant', 'hotel', 'attraction', 'train']:
                goal_venues = self.reader.db.queryJsons(domain, goal[domain]['informable'], return_name=True)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and len(set(venue_offered[domain])& set(goal_venues))>0:
                    match += 1
                    match_stat = 1
            else:
                if '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match)/len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        for domain in domains_in_goal:
            for request in real_requestables[domain]:
                counts[request+'_total'] += 1
                if request in provided_requestables[domain]:
                    counts[request+'_offer'] += 1

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                # for request in set(provided_requestables[domain]):
                #     if request in real_requestables[domain]:
                #         domain_success += 1
                for request in real_requestables[domain]:
                    if request in provided_requestables[domain]:
                        domain_success += 1

                # if domain_success >= len(real_requestables[domain]):
                if domain_success == len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success)/len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        return success, match, stats, counts



    def _parseGoal(self, goal, true_goal, domain):
        """Parses user goal into dictionary format."""
        goal[domain] = {}
        goal[domain] = {'informable': {}, 'requestable': [], 'booking': []}
        if 'info' in true_goal[domain]:
            if domain == 'train':
                # we consider dialogues only where train had to be booked!
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append('reference')
                if 'reqt' in true_goal[domain]:
                    if 'id' in true_goal[domain]['reqt']:
                        goal[domain]['requestable'].append('id')
            else:
                if 'reqt' in true_goal[domain]:
                    for s in true_goal[domain]['reqt']:  # addtional requests:
                        if s in ['phone', 'address', 'postcode', 'reference', 'id']:
                            # ones that can be easily delexicalized
                            goal[domain]['requestable'].append(s)
                if 'book' in true_goal[domain]:
                    goal[domain]['requestable'].append("reference")

            for s, v in true_goal[domain]['info'].items():
                s = self.otlg.slot_normlize.get(s, s)
                if len(v.split())>1:
                    v = ' '.join(tknz(v))
                if '|' in v:   # do not consider multiple names
                    v = v.replace('|',' | ').split('|')[0]
                v = v.strip()
                goal[domain]["informable"][s] = v

            if 'book' in true_goal[domain]:
                goal[domain]["booking"] = true_goal[domain]['book']
        return goal


def metric_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-type')
    args = parser.parse_args()
    ev_class = MultiwozEvaluator
    # elif args.type == 'kvret':
    #     ev_class = KvretEvaluator
    ev = ev_class(args.file)
    # ev.run_metrics()
    ev.dump()


if __name__ == '__main__':
    metric_handler()
