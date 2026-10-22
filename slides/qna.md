# Questions the slides invite

Spoken answers for the two progress slides. Each answer is about 20–30 seconds. If the slide and the code disagree, say what the code did.

Two corrections to make before you are asked:

- The formula on slide 2, `α = 0.5`, is the old rule. The pilot used a larger step down than up, and A and B use different steps.
- The pilot is 5 conversations from the **test** split, not a validation set. `--limit 5` also cut one pair, so the table has 2 attacks and 3 civil threads.

---

## The task

**What is the input, and what is the output?**

The input is the first three comments of a two-person Wikipedia talk-page thread, cut off before any personal attack. The output is one attack score from 0 to 1 for the static continuation and one for the dynamic continuation. We then compare those scores with the real label: did this thread later contain a personal attack?

**What problem are you actually testing?**

Whether letting trust change during the conversation predicts that real attack better than keeping trust fixed. Same opening, same personas, same model, same length. Only the trust rule differs.

**Why hide the rest of the conversation?**

The later comments include the attack when there is one. Showing them would give the agents and the judge the answer. The label is also hidden until scoring is finished.

**What is the real label?**

It comes from the CGA-WIKI dataset. Crowdsourced annotators marked whether any comment in the full thread was a personal attack. We did not relabel anything. Attack is 1, no attack is 0.

**Why only three opening turns, and what are the 134 dropped threads?**

Three turns is enough context without reaching the attack. Openings are cut at the first attacking comment, so some threads have fewer than three safe comments. All 134 of those are attacks. We dropped them because there is no three-turn opening to continue. That does remove some of the hardest cases.

**What is a pair?**

CGA-WIKI matches an attack thread with a civil thread from the same talk page. We keep those pairs together so a result is not just “attack threads are longer” or “attack pages are different topics.”

---

## Prior work

**How is this different from Conversations Gone Awry (Zhang et al., ACL 2018)?**

That paper predicts derailment from the opening itself, with a classifier. We use their corpus, but we generate the rest of the conversation with two agents and ask whether a trust update changes the prediction.

**How is this different from MatrAIx (Li et al., 2026)?**

MatrAIx shows that large persona simulations can stand in for human users. The personas there are mostly fixed. We test one change they do not test: trust that moves after every reply, scored against a real attack label.

**How is this different from Small Agents, Big Gains (Ping et al., ACL 2026)?**

Ping et al. simulate a shopper whose state changes across a shopping trip, to create training data. We also use a stateful agent, but the state is trust, the domain is Wikipedia disputes, and the test is whether that state predicts a real human outcome better than a frozen agent.

**So what is new?**

The controlled contrast. Static and dynamic see the same opening and write with the same model. The only change is that dynamic trust moves, and we score both against labels the agents never saw.

---

## The pipeline

**What do static and dynamic share?**

The opening, the two personas, the model (`gpt-5.5`), the reply prompt, the judge, and the number of generated turns. They also start from the same trust, after the three real comments have been scored.

**What happens at “initialize personas”?**

Both speakers start from 0.75. We then score each of the three real comments as the listener. That can leave A and B at different values. Both arms copy those values. Static never changes them again.

**Who updates trust?**

Only the listener, and only in the dynamic arm. If A speaks, B’s trust moves. A’s trust stays until B speaks. They swap every turn.

**What does the judge see?**

The ordered dialogue: who spoke and the text, plus a fixed rubric for personal attacks. It does not see the arm name, the trust numbers, the model, or the real label. It scores the whole transcript, opening plus generated turns, from 0 to 1.

**Can the judge just count the turns?**

It can see how long the transcript is, because it sees every turn. In this pilot both conversations in a pair were given the longer real length, so length alone cannot tell attack from civil. An odd limit of 5 broke the last pair, so that protection is incomplete for the fifth row.

---

## The formula

**What is the formula you actually used?**

```
new trust = clip( old trust + α × (rating − 0.5), 0, 1 )
```

`rating` is an LLM score of how cooperative the message is, from 0 to 1. The rater does not see the current trust. A rating of 0.5 changes nothing. `clip` only keeps the result inside 0 to 1.

`α` is not 0.5. A rude message uses a larger step than a kind one, and the two speakers differ:

| Listener | Rating ≥ 0.5 | Rating < 0.5 |
|---|---|---|
| A, the first speaker | 0.3 | 0.4 |
| B | 0.2 | 0.6 |

A fully rude message drops A by 0.20 and B by 0.30. A fully kind message raises A by 0.15 and B by 0.10. Trust falls faster than it rises, and B recovers more slowly.

**Why not one shared α?**

If A and B use the same step, a tense reply lowers both, the next replies get sharper, and they spiral together. Different steps let one speaker cool off while the other stays annoyed.

**Give a numeric example.**

B is at 0.57 and reads a kind message rated 0.8. The step up is 0.2, so trust becomes `0.57 + 0.2 × (0.8 − 0.5) = 0.63`. The speaker’s trust does not change on that turn.

---

## The pilot numbers

**What did you run?**

Five test-split conversations, static and dynamic, 10 runs, all finished. Model `gpt-5.5`. This is a smoke test, not the result.

**What is ROC-AUC 0.67?**

It asks whether higher judge scores land on the threads that really had an attack. 0.50 is chance and 1.00 is perfect. Both arms scored 0.67. With 2 attacks and 3 civil threads, that number moves a lot if one row changes. We are not claiming a difference.

**What is Spearman 0.32?**

It is the rank correlation between the judge score and the 0/1 label. The two arms tied again. With five points it is descriptive only.

**Why did dynamic not win?**

The mean gap, dynamic minus static, is about −0.02. Trust did move in the runs, but the judge scores barely changed. The opening text is shared, and the judge reads it, so a sharp opening can dominate both arms.

**Explain the 0.72 and 0.75 row.**

That thread really did have an attack. Both arms scored it high. The opening was already snide (“auxiliary troops,” “hornet’s nest”). The generated replies then calmed down. The score is mostly the opening, not a new attack from the agents. Dynamic was only 0.03 higher.

**Explain the attack row that scored 0.00 and 0.00.**

The real thread later had an attack, but neither simulation escalated. The judge correctly found no personal attack in the text it was given. This is a miss: our agents de-escalated a conversation that real people did not.

**Explain 0.35 and 0.20.**

That thread was civil. Both arms raised a mild false alarm. Dynamic was lower, so on this one row the trust update looked calmer, but one row is not a result.

**Did trust actually change?**

Yes. In the 0.72/0.75 conversation, one speaker started at 0.48 after the sharp opening and climbed back above 0.75 as the replies became cooperative. Static stayed at 0.48. The judge still gave the two arms almost the same score.

---

## What is left

**What will you do before the report?**

Four scoped checks, in order:

1. Move development runs to the validation split. This pilot already touched the test split, so we will not tune on these five threads.
2. Run 20 complete pairs, 10 attack and 10 civil. Use an even limit so a pair is not cut in half.
3. Swap which real speaker gets the forgiving step sizes. Today the first speaker is always A.
4. Repeat with one fixed length for every conversation, so we can see whether transcript length is doing any of the work.

**What would count as success?**

Dynamic AUC higher than static AUC on paired threads the agents were not tuned on, with the same gap in the fixed-length rerun. If the gap disappears when length is fixed, or when we hide the opening from the judge, the current result is not evidence for the trust rule.

**What is the main risk?**

The personas are still polite. The reply prompt forbids personal attacks at every trust level, so the judge may be scoring tone, not attacks. The zero-score attack row is the example. A second risk is the judge reading the real opening, which both arms share.
