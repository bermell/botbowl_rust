# Fix enumerate dice roll outcomes

**Status:** **Shipped 2026-09-14.** `roll_outcomes::block_outcomes` replaces the all-`Pow` script.
Implementation diverges from the Python sketch below in two deliberate ways:

- **Outcomes are classified by *effect*, not by die face**, and there are six of them (attacker-best
  first): defender down *and pushed* (`Pow`, `PowPush` without Dodge, any push die into the crowd);
  defender down *in place* (`BothDown`, attacker Block only); push, nobody down; nothing happens
  (`BothDown`, both Block — *not* folded into push, the Python did); both down; attacker down. The
  attacker-Block-only roll that offers both a knockdown-and-push die and `BothDown` is emitted as a
  seventh **choice** child carrying `[Pow, BothDown, ..]`; `block_dice::scripted_pick` declines to
  pick on exactly that roll, so the search decides between down-in-place and down-and-pushed.
  Uphill, the defender resolves with the reversed order and takes down-in-place over down-and-pushed
  (no choice child).
- **Probabilities are exact counts over the `6^n` face combinations** (n ≤ 3) rather than the
  sequential `probability_left` formula — same numbers, and the choice class falls out for free.

Each child's dice array is a representative that forces the modelled outcome through the engine
(all slots the same face, so only one `Select*` is offered). The engine grew `Block::defender()`,
`Block::num_dices()` and `Push::is_crowd_push()` to keep it a pure function of `state`. The plan 035
default golden (`tests/data/lazy_mover_goldens.txt`) was re-blessed; the `#[ignore]`d full-matrix
golden was not and is stale.

---

the function `enumrate` in mcts `roll_outcomes.rs` currently on returns pows. However it's a delicate functionality that
should be fixed. The overall purpose is this:

This functions returns the probabilities of the outcomes of a block. The crux is that we need to assume outcome the
players wants. Which depends on _who_ decides. normally it's the attacking player who chooses, unless it's an uphill
block, then the defender chooses.

Depending on the skills of the attacker and defender we alter the outcomes as well which needs to be considered. For
example if the defender has the Dodge skill then the powpush die outcome is the same as a push And the if attacker or
defender has the block skill then the both down die means they don't go down.

## Example python implementation:

Here's a somewhat complete implementation in python. But there might be additional nuance to consider...

```python
def expand_block(game: botbowl.Game, parent: Node) -> Node:
    proc: botbowl.Block = game.get_procedure()
    assert type(proc) is botbowl.Block
    assert not proc.gfi, "Can't handle GFI:s here =( "
    assert proc.roll is None

    attacker: botbowl.Player = proc.attacker
    defender: botbowl.Player = proc.defender
    dice = game.num_block_dice(attacker, defender)
    num_dice = abs(dice)

    # initialize as 1d block without skills
    dice_outcomes = np.array([2, 2, 1, 1], dtype=int)
    DEF_DOWN, NOONE_DOWN, ALL_DOWN, ATT_DOWN = (0, 1, 2, 3)

    die_results = ([BBDieResult.DEFENDER_DOWN, BBDieResult.DEFENDER_STUMBLES],
                   [BBDieResult.PUSH],
                   [BBDieResult.BOTH_DOWN],
                   [BBDieResult.ATTACKER_DOWN])

    who_has_block = (attacker.has_skill(Skill.BLOCK), defender.has_skill(Skill.BLOCK))

    if any(who_has_block):
        dice_outcomes[ALL_DOWN] = 0
        die_results[ALL_DOWN].clear()

        if who_has_block == (True, True):  # both
            dice_outcomes[NOONE_DOWN] += 1
            die_results[NOONE_DOWN].append(BBDieResult.BOTH_DOWN)
        elif who_has_block == (True, False):  # only attacker
            dice_outcomes[DEF_DOWN] += 1
            die_results[DEF_DOWN].append(BBDieResult.BOTH_DOWN)
        elif who_has_block == (False, True):  # only defender
            dice_outcomes[ATT_DOWN] += 1
            die_results[ATT_DOWN].append(BBDieResult.BOTH_DOWN)

    crowd_surf: bool = game.get_push_squares(attacker.position, defender.position)[0].out_of_bounds

    if crowd_surf:
        dice_outcomes[DEF_DOWN] += 2
        dice_outcomes[NOONE_DOWN] -= 2
        die_results[DEF_DOWN].append(BBDieResult.PUSH)
        die_results[NOONE_DOWN].remove(BBDieResult.PUSH)
    elif defender.has_skill(Skill.DODGE):  # and not attacker.has_skill(Skill.TACKLE):
        dice_outcomes[DEF_DOWN] -= 1
        dice_outcomes[NOONE_DOWN] += 1
        die_results[DEF_DOWN].remove(BBDieResult.DEFENDER_STUMBLES)
        die_results[NOONE_DOWN].append(BBDieResult.DEFENDER_STUMBLES)

    prob = np.zeros(4)
    probability_left = 1.0
    available_dice = 6
    evaluation_order = [DEF_DOWN, NOONE_DOWN, ALL_DOWN, ATT_DOWN]
    if dice < 0:
        evaluation_order = reversed(evaluation_order)

    for i in evaluation_order:
        prob[i] = probability_left * (1 - (1 - dice_outcomes[i] / available_dice) ** num_dice)
        available_dice -= dice_outcomes[i]
        probability_left -= prob[i]

    assert available_dice == 0 and probability_left == approx(0) and prob.sum() == approx(1)

    new_parent = ChanceNode(game, parent)

    for prob, die_res in zip(prob, die_results):
        if prob == approx(0) or len(die_res) == 0:
            assert prob == approx(0) and len(die_res) == 0
            continue

        expand_with_fixes(game, new_parent, prob,
                          block_dice=np.random.choice(die_res, num_dice))

    assert sum(new_parent.child_probability) == approx(1.0)
    return new_parent
```
