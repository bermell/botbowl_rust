# Skills & Traits — implementation status

Full BB2025 skill/trait list from `Skills & Traits - Blood Bowl Base.html` in this
directory, cross-referenced against `botbowl-engine`'s `Skill` enum
(`botbowl-engine/src/core/table.rs:59`). That enum currently has **6 variants**:
`Dodge`, `Throw`, `Block`, `Catch`, `SureHands`, `SureFeet` — everything else in the
rulebook (~100 skills/traits) has no representation in the engine at all.

- **Implemented** — engine code actually checks/grants this skill's effect (not just
  an enum variant name).
- **Tested** — a test gives the skill via `give_skill`/`has_skill` and asserts on its
  effect by name. "—" means implementation status makes this moot.
- **Effort to add** — how it would land in the codebase if implemented today:
  - **Simple** — a dice/roll modifier, threshold change, or restriction on an
    *existing* action. Same shape as the 6 already implemented: add a `Skill`
    variant, check `has_skill()` in the relevant procedure, add a `SKILL_PLANES`
    entry in `botbowl-nn/src/encode.rs` (see prior discussion in this
    conversation — old training data stays valid, it just reads 0 on the new
    plane). No change to the action space.
  - **Bigger** — grants a brand-new declarable action (a "Special Action" in
    rulebook terms, or an action-shape change like targeting two squares at
    once). This expands the legal-action space itself (the CSR
    `actions.npy`/`policy.npy` encoding + MCTS action generation/pruning), not
    just the tensor's feature planes — a materially larger change than adding a
    skill plane.
  - **Simple\*** — a plain modifier, but it only does anything once **Throw
    Team-mate** (itself a "Bigger" item — an entire missing Action) has been
    added first.

| Skill | Category | Short description | Implemented | Tested | Effort to add |
|---|---|---|---|---|---|
| Catch | Agility | Re-roll a failed Agility Test to catch the ball | Yes (`ball_procs.rs::Catch::reroll_skill`) | No | Simple |
| Diving Catch | Agility | Catch a ball landing in your TZ from pass/throw-in/kick-off; +1 in target square | No | — | Simple |
| Diving Tackle | Agility | -2 to an opponent's dodge/leap/jump roll, then go prone in the vacated square | No | — | Simple |
| Dodge | Agility | Re-roll one failed dodge Agility Test per turn; also affects the Stumble result | Yes (`movement_procs.rs`, `block_procs.rs`) | Yes (`dodge_reroll`) | Simple |
| Defensive | Agility | Marked opponents can't use Guard or Put the Boot In | No | — | Simple |
| Hit and Run | Agility | After a Block/Stab, move one free square ignoring Tackle Zones | No | — | Simple |
| Jump Up | Agility | Stand up for free while prone; can attempt a Block Action while prone | No | — | Simple |
| Leap | Agility | Leap over an adjacent square regardless of contents, reduced negative modifier | No | — | Simple |
| Safe Pair of Hands | Agility | Place the ball in an adjacent square instead of it bouncing when knocked down | No | — | Simple |
| Sidestep | Agility | Choose your own square when pushed back | No | — | Simple |
| Sprint | Agility | One extra Rush attempt during a Move Action | No | — | Simple |
| Sure Feet | Agility | Re-roll one Rush (GFI) die per turn | Yes (`movement_procs.rs::GfiProc::reroll_skill`) | No | Simple |
| Dirty Player | Devious | +1 to Armour or Injury Roll during a Foul Action | No | — | Simple |
| Eye Gouge | Devious | A player pushed back by you can't assist until next activated | No | — | Simple |
| Fumblerooski | Devious | Drop the ball in a vacated square during a Move Action, no turnover | No | — | Simple |
| Lethal Flight | Devious | +1 Armour/Injury when a thrown player (with Right Stuff) knocks an opponent down on landing | No | — | Simple\* |
| Lone Fouler | Devious | Re-roll a failed Armour Roll on an unassisted Foul | No | — | Simple |
| Pile Driver | Devious | Free Foul Action after knocking an opponent down in a Block, then go prone | No | — | Simple |
| Put the Boot In | Devious | Provide an Offensive Assist on Fouls regardless of how many are marking you | No | — | Simple |
| Quick Foul | Devious | Continue your Move Action after a Foul Action | No | — | Simple |
| Saboteur | Devious | Secret-weapon player: chance to also knock down the blocker when knocked down | No | — | Simple |
| Shadowing | Devious | Chance to follow an opponent dodging out of your Tackle Zone | No | — | Simple |
| Sneaky Git | Devious | Not sent off for a natural double Armour Roll on a Foul (unless armour breaks) | No | — | Simple |
| Violent Innovator | Devious | Earn SPP for casualties caused via Special Actions | No | — | Simple |
| Block | General | Choose not to be knocked down on a Both Down result | Yes (`block_procs.rs`) | No | Simple |
| Dauntless | General | Roll to temporarily match a higher-Strength opponent for a Block | No | — | Simple |
| Fend | General | Opponent can't Follow-up after pushing you back | No | — | Simple |
| Frenzy* | General | Must Follow-up and make a second Block Action if the target is Pushed Back | No | — | Simple |
| Kick | General | Kicked ball may deviate D3 instead of D6 | No | — | Simple |
| Pro | General | Once per activation, re-roll one die on a 3+ | No | — | Simple |
| Steady Footing | General | On a 6, avoid being Knocked Down/Fall Over | No | — | Simple |
| Strip Ball | General | Ball carrier drops the ball when pushed back by your Block | No | — | Simple |
| Sure Hands | General | Re-roll a failed pick-up; immune to Strip Ball | Yes (`ball_procs.rs::PickupProc::reroll_skill`) | Yes (`pickup_success`) | Simple |
| Tackle | General | Opponent can't use Dodge Skill leaving your TZ, or vs. a Stumble result | No | — | Simple |
| Taunt | General | Force an opponent to Follow-up when they push you back | No | — | Simple |
| Wrestle | General | Both Down becomes both players placed prone, regardless of other skills | No | — | Simple |
| Big Hand | Mutation | Ignore all negative modifiers when picking up the ball | No | — | Simple |
| Claws | Mutation | Natural 8+ on an Armour Roll you inflict always breaks armour | No | — | Simple |
| Disturbing Presence* | Mutation | -1 to opposition Pass/Throw/Catch/Intercept tests within 3 squares | No | — | Simple |
| Extra Arms | Mutation | +1 Agility Test to Catch/Pick Up/Intercept | No | — | Simple |
| Foul Appearance* | Mutation | Chance to cancel an opponent's Block/Special Action targeting you | No | — | Simple |
| Horns | Mutation | +1 Strength for Block Actions during a Blitz | No | — | Simple |
| Iron Hard Skin | Mutation | No Armour Roll modifiers against you; immune to Claws | No | — | Simple |
| Monstrous Mouth | Mutation | Chomp Special Action roots an adjacent standing opponent in place | No | — | Bigger (new Special Action) |
| Prehensile Tail | Mutation | Extra -1 Agility Test modifier for opponents leaving your TZ | No | — | Simple |
| Tentacles | Mutation | Opposed roll to prevent an opponent leaving your Tackle Zone | No | — | Simple |
| Two Heads | Mutation | +1 Agility Test when dodging | No | — | Simple |
| Very Long Legs | Mutation | +1 Leap/Jump, +2 Intercept, ignore Cloud Burster | No | — | Simple |
| Accurate | Passing | +1 Passing Ability Test on a Quick Pass or Short Pass | No | — | Simple |
| Cannoneer | Passing | +1 Passing Ability Test on a Long Pass or Long Bomb | No | — | Simple |
| Cloud Burster | Passing | Opponents can't attempt to Intercept this pass | No | — | Simple |
| Dump-off | Passing | Free Quick Pass when targeted by a Block/Special Action | No | — | Simple |
| Give and Go | Passing | Continue your Move Action after a Quick Pass or Hand-off | No | — | Simple |
| Hail Mary Pass | Passing | Throw to any square on the pitch as a Long Bomb; can't be intercepted | No | — | Simple |
| Leader | Passing | Grants the team an extra ("Leader") re-roll while on the pitch | No | — | Simple |
| Nerves of Steel | Passing | Ignore Marking modifiers when Catching or Passing | No | — | Simple |
| On the Ball | Passing | Move up to 3 squares in response to an opposing Pass Action or the kick-off | No | — | Simple |
| Pass | Passing | Re-roll a failed Passing Ability Test | No — `Skill::Throw` exists and is granted to the Thrower template, but no code checks it; pass accuracy is driven only by base `pass` stat + modifiers | — | Simple |
| Punt | Passing | Punt Special Action to kick the ball downfield | No | — | Bigger (new Special Action) |
| Safe Pass | Passing | A natural 1 on a Pass doesn't fumble; ends activation instead, no turnover | No | — | Simple |
| Arm Bar | Strength | +1 Armour/Injury Roll when an opponent falls dodging/leaping/jumping from your TZ | No | — | Simple |
| Brawler | Strength | Re-roll a single Both Down result | No | — | Simple |
| Break Tackle | Strength | Once per turn, +1 to +3 Agility Test bonus when dodging, based on your Strength | No | — | Simple |
| Bullseye | Strength | A Superb Throw Team-mate result lands exactly, no scatter | No | — | Simple\* |
| Grab | Strength | Choose the push-back square; opponent can't use Sidestep | No | — | Simple |
| Guard | Strength | Provide Offensive/Defensive Assist regardless of how many are marking you | No | — | Simple |
| Juggernaut | Strength | Both Down treated as Pushed Back during a Blitz; opponent can't Fend/Stand Firm/Wrestle | No | — | Simple |
| Mighty Blow | Strength | +1 Armour/Injury Roll whenever you knock an opponent down in a Block | No | — | Simple |
| Multiple Block | Strength | Block two adjacent opponents at once, at -2 Strength | No | — | Bigger (Block Action gains a second targeted square) |
| Stand Firm | Strength | Choose not to be pushed back during a Block | No | — | Simple |
| Strong Arm | Strength | +1 Passing Ability Test on a Throw Team-mate Action | No | — | Simple\* |
| Thick Skull | Strength | Knocked-out only on a 9 (or 8 if also Stunty) instead of 8 (or 7) | No | — | Simple |
| Always Hungry* | Trait | Risk eating your own team-mate when performing a Throw Team-mate Action | No | — | Simple\* |
| Animal Savagery* | Trait | Risk attacking an adjacent team-mate instead of acting, on activation | No | — | Simple |
| Animosity (X)* | Trait | Risk refusing to Pass/Hand-off to team-mates with a given Keyword | No | — | Simple |
| Ball & Chain* | Trait | Forced random-direction movement Special Action every activation | No | — | Bigger (new Special Action) |
| Bloodlust (X+)* | Trait | Must bite a Thrall team-mate or become Distracted, on activation | No | — | Simple |
| Bombardier | Trait | Throw Bomb Special Action | No | — | Bigger (new Special Action) |
| Bone Head* | Trait | Risk becoming Distracted on activation | No | — | Simple |
| Breathe Fire | Trait | Special Action to knock down/place prone an adjacent opponent | No | — | Bigger (new Special Action) |
| Chainsaw* | Trait | Chainsaw Attack Special Action: +3 Armour Roll modifier, risk of kick-back | No | — | Bigger (new Special Action) |
| Decay* | Trait | +1 modifier to any Casualty Roll made against you | No | — | Simple |
| Drunkard* | Trait | -1 modifier whenever attempting to Rush | No | — | Simple |
| Hatred (X)* | Trait | Re-roll a Both Down result against a hated Keyword | No | — | Simple |
| Hypnotic Gaze | Trait | Special Action to Distract an adjacent standing opponent | No | — | Bigger (new Special Action) |
| Insignificant* | Trait | Roster-building limit: can't outnumber non-Insignificant players | No | — | Simple (roster rule, not in-game action) |
| Kick Team-mate | Trait | Special Action like Throw Team-mate, doesn't use the team's Throw Team-mate slot | No | — | Bigger (new Special Action) |
| Loner (X+)* | Trait | Risk losing a Team Re-roll whenever you try to use one | No | — | Simple |
| My Ball* | Trait | Can't willingly give up the ball (no Pass/Hand-off/etc. while carrying it) | No | — | Simple |
| No Ball* | Trait | Can never possess or attempt to Catch/Pick Up/Intercept the ball | No | — | Simple |
| Pick-Me-Up | Trait | Chance to stand nearby prone team-mates during the opponent's turn | No | — | Simple |
| Plague Ridden | Trait | Gain an extra Lineman on your roster when you cause a Dead result | No | — | Simple |
| Pogo | Trait | Jump over an adjacent square, ignoring all negative modifiers | No | — | Simple |
| Projectile Vomit | Trait | Special Action: unmodified Armour Roll against an adjacent opponent (or self) | No | — | Bigger (new Special Action) |
| Really Stupid* | Trait | Risk becoming Distracted on activation unless a nearby team-mate helps | No | — | Simple |
| Regeneration | Trait | Chance to ignore a Casualty and return to the Reserves box instead | No | — | Simple |
| Right Stuff* | Trait | Can be thrown by a team-mate even while Prone | No | — | Simple\* |
| Secret Weapon* | Trait | Automatically sent off at the end of the drive | No | — | Simple |
| Stab | Trait | Special Action: unmodified Armour Roll against an adjacent opponent | No | — | Bigger (new Special Action) |
| Stunty* | Trait | No Marking penalty when dodging; -1 to Intercept; uses the Stunty Injury Table | No | — | Simple |
| Swoop | Trait | Controlled landing when thrown: no scatter, re-roll the landing Agility Test | No | — | Simple\* |
| Take Root* | Trait | Risk becoming Rooted (can't move or be pushed) on activation | No | — | Simple |
| Throw Team-mate | Trait | Grants the Throw Team-mate Action | No | — | Bigger (new Action) |
| Timmm-ber! | Trait | Bonus to the stand-up roll from adjacent team-mates (low-MA players) | No | — | Simple |
| Titchy* | Trait | +1 to dodge; doesn't impose the usual Marking penalty on opponents | No | — | Simple |
| Trickster | Trait | Reposition yourself before a Block/Special Action targeting you resolves | No | — | Simple |
| Unchannelled Fury* | Trait | Risk your activation ending immediately before acting | No | — | Simple |
| Unsteady* | Trait | Can't declare Secure the Ball Actions | No | — | Simple |

## Summary

- **Implemented:** Dodge, Block, Catch, Sure Hands, Sure Feet (5 of ~107 skills/traits).
- **Tested by name:** Dodge (`dodge_reroll`), Sure Hands (`pickup_success`). Block, Catch
  and Sure Feet are implemented but no test grants/asserts them by name — coverage is
  incidental (e.g. `test_block_2d_bothdown_casualty` only exercises the no-skill branch).
- Everything else in the rulebook — all of Devious, Mutation, Strength, and Traits, plus
  most of Agility/General/Passing — has no engine representation at all.
- **Effort to add:** the large majority (~95) are "Simple" — same shape as the 6 already
  implemented, and safe to layer onto existing training data (old samples just read 0 on
  the new skill plane). A smaller set (~12: Monstrous Mouth, Punt, Multiple Block,
  Ball & Chain, Bombardier, Breathe Fire, Chainsaw, Hypnotic Gaze, Kick Team-mate,
  Projectile Vomit, Stab, Throw Team-mate) is "Bigger" — each introduces a new declarable
  action, which touches the legal-action space (MCTS action generation/pruning, the CSR
  policy encoding) rather than just a tensor feature plane. Six more (Lethal Flight,
  Bullseye, Strong Arm, Always Hungry, Right Stuff, Swoop) are simple modifiers that are
  inert until Throw Team-mate itself is added.
