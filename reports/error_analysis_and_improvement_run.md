# Error analysis and improvement run

**26 November 2026.** Temporal RoBERTa models, seed 4011. Development is Congress 113 (3,968 speeches). Test is the next Congress (3,970 speeches). Gold labels are the documented party, Democrat or Republican, and were not changed. No model was retrained. Scoring used the M1 GPU in 32-bit precision. The saved Week 2 files were scored in bfloat16, so a few labels differ: 5 of 3,968 development text labels and 5 of 3,970 test text labels.

## What was already known

| Partition | Text macro-F1 / accuracy | Text + chamber macro-F1 / accuracy |
|---|---:|---:|
| Development | 0.702 / 0.710 | 0.722 / 0.723 |
| Test | 0.622 / 0.632 | 0.658 / 0.659 |

The intervals in `week3_comparison.md` come from a speaker bootstrap. The local rescore of the original text is 0.702 / 0.709 on development and 0.621 / 0.631 on test. The chamber rescore is 0.723 / 0.724 and 0.657 / 0.658.

## Robustness edits

Rules were fixed before the test pass. Party names (`Democrat`, `Democratic`, `Republican`, `Republicans`, `GOP`, `Independent`) become `[PARTY]`. A leading "Mr. Speaker" or "Madam Chairman" style address is removed. The first sentence, or the last sentence, is removed when the speech has one. An edit that would leave the text empty keeps the original. On development, party masking changed 957 speeches, heading removal 1,622, and each sentence removal 3,964. On test, those counts are 829, 1,918, and 3,970.

Each cell is macro-F1 / accuracy from the local rescore. Flips are predictions that differ from that model's own original-text prediction. The pair after the flip count is Democrat-to-Republican, then Republican-to-Democrat. Correct-to-wrong and wrong-to-correct count flips against the gold label.

### Development

| Edit | Text macro-F1 / accuracy | Text flips (D→R, R→D) | Correct→wrong, wrong→correct | Chamber macro-F1 / accuracy | Chamber flips (D→R, R→D) | Correct→wrong, wrong→correct |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.702 / 0.709 | — | — | 0.723 / 0.724 | — | — |
| Party-name mask | 0.697 / 0.704 | 70 (59, 11) | 46, 24 | 0.708 / 0.708 | 89 (85, 4) | 76, 13 |
| Heading removed | 0.704 / 0.711 | 50 (47, 3) | 22, 28 | 0.722 / 0.722 | 43 (23, 20) | 24, 19 |
| First sentence removed | 0.701 / 0.708 | 114 (71, 43) | 59, 55 | 0.720 / 0.721 | 139 (69, 70) | 75, 64 |
| Last sentence removed | 0.701 / 0.709 | 140 (66, 74) | 71, 69 | 0.720 / 0.721 | 173 (88, 85) | 92, 81 |

### Test

| Edit | Text macro-F1 / accuracy | Text flips (D→R, R→D) | Correct→wrong, wrong→correct | Chamber macro-F1 / accuracy | Chamber flips (D→R, R→D) | Correct→wrong, wrong→correct |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.621 / 0.631 | — | — | 0.657 / 0.658 | — | — |
| Party-name mask | 0.613 / 0.621 | 59 (51, 8) | 48, 11 | 0.636 / 0.636 | 105 (94, 11) | 95, 10 |
| Heading removed | 0.629 / 0.637 | 64 (52, 12) | 19, 45 | 0.654 / 0.655 | 66 (26, 40) | 39, 27 |
| First sentence removed | 0.626 / 0.635 | 97 (70, 27) | 40, 57 | 0.647 / 0.648 | 126 (62, 64) | 83, 43 |
| Last sentence removed | 0.619 / 0.629 | 156 (72, 84) | 81, 75 | 0.656 / 0.656 | 225 (99, 126) | 115, 110 |

### What each edit result means

- **Party names help, on a minority of speeches.** Hiding them lowers the score on development and again on test. The chamber drop is 0.015 then 0.021. Most of those changes are a correct Democrat prediction becoming Republican. The conclusion is that the words are a real shortcut and it survives the next Congress. It is too small to explain the fall from about 0.70 to about 0.62, so hiding party names in a later training run would not by itself fix temporal generalization.
- **The last sentence is not the source of the score.** It changes the most individual predictions, and those swaps go both ways in roughly equal numbers. Macro-F1 stays within 0.003. The conclusion is that a long-context encoder is not justified by this check.
- **The first sentence is a small, uneven effect.** On development the score barely moves. On test the text score rises by 0.006 and the chamber score falls by 0.010. The conclusion is to leave the windowing rule as it is. The test movement is not large enough, and not consistent across the two models, to change preprocessing.
- **Removing the floor address does not hurt.** Text macro-F1 rises by 0.002 on development and 0.008 on test. On test, heading removal fixes 45 text predictions and breaks 19. The conclusion is that "Mr. Speaker" and "Madam Chairman" are weak cues. They are not a training priority.

## Development slices

Cuts were fixed before reading the accuracies. Short speeches are the lowest quarter of length, 146 words or fewer (998 speeches). Long speeches are the highest quarter, 543 words or more (992 speeches). A close call has predicted confidence below 0.60. A sure guess has confidence of at least 0.90. A party mention uses the same expression as the mask. Confidence is the higher of the two class probabilities.

| Slice | Text n | Text macro-F1 / accuracy | Chamber n | Chamber macro-F1 / accuracy |
|---|---:|---:|---:|---:|
| All | 3,968 | 0.702 / 0.709 | 3,968 | 0.723 / 0.724 |
| Short | 998 | 0.615 / 0.644 | 998 | 0.651 / 0.659 |
| Long | 992 | 0.781 / 0.784 | 992 | 0.804 / 0.804 |
| House | 2,538 | 0.675 / 0.680 | 2,538 | 0.692 / 0.692 |
| Senate | 1,430 | 0.750 / 0.762 | 1,430 | 0.777 / 0.780 |
| Party name present | 957 | 0.772 / 0.785 | 957 | 0.796 / 0.800 |
| No party name | 3,011 | 0.680 / 0.686 | 3,011 | 0.699 / 0.699 |
| Close call | 546 | 0.564 / 0.566 | 647 | 0.588 / 0.589 |
| Sure guess | 1,846 | 0.832 / 0.846 | 1,726 | 0.850 / 0.852 |

### What each slice means

- **Length matters more than the sentence edits.** The short quarter is at 0.615 macro-F1 and the long quarter is at 0.781 for the text model. The conclusion is to report short turns as their own error group. It does not say the model is failing only on one-line speeches: the minimum length in this set is 40 words, and the short cut is 146 words.
- **House speeches are harder than Senate speeches.** Text macro-F1 is 0.675 in the House and 0.750 in the Senate. Adding chamber raises the House score to 0.692 and the Senate score to 0.777. The conclusion is that chamber metadata helps both chambers and leaves most of the gap in place.
- **Speeches that name a party are easier, and most speeches do not name one.** Text macro-F1 is 0.772 with a party name and 0.680 without one. The conclusion matches the mask: the name is a useful clue where it appears, and the model still has a real score on the other 3,011 speeches.
- **Unsure predictions are near a coin flip, and sure predictions are usually right.** Close calls sit at 0.564 and 0.588 macro-F1. Sure guesses sit at 0.832 and 0.850. The conclusion is that both kinds of error exist. Because the two probabilities sum to 1, moving the 0.5 cutoff does not change any prediction.

A separate count of floor language (`unanimous consent`, `adjourn`, `morning hour`, `rollcall`) covers 342 development speeches. Their text accuracy is 0.778, above the 0.703 accuracy of the other speeches. The conclusion is that procedural language as a whole is not the weak slice.

## Confident mistakes

A sure mistake is a wrong prediction with confidence of at least 0.90. The text model has 285 of them. Fifty mention a party name, so most confident mistakes are not the party-name shortcut. Of the 285, 223 are Republican speeches predicted as Democratic, and 62 are the reverse. The chamber model has 256 sure mistakes, 42 with a party name, 146 Republican-to-Democrat and 110 Democrat-to-Republican.

The most sure mistakes are short House schedule announcements. Two examples, both gold Republican and predicted Democrat at 0.998, with no party name:

- `hein-daily:1130057577`, 56 words. The member asks unanimous consent to set the next morning-hour debate.
- `hein-daily:1130082711`, 52 words. The member asks unanimous consent for the next adjournment and morning-hour debate.

`hein-daily:1130020385` is the same kind of weekly schedule, 183 words, predicted Democrat at 0.996. It thanks "the Democratic whip" for yielding and then lists when the House will meet. The gold label stays Republican. The model is treating the floor script, and the other party's title, as evidence about the speaker.

These scripts dominate the top of the confidence list. They do not dominate the full set of sure mistakes: only 23 of the 285 text sure mistakes match the floor-language pattern above. The others include issue speeches, such as a farm bill, consumer finance, and war powers, where the model is sure and wrong without a party name.

The conclusion is to quote these as examples of misleading wording. It is not to correct the gold party. A later training change aimed at this error would have to separate a speaker's own party from a title they use for someone else, and would still leave the 235 sure mistakes that are not floor scripts.

## What this does and does not support changing

Supported as a description of the current models:

- Party names are a small, stable shortcut, stronger in the chamber model, and mostly one-directional.
- Short speeches and House speeches account for more of the development errors than any sentence edit.
- The dev-to-test drop remains the larger failure, and these edits do not remove it.

Not supported as a change to make in this run:

- Relabeling high-confidence errors. The documented party stays.
- A new decision threshold. It cannot change a two-class prediction when the probabilities sum to 1.
- A new encoder, longer context, or another training run. The sentence checks do not show that the score lives at the end of the speech, and the party-name effect is much smaller than the temporal drop.
- Hong Kong. The U.S. report and slides come first.

## Files

- Scores: `results/week3/robustness/dev/scores/` and `results/week3/robustness/test/scores/`
- Slice counts: `results/week3/robustness/dev/error_slices.json`
- Slides: `reports/week3_presentation.md`
- Matched Week 2 comparison: `reports/week3_comparison.md`
