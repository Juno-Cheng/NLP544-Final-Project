# 3-minute progress talk

Read this with slide 2. About 430 words, roughly 3 minutes. Pause at the line breaks.

---

This slide is where the project is right now. The question is simple: if trust changes while two people talk, do we predict a real personal attack better than if trust stays fixed?

Start on the left. Every run begins with a real Wikipedia talk-page opening: the first three comments, and we stop before anyone attacks. The agents never see the rest of the thread, and they never see the label.

Next we score those three comments. That gives each speaker a starting trust. The static run and the dynamic run both start from those same numbers, so the only difference later is whether trust is allowed to move.

Then the two agents keep talking. In the static arm, trust stays where it is, so the tone stays the same. In the dynamic arm, the person who just received a message updates their trust. A kind reply raises it a little. A rude reply lowers it more. Speaker A gets over it faster. Speaker B stays annoyed longer. The next reply follows that mood.

A judge then reads only the dialogue. It does not know which arm it is looking at, and it does not know whether the real conversation had an attack. After both arms are scored, we compare those scores with the real label.

On the right is a pilot of five conversations. All ten runs finished. Read the table down the rows.

The first one was a real attack. Both arms scored it high, about 0.72 and 0.75. That score mostly comes from the real opening, which was already sharp. Our agents then calmed down. They did not add a new attack.

The next two rows are civil conversations. Both arms scored them zero. That is what we want.

The fourth row is the important miss. The real conversation did have an attack, but both arms scored zero. Our agents never escalated, so the judge had nothing to flag.

The last row is a civil conversation with a mild false alarm. Static scored 0.35 and dynamic scored 0.20, so the dynamic arm was a bit calmer, but it was still not a clean zero.

Put together, both arms land on the same summary number: an AUC of 0.67. Chance would be 0.50. The gap between dynamic and static is about zero. So trust is moving — in one conversation a listener started wary, then recovered — but on five conversations, changing trust does not beat keeping it fixed. This is a progress check, not a result.

Next we run twenty complete attack-and-civil pairs, and then the full test set. We will also swap which speaker is the forgiving one, because today the person who speaks first is always the one who forgives faster.
